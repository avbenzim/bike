#!/usr/bin/env python3
"""
Test network connectivity and identify potential issues with bike lane connections.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
from pyproj import Transformer
import fiona
import warnings

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
data_dir = script_dir.parent  # Input files are in parent directory
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326
NODE_TOLERANCE = 15
VIRTUAL_EDGE_THRESHOLD = 50

def load_data():
    """Load all data files."""
    areas = gpd.read_file(data_dir / "jer_areas.shp")
    areas = areas[areas['in_jeru'] == 1].copy()
    areas['pop'] = areas['pop_2025'].fillna(0)
    areas['emp'] = areas['emp_2025'].fillna(0)

    roads = gpd.read_file(data_dir / "jerusalem_roads.kml", driver='KML')
    completed = gpd.read_file(data_dir / "bike_lanes_completed.kml", driver='KML')
    construction = gpd.read_file(data_dir / "bike_lanes_construction.kml", driver='KML')

    try:
        plan = gpd.read_file(data_dir / "bike_lanes_plan.kml", driver='KML', on_invalid='ignore')
        plan = plan[plan.geometry.notnull()].copy()
    except:
        plan = gpd.GeoDataFrame(columns=['geometry', 'Name'], geometry='geometry', crs='EPSG:4326')

    try:
        check = gpd.read_file(data_dir / "bike_lanes_check.kml", driver='KML', on_invalid='ignore')
        check = check[check.geometry.notnull()].copy()
    except:
        check = gpd.GeoDataFrame(columns=['geometry', 'Name'], geometry='geometry', crs='EPSG:4326')

    wishing = gpd.read_file(data_dir / "bike_lanes_wishing_list.kml", driver='KML')

    return areas, roads, completed, construction, plan, check, wishing


def build_network(roads_proj, bike_lanes_list, areas_proj=None, tolerance=NODE_TOLERANCE):
    """Build the network graph."""
    from shapely.strtree import STRtree
    from shapely.ops import substring

    G = nx.Graph()
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]
    edge_to_geom = {}

    def get_or_create_node(x, y):
        key = (round(x / tolerance) * tolerance, round(y / tolerance) * tolerance)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        return nid

    # Add road segments
    road_geoms = []
    road_edges = []
    for _, row in roads_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) >= 2:
            s = get_or_create_node(coords[0][0], coords[0][1])
            e = get_or_create_node(coords[-1][0], coords[-1][1])
            if s != e:
                G.add_edge(s, e, length=geom.length, has_bike_lane=False, source='road')
                edge_key = (min(s, e), max(s, e))
                edge_to_geom[edge_key] = geom
                road_geoms.append(geom)
                road_edges.append(edge_key)

    road_tree_spatial = STRtree(road_geoms) if road_geoms else None
    BUFFER_DIST = 15

    # Track which bike lanes mark which roads
    bike_lane_to_roads = {}  # lane_idx -> list of road edge_keys

    def mark_bike_lane_roads(line_geom, lane_idx=None):
        if road_tree_spatial is None:
            return
        buffered = line_geom.buffer(BUFFER_DIST)
        candidate_indices = road_tree_spatial.query(buffered)
        marked_edges = []
        for idx in candidate_indices:
            road_geom = road_geoms[idx]
            try:
                intersection = road_geom.intersection(buffered)
                if intersection.is_empty:
                    continue
                overlap_ratio = intersection.length / road_geom.length if road_geom.length > 0 else 0
                should_mark = (
                    overlap_ratio > 0.5 or
                    (road_geom.length < 50 and overlap_ratio > 0.3) or
                    intersection.length > 20
                )
                if should_mark:
                    edge_key = road_edges[idx]
                    s, e = edge_key
                    if G.has_edge(s, e):
                        G[s][e]['has_bike_lane'] = True
                        marked_edges.append(edge_key)
            except:
                pass
        if lane_idx is not None:
            bike_lane_to_roads[lane_idx] = marked_edges

    # Process bike lanes
    lane_idx = 0
    for bl_gdf in bike_lanes_list:
        if bl_gdf is None or len(bl_gdf) == 0:
            continue
        bl_proj = bl_gdf.to_crs(TARGET_CRS)
        for _, row in bl_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'LineString':
                mark_bike_lane_roads(geom, lane_idx)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    mark_bike_lane_roads(line, lane_idx)
            lane_idx += 1

    # Create virtual edges
    virtual_edges = []

    def create_virtual_edges_for_lane(line_geom, lane_name="unknown"):
        if len(node_coords) == 0:
            return 0

        current_node_ids = list(node_coords.keys())
        current_coords = np.array([node_coords[n] for n in current_node_ids])
        if len(current_coords) < 2:
            return 0
        current_tree = cKDTree(current_coords)

        line_length = line_geom.length
        if line_length < 10:
            return 0

        nearby_nodes = []
        line_buffer = line_geom.buffer(VIRTUAL_EDGE_THRESHOLD)

        for idx, nid in enumerate(current_node_ids):
            pt = Point(current_coords[idx])
            if line_buffer.contains(pt):
                proj_dist = line_geom.project(pt)
                perp_dist = pt.distance(line_geom)
                if perp_dist <= VIRTUAL_EDGE_THRESHOLD:
                    nearby_nodes.append({
                        'node_id': nid,
                        'proj_dist': proj_dist,
                        'perp_dist': perp_dist
                    })

        nearby_nodes.sort(key=lambda x: x['proj_dist'])

        virtual_edge_count = 0
        for i in range(len(nearby_nodes) - 1):
            n1 = nearby_nodes[i]
            n2 = nearby_nodes[i + 1]
            edge_len = n2['proj_dist'] - n1['proj_dist']

            if edge_len > 5:
                node_a, node_b = n1['node_id'], n2['node_id']
                if not G.has_edge(node_a, node_b):
                    G.add_edge(node_a, node_b, length=edge_len, has_bike_lane=True,
                              is_virtual=True, source=f'virtual_{lane_name}')
                    virtual_edges.append((node_a, node_b, edge_len, lane_name))
                    virtual_edge_count += 1
                elif not G[node_a][node_b].get('has_bike_lane'):
                    G[node_a][node_b]['has_bike_lane'] = True

        return virtual_edge_count, nearby_nodes

    # Track virtual edge creation per lane
    lane_virtual_info = {}
    total_virtual_edges = 0
    lane_idx = 0

    for layer_name, bl_gdf in [('completed', bike_lanes_list[0]),
                                ('construction', bike_lanes_list[1]),
                                ('plan', bike_lanes_list[2]),
                                ('check', bike_lanes_list[3]),
                                ('wishing', bike_lanes_list[4])]:
        if bl_gdf is None or len(bl_gdf) == 0:
            continue
        bl_proj = bl_gdf.to_crs(TARGET_CRS)
        for idx, row in bl_proj.iterrows():
            geom = row.geometry
            lane_name = row.get('Name', f'{layer_name}_{idx}')
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'LineString':
                result = create_virtual_edges_for_lane(geom, lane_name)
                if result:
                    count, nearby = result
                    total_virtual_edges += count
                    lane_virtual_info[lane_name] = {'count': count, 'nearby_nodes': len(nearby), 'layer': layer_name}
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    result = create_virtual_edges_for_lane(line, lane_name)
                    if result:
                        count, nearby = result
                        total_virtual_edges += count
                        if lane_name in lane_virtual_info:
                            lane_virtual_info[lane_name]['count'] += count
                        else:
                            lane_virtual_info[lane_name] = {'count': count, 'nearby_nodes': len(nearby), 'layer': layer_name}
            lane_idx += 1

    # Connect area centroids
    if areas_proj is not None and len(road_geoms) > 0:
        road_tree_for_areas = STRtree(road_geoms)

        for _, area_row in areas_proj.iterrows():
            centroid = area_row.geometry.centroid
            centroid_pt = Point(centroid.x, centroid.y)

            nearest_idx = road_tree_for_areas.nearest(centroid_pt)
            nearest_road = road_geoms[nearest_idx]
            nearest_point_on_road = nearest_road.interpolate(nearest_road.project(centroid_pt))
            dist_to_road = centroid_pt.distance(nearest_point_on_road)

            road_node = get_or_create_node(nearest_point_on_road.x, nearest_point_on_road.y)

            if dist_to_road > tolerance:
                centroid_node = get_or_create_node(centroid.x, centroid.y)
                if centroid_node != road_node:
                    G.add_edge(centroid_node, road_node, length=dist_to_road, has_bike_lane=False, source='area_connector')

    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_array) if len(coords_array) > 0 else None

    # Ensure connectivity to largest component
    if areas_proj is not None and node_tree is not None:
        largest_cc = max(nx.connected_components(G), key=len)
        cc_nodes = [n for n in node_ids if n in largest_cc]
        if cc_nodes:
            cc_coords = np.array([node_coords[n] for n in cc_nodes])
            cc_tree = cKDTree(cc_coords)

            for _, area_row in areas_proj.iterrows():
                centroid = area_row.geometry.centroid
                _, nearest_idx = node_tree.query([centroid.x, centroid.y])
                nearest_node = node_ids[nearest_idx]

                if nearest_node not in largest_cc:
                    dist, cc_idx = cc_tree.query([centroid.x, centroid.y])
                    cc_node = cc_nodes[cc_idx]

                    if nearest_node != cc_node:
                        nearest_coord = node_coords[nearest_node]
                        cc_coord = node_coords[cc_node]
                        edge_length = np.sqrt((nearest_coord[0] - cc_coord[0])**2 +
                                             (nearest_coord[1] - cc_coord[1])**2)
                        G.add_edge(nearest_node, cc_node, length=edge_length, has_bike_lane=False, source='cc_connector')
                        largest_cc = max(nx.connected_components(G), key=len)

    return G, node_coords, node_tree, node_ids, lane_virtual_info, virtual_edges


def test_connectivity():
    """Run connectivity tests on the network."""
    print("=" * 70)
    print("BIKE NETWORK CONNECTIVITY TEST")
    print("=" * 70)

    print("\n1. Loading data...")
    areas, roads, completed, construction, plan, check, wishing = load_data()

    areas_proj = areas.to_crs(TARGET_CRS)
    roads_proj = roads.to_crs(TARGET_CRS)

    print(f"   - Areas: {len(areas)}")
    print(f"   - Roads: {len(roads)}")
    print(f"   - Completed lanes: {len(completed)}")
    print(f"   - Construction lanes: {len(construction)}")
    print(f"   - Plan lanes: {len(plan)}")
    print(f"   - Check lanes: {len(check)}")
    print(f"   - Wishing list lanes: {len(wishing)}")

    print("\n2. Building network...")
    all_lanes = [completed, construction, plan, check, wishing]
    G, node_coords, node_tree, node_ids, lane_virtual_info, virtual_edges = build_network(
        roads_proj, all_lanes, areas_proj
    )

    print(f"   - Total nodes: {G.number_of_nodes()}")
    print(f"   - Total edges: {G.number_of_edges()}")

    # Count edge types
    road_edges = sum(1 for u, v, d in G.edges(data=True) if d.get('source') == 'road')
    virtual_edges_count = sum(1 for u, v, d in G.edges(data=True) if d.get('is_virtual'))
    bike_lane_edges = sum(1 for u, v, d in G.edges(data=True) if d.get('has_bike_lane'))
    connector_edges = sum(1 for u, v, d in G.edges(data=True) if 'connector' in str(d.get('source', '')))

    print(f"   - Road edges: {road_edges}")
    print(f"   - Edges with bike lanes: {bike_lane_edges}")
    print(f"   - Virtual edges: {virtual_edges_count}")
    print(f"   - Connector edges: {connector_edges}")

    # 3. Check connected components
    print("\n3. Analyzing connected components...")
    components = list(nx.connected_components(G))
    print(f"   - Number of components: {len(components)}")

    largest_cc = max(components, key=len)
    print(f"   - Largest component size: {len(largest_cc)} nodes ({100*len(largest_cc)/G.number_of_nodes():.1f}%)")

    if len(components) > 1:
        print("\n   Smaller components:")
        for i, cc in enumerate(sorted(components, key=len, reverse=True)[1:10]):  # Top 10 smaller
            nodes_in_cc = list(cc)
            avg_x = np.mean([node_coords[n][0] for n in nodes_in_cc])
            avg_y = np.mean([node_coords[n][1] for n in nodes_in_cc])
            print(f"     Component {i+2}: {len(cc)} nodes, centroid approx at ({avg_x:.0f}, {avg_y:.0f})")

    # 4. Test area-to-area reachability
    print("\n4. Testing area-to-area reachability...")
    areas_proj['area_id'] = range(len(areas_proj))

    # Find nearest node to each area centroid
    area_nodes = {}
    unreachable_areas = []

    for idx, row in areas_proj.iterrows():
        centroid = row.geometry.centroid
        _, nearest_idx = node_tree.query([centroid.x, centroid.y])
        nearest_node = node_ids[nearest_idx]
        area_nodes[row['area_id']] = nearest_node

        if nearest_node not in largest_cc:
            area_name = row.get('name', f'Area {row["area_id"]}')
            unreachable_areas.append((row['area_id'], area_name))

    if unreachable_areas:
        print(f"\n   WARNING: {len(unreachable_areas)} areas are in disconnected components!")
        for aid, name in unreachable_areas[:10]:
            print(f"     - Area {aid}: {name}")
    else:
        print("   All areas are in the main connected component.")

    # 5. Test path lengths between sample area pairs
    print("\n5. Testing shortest paths between sample area pairs...")

    # Create weighted graph
    Gw = G.copy()
    K = 100  # Penalty for non-bike-lane roads
    for u, v in Gw.edges():
        l = Gw[u][v]['length']
        Gw[u][v]['weight'] = l if Gw[u][v].get('has_bike_lane') else l * K

    # Sample some area pairs
    n_areas = len(areas_proj)
    sample_pairs = []
    np.random.seed(42)

    # Test 20 random pairs
    for _ in range(20):
        i, j = np.random.choice(n_areas, 2, replace=False)
        sample_pairs.append((i, j))

    path_results = []
    for i, j in sample_pairs:
        source = area_nodes[i]
        target = area_nodes[j]

        try:
            path_length = nx.dijkstra_path_length(Gw, source, target, weight='weight')
            path = nx.dijkstra_path(Gw, source, target, weight='weight')

            # Calculate actual distance and bike lane %
            actual_dist = 0
            bike_dist = 0
            for k in range(len(path) - 1):
                u, v = path[k], path[k+1]
                edge_len = G[u][v]['length']
                actual_dist += edge_len
                if G[u][v].get('has_bike_lane'):
                    bike_dist += edge_len

            bike_pct = 100 * bike_dist / actual_dist if actual_dist > 0 else 0
            path_results.append({
                'from': i, 'to': j,
                'weighted_dist': path_length,
                'actual_dist': actual_dist,
                'bike_pct': bike_pct,
                'path_nodes': len(path),
                'reachable': True
            })
        except nx.NetworkXNoPath:
            path_results.append({
                'from': i, 'to': j,
                'weighted_dist': float('inf'),
                'actual_dist': float('inf'),
                'bike_pct': 0,
                'path_nodes': 0,
                'reachable': False
            })

    reachable = [r for r in path_results if r['reachable']]
    unreachable = [r for r in path_results if not r['reachable']]

    print(f"   - Reachable pairs: {len(reachable)}/{len(sample_pairs)}")
    if unreachable:
        print(f"   - UNREACHABLE pairs: {len(unreachable)}")
        for r in unreachable:
            print(f"     Area {r['from']} -> Area {r['to']}: NO PATH")

    if reachable:
        avg_dist = np.mean([r['actual_dist'] for r in reachable])
        avg_bike_pct = np.mean([r['bike_pct'] for r in reachable])
        print(f"   - Average path distance: {avg_dist/1000:.2f} km")
        print(f"   - Average bike lane usage: {avg_bike_pct:.1f}%")

        # Find paths with low bike lane usage
        low_bike_paths = [r for r in reachable if r['bike_pct'] < 10]
        if low_bike_paths:
            print(f"\n   Paths with <10% bike lane coverage ({len(low_bike_paths)} found):")
            for r in low_bike_paths[:5]:
                print(f"     Area {r['from']} -> Area {r['to']}: {r['actual_dist']/1000:.1f}km, {r['bike_pct']:.1f}% bike lane")

    # 6. Analyze virtual edge coverage
    print("\n6. Analyzing virtual edge creation for bike lanes...")

    if lane_virtual_info:
        # Lanes with good virtual coverage
        good_coverage = [(name, info) for name, info in lane_virtual_info.items() if info['count'] >= 2]
        no_virtuals = [(name, info) for name, info in lane_virtual_info.items() if info['count'] == 0]

        print(f"   - Lanes with virtual edges: {len(good_coverage)}")
        print(f"   - Lanes WITHOUT virtual edges: {len(no_virtuals)}")

        if no_virtuals:
            print("\n   Lanes with no virtual connections (may be isolated):")
            for name, info in no_virtuals[:15]:
                print(f"     - {name} ({info['layer']}): {info['nearby_nodes']} nearby nodes but 0 virtual edges")
    else:
        print("   No virtual edge info available")

    # 7. Test specific bike lane connectivity
    print("\n7. Checking if new bike lanes improve connectivity...")

    # Build network WITHOUT bike lanes
    G_roads_only, nc_roads, nt_roads, ni_roads, _, _ = build_network(
        roads_proj, [gpd.GeoDataFrame() for _ in range(5)], areas_proj
    )

    # Compare
    print(f"   - Roads only: {G_roads_only.number_of_edges()} edges")
    print(f"   - With bike lanes: {G.number_of_edges()} edges")
    print(f"   - Additional edges from bike lanes: {G.number_of_edges() - G_roads_only.number_of_edges()}")

    # 8. Find potential disconnected bike lane segments
    print("\n8. Identifying potentially isolated bike lane segments...")

    # Create a subgraph of only bike lane edges
    bike_only_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('has_bike_lane')]
    G_bike = G.edge_subgraph(bike_only_edges).copy()

    bike_components = list(nx.connected_components(G_bike))
    print(f"   - Bike lane network has {len(bike_components)} separate components")

    if len(bike_components) > 1:
        sorted_components = sorted(bike_components, key=len, reverse=True)
        print(f"   - Largest bike component: {len(sorted_components[0])} nodes")

        # Find small isolated bike lane components
        small_components = [c for c in sorted_components if len(c) < 10]
        print(f"   - Small isolated components (<10 nodes): {len(small_components)}")

        # Check if these small components connect to the main road network
        main_bike_cc = sorted_components[0]
        isolated_from_main = []

        for cc in small_components[:10]:
            cc_nodes = list(cc)
            # Check if any node in this component connects to the main bike network via roads
            connects_to_main = False
            for node in cc_nodes:
                for neighbor in G.neighbors(node):
                    if neighbor in main_bike_cc:
                        connects_to_main = True
                        break
                if connects_to_main:
                    break

            if not connects_to_main:
                # Get approximate location
                avg_x = np.mean([node_coords[n][0] for n in cc_nodes])
                avg_y = np.mean([node_coords[n][1] for n in cc_nodes])
                isolated_from_main.append((len(cc), avg_x, avg_y))

        if isolated_from_main:
            print(f"\n   Bike lane segments NOT connected to main bike network:")
            transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)
            for size, x, y in isolated_from_main[:10]:
                lon, lat = transformer.transform(x, y)
                print(f"     - {size} nodes at approx ({lat:.5f}, {lon:.5f})")

    # 9. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    issues = []
    if len(components) > 1:
        issues.append(f"Network has {len(components)} disconnected components")
    if unreachable_areas:
        issues.append(f"{len(unreachable_areas)} areas are unreachable")
    if unreachable:
        issues.append(f"{len(unreachable)} test paths failed (no path found)")
    if len(bike_components) > 5:
        issues.append(f"Bike network is fragmented into {len(bike_components)} pieces")

    if issues:
        print("\nPOTENTIAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo major connectivity issues found!")

    print(f"\nNetwork statistics:")
    print(f"  - {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  - {bike_lane_edges} edges have bike lanes ({100*bike_lane_edges/G.number_of_edges():.1f}%)")
    print(f"  - {virtual_edges_count} virtual edges created for off-road bike paths")

    return G, node_coords, areas_proj, area_nodes


if __name__ == "__main__":
    test_connectivity()
