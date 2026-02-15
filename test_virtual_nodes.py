#!/usr/bin/env python3
"""
Test virtual node creation and connectivity to the road network.
Focus on places outside the road network that need virtual connections.
"""
import geopandas as gpd
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from pyproj import Transformer
import fiona
import warnings
import json

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326
NODE_TOLERANCE = 15
VIRTUAL_EDGE_THRESHOLD = 50

def load_all_data():
    """Load all data files."""
    areas = gpd.read_file(script_dir / "jer_areas.shp")
    areas = areas[areas['in_jeru'] == 1].copy()
    roads = gpd.read_file(script_dir / "jerusalem_roads.kml", driver='KML')

    layers = {}
    layer_files = {
        'completed': 'bike_lanes_completed.kml',
        'construction': 'bike_lanes_construction.kml',
        'plan': 'bike_lanes_plan.kml',
        'check': 'bike_lanes_check.kml',
        'wishing': 'bike_lanes_wishing_list.kml'
    }

    for name, filename in layer_files.items():
        try:
            gdf = gpd.read_file(script_dir / filename, driver='KML', on_invalid='ignore')
            gdf = gdf[gdf.geometry.notnull()].copy()
            layers[name] = gdf
        except:
            layers[name] = gpd.GeoDataFrame(columns=['geometry', 'Name'], geometry='geometry', crs='EPSG:4326')

    return areas, roads, layers


def analyze_virtual_node_needs():
    """Analyze where virtual nodes should be created but aren't."""
    print("=" * 70)
    print("VIRTUAL NODE ANALYSIS")
    print("=" * 70)

    areas, roads, layers = load_all_data()
    roads_proj = roads.to_crs(TARGET_CRS)
    areas_proj = areas.to_crs(TARGET_CRS)

    # Build road network
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]
    G = nx.Graph()

    def get_or_create_node(x, y):
        key = (round(x / NODE_TOLERANCE) * NODE_TOLERANCE, round(y / NODE_TOLERANCE) * NODE_TOLERANCE)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        return nid

    road_geoms = []
    for _, row in roads_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) >= 2:
            s = get_or_create_node(coords[0][0], coords[0][1])
            e = get_or_create_node(coords[-1][0], coords[-1][1])
            if s != e:
                G.add_edge(s, e, length=geom.length, has_bike_lane=False)
            road_geoms.append(geom)

    print(f"\n1. Road network: {len(node_coords)} nodes, {G.number_of_edges()} edges")

    # Build spatial index for roads
    road_tree = STRtree(road_geoms) if road_geoms else None

    # Analyze each bike lane layer
    node_ids = list(node_coords.keys())
    coords_arr = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_arr)

    transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)

    print("\n2. Analyzing bike lanes for virtual node opportunities...")

    virtual_node_issues = []
    off_road_segments = []

    for layer_name, gdf in layers.items():
        if len(gdf) == 0:
            continue

        gdf_proj = gdf.to_crs(TARGET_CRS)

        for idx, row in gdf_proj.iterrows():
            geom = row.geometry
            lane_name = row.get('Name', f'{layer_name}_{idx}')

            if geom is None or geom.is_empty:
                continue

            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)

            for line_idx, line in enumerate(lines):
                if line.length < 20:
                    continue

                # Sample points along the line to check road coverage
                num_samples = max(5, int(line.length / 50))  # Sample every ~50m
                samples = []

                for i in range(num_samples):
                    t = i / (num_samples - 1) if num_samples > 1 else 0
                    pt = line.interpolate(t, normalized=True)

                    # Find distance to nearest road
                    nearest_idx = road_tree.nearest(pt)
                    nearest_road = road_geoms[nearest_idx]
                    road_dist = pt.distance(nearest_road)

                    # Find distance to nearest node
                    node_dist, _ = node_tree.query([pt.x, pt.y])

                    samples.append({
                        't': t,
                        'road_dist': road_dist,
                        'node_dist': node_dist,
                        'x': pt.x,
                        'y': pt.y
                    })

                # Check for segments that are far from roads
                off_road_samples = [s for s in samples if s['road_dist'] > 30]

                if len(off_road_samples) > len(samples) * 0.3:
                    # More than 30% of the lane is off-road
                    avg_road_dist = np.mean([s['road_dist'] for s in off_road_samples])
                    avg_node_dist = np.mean([s['node_dist'] for s in samples])

                    center = line.interpolate(0.5, normalized=True)
                    lon, lat = transformer.transform(center.x, center.y)

                    off_road_segments.append({
                        'layer': layer_name,
                        'name': lane_name,
                        'length': line.length,
                        'off_road_pct': 100 * len(off_road_samples) / len(samples),
                        'avg_road_dist': avg_road_dist,
                        'avg_node_dist': avg_node_dist,
                        'lat': lat,
                        'lon': lon
                    })

                # Check if virtual edges could be created
                nodes_within_50m = [s for s in samples if s['node_dist'] <= VIRTUAL_EDGE_THRESHOLD]

                if len(nodes_within_50m) < 2 and line.length > 100:
                    # Long segment with few nearby nodes
                    lon, lat = transformer.transform(line.centroid.x, line.centroid.y)
                    virtual_node_issues.append({
                        'layer': layer_name,
                        'name': lane_name,
                        'length': line.length,
                        'nearby_nodes': len(nodes_within_50m),
                        'lat': lat,
                        'lon': lon,
                        'issue': 'Few nodes within 50m for virtual edges'
                    })

    # Report findings
    print(f"\n3. Off-road bike lane segments (>30% away from roads):")
    if off_road_segments:
        off_road_segments.sort(key=lambda x: x['off_road_pct'], reverse=True)
        for layer_name in layers.keys():
            layer_segs = [s for s in off_road_segments if s['layer'] == layer_name]
            if layer_segs:
                print(f"\n   {layer_name.upper()}:")
                for s in layer_segs[:10]:
                    print(f"     {s['name']} ({s['length']:.0f}m)")
                    print(f"       {s['off_road_pct']:.0f}% off-road, avg {s['avg_road_dist']:.0f}m from roads")
                    print(f"       Avg node distance: {s['avg_node_dist']:.0f}m")
                    print(f"       Location: {s['lat']:.5f}, {s['lon']:.5f}")
                if len(layer_segs) > 10:
                    print(f"     ... and {len(layer_segs)-10} more")
    else:
        print("   None found")

    print(f"\n4. Segments needing more virtual node connections:")
    if virtual_node_issues:
        for layer_name in layers.keys():
            layer_issues = [i for i in virtual_node_issues if i['layer'] == layer_name]
            if layer_issues:
                print(f"\n   {layer_name.upper()}:")
                for i in layer_issues[:10]:
                    print(f"     {i['name']} ({i['length']:.0f}m)")
                    print(f"       Only {i['nearby_nodes']} nodes within 50m")
                    print(f"       Location: {i['lat']:.5f}, {i['lon']:.5f}")
                if len(layer_issues) > 10:
                    print(f"     ... and {len(layer_issues)-10} more")
    else:
        print("   None found")

    # 5. Check for areas that need virtual connections
    print(f"\n5. Checking area centroid connections...")

    # Find areas whose centroids are far from any road
    areas_with_issues = []
    for idx, row in areas_proj.iterrows():
        centroid = row.geometry.centroid
        area_name = row.get('name', f'Area {idx}')

        # Find nearest road
        nearest_road = road_geoms[road_tree.nearest(Point(centroid.x, centroid.y))]
        road_dist = Point(centroid.x, centroid.y).distance(nearest_road)

        # Find nearest node
        node_dist, _ = node_tree.query([centroid.x, centroid.y])

        if road_dist > 200:  # More than 200m from any road
            lon, lat = transformer.transform(centroid.x, centroid.y)
            areas_with_issues.append({
                'name': area_name,
                'road_dist': road_dist,
                'node_dist': node_dist,
                'lat': lat,
                'lon': lon
            })

    if areas_with_issues:
        print(f"   {len(areas_with_issues)} areas are far (>200m) from roads:")
        areas_with_issues.sort(key=lambda x: x['road_dist'], reverse=True)
        for a in areas_with_issues[:15]:
            print(f"     {a['name']}: {a['road_dist']:.0f}m from road, {a['node_dist']:.0f}m from node")
            print(f"       Location: {a['lat']:.5f}, {a['lon']:.5f}")
    else:
        print("   All areas are within 200m of roads")

    return {
        'off_road_segments': off_road_segments,
        'virtual_node_issues': virtual_node_issues,
        'areas_with_issues': areas_with_issues
    }


def test_specific_pathfinding_cases():
    """Test pathfinding for specific problematic cases."""
    print("\n" + "=" * 70)
    print("PATHFINDING EDGE CASE TESTS")
    print("=" * 70)

    areas, roads, layers = load_all_data()
    roads_proj = roads.to_crs(TARGET_CRS)
    areas_proj = areas.to_crs(TARGET_CRS)

    # Build full network
    from shapely.ops import substring

    coord_to_node = {}
    node_coords = {}
    node_counter = [0]
    G = nx.Graph()

    def get_or_create_node(x, y):
        key = (round(x / NODE_TOLERANCE) * NODE_TOLERANCE, round(y / NODE_TOLERANCE) * NODE_TOLERANCE)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        return nid

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
                G.add_edge(s, e, length=geom.length, has_bike_lane=False)
                road_geoms.append(geom)
                road_edges.append((min(s, e), max(s, e)))

    road_tree = STRtree(road_geoms) if road_geoms else None
    BUFFER_DIST = 15

    # Mark bike lanes and create virtuals
    def process_layer(gdf):
        if len(gdf) == 0:
            return 0
        gdf_proj = gdf.to_crs(TARGET_CRS)
        virtual_count = 0

        node_ids = list(node_coords.keys())
        coords_arr = np.array([node_coords[n] for n in node_ids])
        tree = cKDTree(coords_arr) if len(coords_arr) > 0 else None

        for _, row in gdf_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)

            for line in lines:
                # Mark roads
                if road_tree:
                    buffered = line.buffer(BUFFER_DIST)
                    for idx in road_tree.query(buffered):
                        road_geom = road_geoms[idx]
                        try:
                            intersection = road_geom.intersection(buffered)
                            if not intersection.is_empty:
                                overlap = intersection.length / road_geom.length if road_geom.length > 0 else 0
                                if overlap > 0.5 or intersection.length > 20:
                                    s, e = road_edges[idx]
                                    if G.has_edge(s, e):
                                        G[s][e]['has_bike_lane'] = True
                        except:
                            pass

                # Create virtual edges
                if tree and line.length >= 10:
                    line_buffer = line.buffer(VIRTUAL_EDGE_THRESHOLD)
                    nearby = []
                    for i, nid in enumerate(node_ids):
                        pt = Point(coords_arr[i])
                        if line_buffer.contains(pt):
                            proj = line.project(pt)
                            dist = pt.distance(line)
                            if dist <= VIRTUAL_EDGE_THRESHOLD:
                                nearby.append({'nid': nid, 'proj': proj})

                    nearby.sort(key=lambda x: x['proj'])
                    for i in range(len(nearby) - 1):
                        n1, n2 = nearby[i]['nid'], nearby[i+1]['nid']
                        edge_len = nearby[i+1]['proj'] - nearby[i]['proj']
                        if edge_len > 5 and not G.has_edge(n1, n2):
                            G.add_edge(n1, n2, length=edge_len, has_bike_lane=True, is_virtual=True)
                            virtual_count += 1

        return virtual_count

    for name, gdf in layers.items():
        v = process_layer(gdf)
        print(f"   {name}: {v} virtual edges created")

    # Connect area centroids
    for _, row in areas_proj.iterrows():
        centroid = row.geometry.centroid
        if road_tree:
            nearest_idx = road_tree.nearest(Point(centroid.x, centroid.y))
            nearest_road = road_geoms[nearest_idx]
            nearest_pt = nearest_road.interpolate(nearest_road.project(Point(centroid.x, centroid.y)))
            road_node = get_or_create_node(nearest_pt.x, nearest_pt.y)
            dist = Point(centroid.x, centroid.y).distance(nearest_pt)
            if dist > NODE_TOLERANCE:
                centroid_node = get_or_create_node(centroid.x, centroid.y)
                if centroid_node != road_node:
                    G.add_edge(centroid_node, road_node, length=dist, has_bike_lane=False)

    print(f"\nNetwork: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Create weighted graph
    Gw = G.copy()
    K = 100
    for u, v in Gw.edges():
        l = Gw[u][v]['length']
        Gw[u][v]['weight'] = l if Gw[u][v].get('has_bike_lane') else l * K

    # Find area centroids
    node_ids = list(node_coords.keys())
    coords_arr = np.array([node_coords[n] for n in node_ids])
    tree = cKDTree(coords_arr)

    area_nodes = {}
    for idx, row in areas_proj.iterrows():
        centroid = row.geometry.centroid
        _, nearest_idx = tree.query([centroid.x, centroid.y])
        area_nodes[idx] = node_ids[nearest_idx]

    # Test specific pathfinding scenarios
    print("\n1. Testing cross-city routes...")
    transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)

    # Define some test origin-destination pairs from different parts of the city
    test_pairs = []

    # Find areas in different parts of the city (based on coordinates)
    area_centers = []
    for idx, row in areas_proj.iterrows():
        centroid = row.geometry.centroid
        area_centers.append({'idx': idx, 'x': centroid.x, 'y': centroid.y, 'name': row.get('name', f'Area {idx}')})

    # Sort by x and y to get spread
    sorted_by_x = sorted(area_centers, key=lambda a: a['x'])
    sorted_by_y = sorted(area_centers, key=lambda a: a['y'])

    # West to East
    test_pairs.append((sorted_by_x[0], sorted_by_x[-1], 'West to East'))
    test_pairs.append((sorted_by_x[len(sorted_by_x)//4], sorted_by_x[3*len(sorted_by_x)//4], 'West-center to East-center'))

    # North to South
    test_pairs.append((sorted_by_y[0], sorted_by_y[-1], 'South to North'))
    test_pairs.append((sorted_by_y[len(sorted_by_y)//4], sorted_by_y[3*len(sorted_by_y)//4], 'South-center to North-center'))

    for origin, dest, desc in test_pairs:
        print(f"\n   {desc}:")
        print(f"     From: {origin['name']}")
        print(f"     To: {dest['name']}")

        source = area_nodes.get(origin['idx'])
        target = area_nodes.get(dest['idx'])

        if source is None or target is None:
            print("     SKIP: Could not find nodes")
            continue

        try:
            path = nx.dijkstra_path(Gw, source, target, weight='weight')
            path_length = nx.dijkstra_path_length(Gw, source, target, weight='weight')

            # Calculate actual distance and bike %
            actual_dist = 0
            bike_dist = 0
            virtual_dist = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_len = G[u][v]['length']
                actual_dist += edge_len
                if G[u][v].get('has_bike_lane'):
                    bike_dist += edge_len
                if G[u][v].get('is_virtual'):
                    virtual_dist += edge_len

            bike_pct = 100 * bike_dist / actual_dist if actual_dist > 0 else 0

            print(f"     Distance: {actual_dist/1000:.2f} km")
            print(f"     Bike lane: {bike_pct:.1f}%")
            print(f"     Virtual edges: {virtual_dist/1000:.2f} km ({100*virtual_dist/actual_dist:.1f}%)")
            print(f"     Path nodes: {len(path)}")

        except nx.NetworkXNoPath:
            print("     ERROR: No path found!")

    # 2. Test routes through off-road bike paths
    print("\n2. Testing if off-road bike paths are being used...")

    # Find virtual edges and check if they're used
    virtual_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('is_virtual')]
    print(f"   Total virtual edges: {len(virtual_edges)}")

    if virtual_edges:
        # Pick a random virtual edge and test if a path goes through it
        test_edge = virtual_edges[len(virtual_edges)//2]
        u, v = test_edge

        # Find areas near each endpoint
        u_coord = node_coords[u]
        v_coord = node_coords[v]

        near_u = []
        near_v = []
        for idx, row in areas_proj.iterrows():
            centroid = row.geometry.centroid
            dist_u = np.sqrt((centroid.x - u_coord[0])**2 + (centroid.y - u_coord[1])**2)
            dist_v = np.sqrt((centroid.x - v_coord[0])**2 + (centroid.y - v_coord[1])**2)
            if dist_u < 1000:
                near_u.append((idx, dist_u, row.get('name', f'Area {idx}')))
            if dist_v < 1000:
                near_v.append((idx, dist_v, row.get('name', f'Area {idx}')))

        near_u.sort(key=lambda x: x[1])
        near_v.sort(key=lambda x: x[1])

        if near_u and near_v:
            origin = near_u[0]
            dest = near_v[-1] if len(near_v) > 1 else near_v[0]

            print(f"\n   Testing path that should use virtual edge:")
            print(f"   From: {origin[2]} -> To: {dest[2]}")

            source = area_nodes.get(origin[0])
            target = area_nodes.get(dest[0])

            if source and target:
                try:
                    path = nx.dijkstra_path(Gw, source, target, weight='weight')

                    # Check if virtual edge is used
                    used_virtual = False
                    for i in range(len(path) - 1):
                        if G[path[i]][path[i+1]].get('is_virtual'):
                            used_virtual = True
                            break

                    print(f"   Virtual edges used: {'Yes' if used_virtual else 'No'}")

                except nx.NetworkXNoPath:
                    print("   ERROR: No path found")

    return G, node_coords, area_nodes


if __name__ == "__main__":
    results = analyze_virtual_node_needs()
    G, node_coords, area_nodes = test_specific_pathfinding_cases()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Off-road segments: {len(results['off_road_segments'])}")
    print(f"Virtual node issues: {len(results['virtual_node_issues'])}")
    print(f"Areas far from roads: {len(results['areas_with_issues'])}")
