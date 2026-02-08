"""
Rank wishing list lanes by their individual improvement to total accessibility.
Fixed: Creates nodes at all road intersections, not just lane endpoints.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
from shapely.ops import split, snap
import fiona
import warnings

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
ROADS_FILE = script_dir / "jerusalem_roads.kml"
BIKE_LANES_COMPLETED = script_dir / "bike_lanes_completed.kml"
BIKE_LANES_CONSTRUCTION = script_dir / "bike_lanes_construction.kml"
BIKE_LANES_WISHING_LIST = script_dir / "bike_lanes_wishing_list.kml"
AREAS_FILE = script_dir / "jer_areas.shp"

TARGET_CRS = 2039
NODE_TOLERANCE = 15
K_PENALTY = 100
THETA = -1

fiona.drvsupport.supported_drivers['KML'] = 'rw'


def load_data():
    """Load all required data."""
    print("Loading data...")

    areas = gpd.read_file(AREAS_FILE)
    areas = areas[areas['in_jeru'] == 1].copy()
    areas['pop'] = areas['pop_2025'].fillna(0)
    areas['emp'] = areas['emp_2025'].fillna(0)
    areas = areas.to_crs(TARGET_CRS)

    roads = gpd.read_file(ROADS_FILE, driver='KML').to_crs(TARGET_CRS)
    jeru_bounds = areas.total_bounds
    buffer = 1000
    roads = roads.cx[jeru_bounds[0]-buffer:jeru_bounds[2]+buffer,
                     jeru_bounds[1]-buffer:jeru_bounds[3]+buffer]

    completed = gpd.read_file(BIKE_LANES_COMPLETED, driver='KML')
    construction = gpd.read_file(BIKE_LANES_CONSTRUCTION, driver='KML')
    wishing_list = gpd.read_file(BIKE_LANES_WISHING_LIST, driver='KML')

    print(f"  Areas: {len(areas)}, Roads: {len(roads)}")
    print(f"  Completed: {len(completed)}, Construction: {len(construction)}, Wishing: {len(wishing_list)}")

    return areas, roads, completed, construction, wishing_list


def build_network_with_intersections(roads_gdf, bike_lanes_list, areas_gdf=None, tolerance=15):
    """
    Build network where bike lanes create nodes at all road intersections.
    Also connects area centroids to the nearest roads for proper accessibility.
    """
    from shapely.strtree import STRtree

    G = nx.Graph()
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]

    def get_or_create_node(x, y):
        key = (round(x / tolerance) * tolerance, round(y / tolerance) * tolerance)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        G.add_node(nid, x=x, y=y)
        return nid

    # Collect all road geometries
    road_geoms = []
    for _, row in roads_gdf.iterrows():
        g = row.geometry
        if g and not g.is_empty and g.geom_type == 'LineString':
            road_geoms.append(g)

    # Add road edges
    for geom in road_geoms:
        coords = list(geom.coords)
        if len(coords) >= 2:
            start = get_or_create_node(coords[0][0], coords[0][1])
            end = get_or_create_node(coords[-1][0], coords[-1][1])
            if start != end:
                if not G.has_edge(start, end) or G[start][end]['length'] > geom.length:
                    G.add_edge(start, end, length=geom.length, has_bike_lane=False)

    # Create road union for intersection detection
    from shapely.ops import unary_union
    road_union = unary_union(road_geoms)

    # Process bike lanes - find intersections with roads
    for bl_gdf in bike_lanes_list:
        if bl_gdf is None:
            continue
        bl_proj = bl_gdf.to_crs(TARGET_CRS)

        for _, row in bl_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or geom.geom_type != 'LineString':
                continue

            # Find intersection points with road network
            intersection = geom.intersection(road_union)

            # Collect all intersection points
            int_points = []
            if intersection.is_empty:
                # No intersection - just use endpoints
                coords = list(geom.coords)
                int_points = [Point(coords[0]), Point(coords[-1])]
            elif intersection.geom_type == 'Point':
                int_points = [intersection]
            elif intersection.geom_type == 'MultiPoint':
                int_points = list(intersection.geoms)
            elif intersection.geom_type in ['LineString', 'MultiLineString']:
                # Lane overlaps road - get endpoints of overlapping segments
                if intersection.geom_type == 'LineString':
                    segs = [intersection]
                else:
                    segs = list(intersection.geoms)
                for seg in segs:
                    coords = list(seg.coords)
                    int_points.append(Point(coords[0]))
                    int_points.append(Point(coords[-1]))
            elif intersection.geom_type == 'GeometryCollection':
                for g in intersection.geoms:
                    if g.geom_type == 'Point':
                        int_points.append(g)
                    elif g.geom_type == 'LineString':
                        coords = list(g.coords)
                        int_points.append(Point(coords[0]))
                        int_points.append(Point(coords[-1]))

            # Always include lane endpoints
            lane_coords = list(geom.coords)
            int_points.append(Point(lane_coords[0]))
            int_points.append(Point(lane_coords[-1]))

            # Also find nearby road endpoints (for lanes parallel to roads)
            # Use the original road endpoint (not projected) to ensure connection
            NEAR_TOLERANCE = 30
            for road_geom in road_geoms:
                road_coords = list(road_geom.coords)
                for rc in [road_coords[0], road_coords[-1]]:
                    pt = Point(rc[0], rc[1])
                    if geom.distance(pt) < NEAR_TOLERANCE:
                        int_points.append(pt)  # Use original point, not projected

            # Sort points along the lane
            points_with_dist = []
            for pt in int_points:
                d = geom.project(pt)
                points_with_dist.append((d, pt))
            points_with_dist.sort(key=lambda x: x[0])

            # Remove duplicates (points within tolerance)
            unique_points = []
            for d, pt in points_with_dist:
                if not unique_points or d - unique_points[-1][0] > tolerance:
                    unique_points.append((d, pt))

            # Create edges between consecutive points
            for i in range(len(unique_points) - 1):
                d1, pt1 = unique_points[i]
                d2, pt2 = unique_points[i + 1]

                n1 = get_or_create_node(pt1.x, pt1.y)
                n2 = get_or_create_node(pt2.x, pt2.y)

                if n1 != n2:
                    seg_length = d2 - d1
                    if G.has_edge(n1, n2):
                        G[n1][n2]['has_bike_lane'] = True
                        # Keep shorter length
                        G[n1][n2]['length'] = min(G[n1][n2]['length'], seg_length)
                    else:
                        G.add_edge(n1, n2, length=seg_length, has_bike_lane=True)

    # Connect area centroids to the nearest roads
    # This ensures every area has a proper connection to the network
    if areas_gdf is not None and len(road_geoms) > 0:
        road_tree_for_areas = STRtree(road_geoms)

        # Build a mapping from road geom index to its edge nodes
        road_edges_list = []
        for geom in road_geoms:
            coords = list(geom.coords)
            if len(coords) >= 2:
                sx, sy = coords[0][0], coords[0][1]
                ex, ey = coords[-1][0], coords[-1][1]
                s_key = (round(sx / tolerance) * tolerance, round(sy / tolerance) * tolerance)
                e_key = (round(ex / tolerance) * tolerance, round(ey / tolerance) * tolerance)
                s_node = coord_to_node.get(s_key)
                e_node = coord_to_node.get(e_key)
                road_edges_list.append((s_node, e_node))
            else:
                road_edges_list.append((None, None))

        for _, area_row in areas_gdf.iterrows():
            centroid = area_row.geometry.centroid
            centroid_pt = Point(centroid.x, centroid.y)

            # Find nearest road geometry
            nearest_idx = road_tree_for_areas.nearest(centroid_pt)
            nearest_road = road_geoms[nearest_idx]

            # Find the nearest point on that road
            nearest_point_on_road = nearest_road.interpolate(nearest_road.project(centroid_pt))
            dist_to_road = centroid_pt.distance(nearest_point_on_road)

            # Create a node at the nearest point on the road
            road_node = get_or_create_node(nearest_point_on_road.x, nearest_point_on_road.y)

            # Get the edge endpoints for this road
            s, e = road_edges_list[nearest_idx]

            # If the road node is different from both endpoints, we need to split the edge
            if s is not None and e is not None and road_node != s and road_node != e and G.has_edge(s, e):
                old_length = G[s][e]['length']
                old_has_bike = G[s][e].get('has_bike_lane', False)

                # Calculate distances from road node to both endpoints
                s_coord = node_coords[s]
                e_coord = node_coords[e]
                road_node_coord = node_coords[road_node]

                dist_to_s = np.sqrt((road_node_coord[0] - s_coord[0])**2 + (road_node_coord[1] - s_coord[1])**2)
                dist_to_e = np.sqrt((road_node_coord[0] - e_coord[0])**2 + (road_node_coord[1] - e_coord[1])**2)

                # Only split if the new node is meaningfully inside the edge
                if dist_to_s > tolerance and dist_to_e > tolerance:
                    # Remove old edge
                    G.remove_edge(s, e)
                    # Add two new edges
                    G.add_edge(s, road_node, length=dist_to_s, has_bike_lane=old_has_bike)
                    G.add_edge(road_node, e, length=dist_to_e, has_bike_lane=old_has_bike)

            # If centroid is significantly far from the road, add a connector edge
            # This creates a direct path from area to road network
            if dist_to_road > tolerance:
                centroid_node = get_or_create_node(centroid.x, centroid.y)
                if centroid_node != road_node:
                    # Add edge connecting centroid to road network
                    G.add_edge(centroid_node, road_node, length=dist_to_road, has_bike_lane=False)

    # Build node lookup tree
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_array)

    # Ensure all areas are connected to the largest component
    # Some areas might be near disconnected road segments
    if areas_gdf is not None and len(coords_array) > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        # Build a KD-tree of only the nodes in the largest component
        cc_nodes = [n for n in node_ids if n in largest_cc]
        if cc_nodes:
            cc_coords = np.array([node_coords[n] for n in cc_nodes])
            cc_tree = cKDTree(cc_coords)

            for _, area_row in areas_gdf.iterrows():
                centroid = area_row.geometry.centroid
                # Find the nearest node to this centroid
                _, nearest_idx = node_tree.query([centroid.x, centroid.y])
                nearest_node = node_ids[nearest_idx]

                # If it's not in the largest component, connect it
                if nearest_node not in largest_cc:
                    # Find the nearest node in the largest component
                    dist, cc_idx = cc_tree.query([centroid.x, centroid.y])
                    cc_node = cc_nodes[cc_idx]

                    # Add edge from the nearest node to the main network
                    if nearest_node != cc_node:
                        nearest_coord = node_coords[nearest_node]
                        cc_coord = node_coords[cc_node]
                        edge_length = np.sqrt((nearest_coord[0] - cc_coord[0])**2 +
                                             (nearest_coord[1] - cc_coord[1])**2)
                        G.add_edge(nearest_node, cc_node, length=edge_length, has_bike_lane=False)
                        # Update the largest component (now includes the connected nodes)
                        largest_cc = max(nx.connected_components(G), key=len)

    return G, node_coords, node_tree, node_ids


def apply_weights(G, k_penalty):
    G = G.copy()
    for u, v in G.edges():
        length = G[u][v]['length']
        has_bike = G[u][v].get('has_bike_lane', False)
        G[u][v]['weight'] = length if has_bike else length * k_penalty
    return G


def calculate_total_N(G, node_coords, node_tree, node_ids, areas_gdf, theta, k_penalty):
    G = apply_weights(G, k_penalty)
    centroids = areas_gdf.geometry.centroid
    n_areas = len(areas_gdf)

    center_nodes = []
    for centroid in centroids:
        _, idx = node_tree.query([centroid.x, centroid.y])
        center_nodes.append(node_ids[idx])

    pop = areas_gdf['pop'].values
    emp = areas_gdf['emp'].values
    largest_cc = max(nx.connected_components(G), key=len)

    total_N = 0
    for i in range(n_areas):
        if center_nodes[i] not in largest_cc:
            continue
        try:
            distances = nx.single_source_dijkstra_path_length(G, center_nodes[i], weight='weight')
        except:
            continue

        for j in range(n_areas):
            if i != j and center_nodes[j] in distances:
                tau = max(distances[center_nodes[j]] / 1000, 0.1)
                total_N += pop[i] * emp[j] * (tau ** theta)
    return total_N


def main():
    print("=" * 70)
    print("RANKING WISHING LIST LANES (with intersection detection)")
    print(f"K = {K_PENALTY}, theta = {THETA}")
    print("=" * 70)

    areas, roads, completed, construction, wishing_list = load_data()

    print("\nCalculating baseline (without any wishing list lanes)...")
    G_base, node_coords, node_tree, node_ids = build_network_with_intersections(
        roads, [completed, construction], areas_gdf=areas, tolerance=NODE_TOLERANCE
    )
    print(f"  Network: {G_base.number_of_nodes()} nodes, {G_base.number_of_edges()} edges")
    baseline_N = calculate_total_N(G_base, node_coords, node_tree, node_ids, areas, THETA, K_PENALTY)
    print(f"  Baseline N = {baseline_N:.4e}")

    print(f"\nEvaluating {len(wishing_list)} wishing list lanes...")
    results = []

    for i in range(len(wishing_list)):
        lane = wishing_list.iloc[[i]]
        lane_name = lane['Name'].iloc[0] if 'Name' in lane.columns else f"Lane_{i}"
        lane_length = lane.to_crs(TARGET_CRS).geometry.length.iloc[0]

        G_with_lane, nc, nt, ni = build_network_with_intersections(
            roads, [completed, construction, lane], areas_gdf=areas, tolerance=NODE_TOLERANCE
        )
        N_with_lane = calculate_total_N(G_with_lane, nc, nt, ni, areas, THETA, K_PENALTY)

        improvement = N_with_lane - baseline_N
        pct_improvement = 100 * improvement / baseline_N if baseline_N > 0 else 0

        results.append({
            'rank': 0,
            'name': lane_name,
            'length_m': lane_length,
            'N_with_lane': N_with_lane,
            'improvement': improvement,
            'pct_improvement': pct_improvement,
        })

        print(f"  [{i+1:2d}/{len(wishing_list)}] {lane_name[:40]:40s} +{pct_improvement:.3f}%")

    results.sort(key=lambda x: x['improvement'], reverse=True)
    for i, r in enumerate(results):
        r['rank'] = i + 1

    print("\n" + "=" * 70)
    print("RANKING (by total improvement)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Lane Name':<45} {'Length':>8} {'Improvement':>12} {'%':>8}")
    print("-" * 80)

    for r in results:
        print(f"{r['rank']:<5} {r['name'][:44]:<45} {r['length_m']:>7.0f}m {r['improvement']:>12.2e} {r['pct_improvement']:>7.3f}%")

    df = pd.DataFrame(results)
    df.to_csv(script_dir / 'wishing_list_ranking.csv', index=False)
    print(f"\nSaved ranking to wishing_list_ranking.csv")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline N: {baseline_N:.4e}")
    print(f"\nTop 5 lanes:")
    for r in results[:5]:
        print(f"  {r['rank']}. {r['name'][:50]} (+{r['pct_improvement']:.3f}%)")

    total_all = sum(r['improvement'] for r in results)
    print(f"\nSum of individual improvements: {total_all:.2e} ({100*total_all/baseline_N:.2f}%)")


if __name__ == '__main__':
    main()
