#!/usr/bin/env python3
"""
Test that bike lanes are properly connected to the road network.
Key requirements:
1. Every bike lane should connect to at least one road network node
2. Where bike lanes cross roads, there should be intersection nodes
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

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

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326

from generate_interactive_map import load_data, build_network

def test_lane_road_connections():
    print("=" * 70)
    print("BIKE LANE TO ROAD NETWORK CONNECTION TEST")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    areas, roads, completed, construction, plan, check, wishing = load_data()
    areas_proj = areas.to_crs(TARGET_CRS)
    roads_proj = roads.to_crs(TARGET_CRS)

    # Build network
    print("\n2. Building network...")
    all_lanes = [completed, construction, plan, check, wishing]
    G, node_coords, node_tree, node_ids, edge_geoms = build_network(
        roads_proj, all_lanes, areas_proj
    )

    transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)

    # Get all road geometries for intersection testing
    road_geoms = []
    for _, row in roads_proj.iterrows():
        geom = row.geometry
        if geom and not geom.is_empty and geom.geom_type == 'LineString':
            road_geoms.append(geom)

    road_tree = STRtree(road_geoms) if road_geoms else None
    print(f"   Roads: {len(road_geoms)}")

    # Build coords array for node queries
    coords_arr = np.array([node_coords[n] for n in node_ids])

    # Test each bike lane layer
    print("\n3. Testing bike lane connections to road network...")

    layer_data = [
        ('completed', completed),
        ('construction', construction),
        ('plan', plan),
        ('check', check),
        ('wishing', wishing)
    ]

    all_issues = []
    crossing_issues = []

    for layer_name, gdf in layer_data:
        if len(gdf) == 0:
            continue

        gdf_proj = gdf.to_crs(TARGET_CRS)
        layer_issues = []
        layer_crossings = []

        for idx, row in gdf_proj.iterrows():
            geom = row.geometry
            lane_name = row.get('Name', f'{layer_name}_{idx}')

            if geom is None or geom.is_empty:
                continue

            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)

            for line_idx, line in enumerate(lines):
                if line.length < 10:  # Skip very short segments
                    continue

                # Check 1: Does this lane have ANY connection to the road network?
                # Sample points along the lane and check for nearby nodes
                CONNECTION_THRESHOLD = 30  # meters - lane should be within 30m of a node

                num_samples = max(5, int(line.length / 50))
                min_node_dist = float('inf')
                closest_node_loc = None

                for i in range(num_samples):
                    t = i / (num_samples - 1) if num_samples > 1 else 0
                    pt = line.interpolate(t, normalized=True)
                    dist, idx_node = node_tree.query([pt.x, pt.y])
                    if dist < min_node_dist:
                        min_node_dist = dist
                        closest_node_loc = (pt.x, pt.y)

                # Also check endpoints specifically
                start = Point(line.coords[0])
                end = Point(line.coords[-1])
                start_dist, _ = node_tree.query([start.x, start.y])
                end_dist, _ = node_tree.query([end.x, end.y])

                min_endpoint_dist = min(start_dist, end_dist)
                min_node_dist = min(min_node_dist, min_endpoint_dist)

                if min_node_dist > CONNECTION_THRESHOLD:
                    center = line.interpolate(0.5, normalized=True)
                    lon, lat = transformer.transform(center.x, center.y)
                    layer_issues.append({
                        'name': lane_name,
                        'length': line.length,
                        'min_node_dist': min_node_dist,
                        'start_dist': start_dist,
                        'end_dist': end_dist,
                        'lat': lat,
                        'lon': lon
                    })

                # Check 2: Does this lane cross any roads without an intersection node?
                if road_tree:
                    # Find roads that this lane intersects
                    lane_buffer = line.buffer(5)  # 5m buffer for intersection detection
                    for road_idx in road_tree.query(lane_buffer):
                        road = road_geoms[road_idx]
                        try:
                            intersection = line.intersection(road)
                            if intersection.is_empty:
                                continue

                            # Get intersection point(s)
                            if intersection.geom_type == 'Point':
                                int_points = [intersection]
                            elif intersection.geom_type == 'MultiPoint':
                                int_points = list(intersection.geoms)
                            elif intersection.geom_type == 'LineString':
                                # They overlap - check endpoints
                                int_points = [Point(intersection.coords[0]), Point(intersection.coords[-1])]
                            else:
                                continue

                            for int_pt in int_points:
                                # Check if there's a network node near this intersection
                                dist, _ = node_tree.query([int_pt.x, int_pt.y])
                                if dist > 20:  # No node within 20m of intersection
                                    lon, lat = transformer.transform(int_pt.x, int_pt.y)
                                    layer_crossings.append({
                                        'lane_name': lane_name,
                                        'node_dist': dist,
                                        'lat': lat,
                                        'lon': lon
                                    })
                        except:
                            pass

        if layer_issues:
            all_issues.extend([(layer_name, i) for i in layer_issues])
        if layer_crossings:
            crossing_issues.extend([(layer_name, c) for c in layer_crossings])

    # Report disconnected lanes
    print("\n" + "=" * 70)
    print("LANES NOT CONNECTED TO ROAD NETWORK (>30m from any node)")
    print("=" * 70)

    if all_issues:
        # Group by layer
        for layer_name in ['completed', 'construction', 'plan', 'check', 'wishing']:
            layer_items = [(l, i) for l, i in all_issues if l == layer_name]
            if layer_items:
                print(f"\n{layer_name.upper()} ({len(layer_items)} issues):")
                # Sort by distance
                layer_items.sort(key=lambda x: x[1]['min_node_dist'], reverse=True)
                for _, issue in layer_items[:15]:
                    print(f"\n  {issue['name']} ({issue['length']:.0f}m)")
                    print(f"    Closest node: {issue['min_node_dist']:.0f}m away")
                    print(f"    Start: {issue['start_dist']:.0f}m, End: {issue['end_dist']:.0f}m from nodes")
                    print(f"    Location: {issue['lat']:.5f}, {issue['lon']:.5f}")
                if len(layer_items) > 15:
                    print(f"\n  ... and {len(layer_items) - 15} more")
    else:
        print("\nAll bike lanes are connected to the road network!")

    # Report road crossings without nodes
    print("\n" + "=" * 70)
    print("ROAD CROSSINGS WITHOUT NETWORK NODES (>20m from intersection)")
    print("=" * 70)

    if crossing_issues:
        # Deduplicate by location (round to 10m grid)
        unique_crossings = {}
        for layer, crossing in crossing_issues:
            key = (round(crossing['lat'], 4), round(crossing['lon'], 4))
            if key not in unique_crossings or crossing['node_dist'] > unique_crossings[key]['node_dist']:
                unique_crossings[key] = {**crossing, 'layer': layer}

        print(f"\nFound {len(unique_crossings)} unique crossing issues:")
        sorted_crossings = sorted(unique_crossings.values(), key=lambda x: x['node_dist'], reverse=True)
        for crossing in sorted_crossings[:30]:
            print(f"  {crossing['lane_name'][:40]} ({crossing['layer']})")
            print(f"    Node distance: {crossing['node_dist']:.0f}m at ({crossing['lat']:.5f}, {crossing['lon']:.5f})")
    else:
        print("\nAll road crossings have network nodes!")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_disconnected = len(all_issues)
    total_crossing_issues = len(set((round(c[1]['lat'], 4), round(c[1]['lon'], 4)) for c in crossing_issues))

    print(f"\nDisconnected bike lanes: {total_disconnected}")
    print(f"Road crossings without nodes: {total_crossing_issues}")

    if total_disconnected > 0 or total_crossing_issues > 0:
        print("\n*** ISSUES FOUND - Some bike lanes may not be usable for routing ***")
    else:
        print("\n*** ALL GOOD - All bike lanes are connected to the road network ***")

    return all_issues, crossing_issues


if __name__ == "__main__":
    issues, crossings = test_lane_road_connections()
