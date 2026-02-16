#!/usr/bin/env python3
"""
Diagnose specific gaps in the bike network.
"""
import sys
import os

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
data_dir = script_dir.parent  # Input files and main script are in parent directory
sys.path.insert(0, str(data_dir))  # Add parent to path for imports
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326

from generate_interactive_map import load_data, build_network

def diagnose_specific_gaps():
    print("=" * 70)
    print("DIAGNOSING SPECIFIC BIKE NETWORK GAPS")
    print("=" * 70)

    # Load data
    areas, roads, completed, construction, plan, check, wishing = load_data()
    areas_proj = areas.to_crs(TARGET_CRS)
    roads_proj = roads.to_crs(TARGET_CRS)

    # Build network
    all_lanes = [completed, construction, plan, check, wishing]
    G, node_coords, node_tree, node_ids, edge_geoms = build_network(
        roads_proj, all_lanes, areas_proj
    )

    transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)
    transformer_inv = Transformer.from_crs(WGS84, TARGET_CRS, always_xy=True)

    # Key gap locations (from previous analysis)
    # Component 2 (134 nodes): 31m gap at 31.80769, 35.20448
    gaps_to_check = [
        ("Component 2 (134 nodes)", 31.80769, 35.20448, 31),
        ("Component 4 (29 nodes)", 31.78649, 35.17463, 132),
        ("Component 9 (10 nodes)", 31.77613, 35.20903, 15),
    ]

    # All bike lanes as projected geometries
    all_bike_geoms = []
    all_bike_names = []

    for layer_name, gdf in [('completed', completed), ('construction', construction),
                             ('plan', plan), ('check', check), ('wishing', wishing)]:
        if len(gdf) == 0:
            continue
        gdf_proj = gdf.to_crs(TARGET_CRS)
        for idx, row in gdf_proj.iterrows():
            geom = row.geometry
            name = row.get('Name', f'{layer_name}_{idx}')
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'LineString':
                all_bike_geoms.append(geom)
                all_bike_names.append(f"{layer_name}: {name}")
            elif geom.geom_type == 'MultiLineString':
                for i, line in enumerate(geom.geoms):
                    all_bike_geoms.append(line)
                    all_bike_names.append(f"{layer_name}: {name} (part {i})")

    print(f"\nLoaded {len(all_bike_geoms)} bike lane geometries")

    bike_tree = STRtree(all_bike_geoms)

    # Analyze each gap
    for gap_name, lat, lon, expected_dist in gaps_to_check:
        print(f"\n{'=' * 60}")
        print(f"Checking: {gap_name}")
        print(f"Location: {lat:.5f}, {lon:.5f}")
        print(f"Expected gap: ~{expected_dist}m")
        print("=" * 60)

        # Convert to projected coordinates
        x, y = transformer_inv.transform(lon, lat)
        gap_point = Point(x, y)

        # Find nearby bike lanes
        search_radius = 200  # meters
        buffer = gap_point.buffer(search_radius)

        nearby_lanes = []
        for idx in bike_tree.query(buffer):
            geom = all_bike_geoms[idx]
            name = all_bike_names[idx]
            dist = gap_point.distance(geom)
            if dist < search_radius:
                # Find closest point on the lane
                proj_dist = geom.project(gap_point)
                closest_pt = geom.interpolate(proj_dist)
                lon_c, lat_c = transformer.transform(closest_pt.x, closest_pt.y)

                nearby_lanes.append({
                    'name': name,
                    'distance': dist,
                    'closest_point': (lat_c, lon_c),
                    'length': geom.length,
                    'geom': geom
                })

        nearby_lanes.sort(key=lambda x: x['distance'])

        print(f"\nNearby bike lanes (within {search_radius}m):")
        for lane in nearby_lanes[:10]:
            print(f"  {lane['distance']:.1f}m: {lane['name']}")
            print(f"        Length: {lane['length']:.0f}m")
            print(f"        Closest point: {lane['closest_point'][0]:.5f}, {lane['closest_point'][1]:.5f}")

        # Check if there are lanes on both sides of the gap
        if len(nearby_lanes) >= 2:
            lane1 = nearby_lanes[0]
            lane2 = nearby_lanes[1]

            # Measure distance between the two closest lanes
            min_dist = lane1['geom'].distance(lane2['geom'])
            print(f"\n  Distance between two closest lanes: {min_dist:.1f}m")

            # Get the closest points between the two geometries
            # Find endpoints of each lane
            for i, lane in enumerate([lane1, lane2]):
                geom = lane['geom']
                start = Point(geom.coords[0])
                end = Point(geom.coords[-1])
                start_lon, start_lat = transformer.transform(start.x, start.y)
                end_lon, end_lat = transformer.transform(end.x, end.y)
                print(f"\n  Lane {i+1}: {lane['name'][:50]}")
                print(f"    Start: {start_lat:.5f}, {start_lon:.5f}")
                print(f"    End: {end_lat:.5f}, {end_lon:.5f}")

        # Find nearest network nodes
        print(f"\nNearest network nodes:")
        dists, indices = node_tree.query([x, y], k=5)
        for d, idx in zip(dists, indices):
            nid = node_ids[idx]
            coord = node_coords[nid]
            lon_n, lat_n = transformer.transform(coord[0], coord[1])

            # Check what edges this node has
            edges = list(G.edges(nid, data=True))
            bike_edges = [e for e in edges if e[2].get('has_bike_lane')]

            print(f"  Node {nid}: {d:.1f}m away at ({lat_n:.5f}, {lon_n:.5f})")
            print(f"    Total edges: {len(edges)}, Bike edges: {len(bike_edges)}")

    # Check virtual edge creation
    print("\n" + "=" * 70)
    print("VIRTUAL EDGE ANALYSIS")
    print("=" * 70)

    virtual_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('is_virtual')]
    print(f"\nTotal virtual edges: {len(virtual_edges)}")

    # Group by rough location
    print("\nVirtual edge distribution:")
    areas_with_virtuals = {}
    for u, v, d in virtual_edges:
        mid_x = (node_coords[u][0] + node_coords[v][0]) / 2
        mid_y = (node_coords[u][1] + node_coords[v][1]) / 2
        lon, lat = transformer.transform(mid_x, mid_y)

        # Round to 0.01 degree grid
        grid_key = (round(lat, 2), round(lon, 2))
        if grid_key not in areas_with_virtuals:
            areas_with_virtuals[grid_key] = 0
        areas_with_virtuals[grid_key] += 1

    for (lat, lon), count in sorted(areas_with_virtuals.items(), key=lambda x: -x[1])[:15]:
        print(f"  {lat:.2f}, {lon:.2f}: {count} virtual edges")

    # Check specific issue: bike lanes that should create virtual connections but don't
    print("\n" + "=" * 70)
    print("CHECKING VIRTUAL EDGE THRESHOLD ISSUES")
    print("=" * 70)

    VIRTUAL_EDGE_THRESHOLD = 50  # Same as in main script

    # Find bike lanes with no nearby nodes
    lanes_with_no_nodes = []
    for idx, geom in enumerate(all_bike_geoms):
        if geom.length < 50:  # Skip short segments
            continue

        # Sample points along the line
        num_samples = max(3, int(geom.length / 100))
        nearby_node_count = 0

        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0
            pt = geom.interpolate(t, normalized=True)

            # Find nearest node
            dist, _ = node_tree.query([pt.x, pt.y])
            if dist <= VIRTUAL_EDGE_THRESHOLD:
                nearby_node_count += 1

        if nearby_node_count < 2:
            center = geom.interpolate(0.5, normalized=True)
            lon, lat = transformer.transform(center.x, center.y)
            lanes_with_no_nodes.append({
                'name': all_bike_names[idx],
                'length': geom.length,
                'nearby_nodes': nearby_node_count,
                'lat': lat, 'lon': lon
            })

    print(f"\nBike lanes with <2 nodes within {VIRTUAL_EDGE_THRESHOLD}m:")
    lanes_with_no_nodes.sort(key=lambda x: x['length'], reverse=True)
    for lane in lanes_with_no_nodes[:20]:
        print(f"  {lane['name'][:50]} ({lane['length']:.0f}m)")
        print(f"    Location: {lane['lat']:.5f}, {lane['lon']:.5f}")
        print(f"    Nearby nodes: {lane['nearby_nodes']}")


if __name__ == "__main__":
    diagnose_specific_gaps()
