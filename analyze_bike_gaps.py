#!/usr/bin/env python3
"""
Detailed analysis of bike network gaps and missing connections.
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
from pyproj import Transformer
import fiona
import warnings
import json

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326

from generate_interactive_map import load_data, build_network

def analyze_gaps():
    print("=" * 70)
    print("DETAILED BIKE NETWORK GAP ANALYSIS")
    print("=" * 70)

    # Load and build network
    print("\n1. Building network...")
    areas, roads, completed, construction, plan, check, wishing = load_data()
    areas_proj = areas.to_crs(TARGET_CRS)
    roads_proj = roads.to_crs(TARGET_CRS)

    all_lanes = [completed, construction, plan, check, wishing]
    G, node_coords, node_tree, node_ids, edge_geoms = build_network(
        roads_proj, all_lanes, areas_proj
    )

    # Extract bike-only network
    bike_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('has_bike_lane')]
    G_bike = nx.Graph()
    for u, v, d in bike_edges:
        G_bike.add_edge(u, v, **d)

    components = list(nx.connected_components(G_bike))
    components = sorted(components, key=len, reverse=True)

    print(f"   Total bike edges: {len(bike_edges)}")
    print(f"   Bike components: {len(components)}")
    print(f"   Largest: {len(components[0])} nodes")

    # Analyze gaps between components
    print("\n2. Finding gaps between bike network components...")
    transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)

    gaps = []

    # Build KD-tree for main component
    main_comp = components[0]
    main_nodes = list(main_comp)
    main_coords = np.array([node_coords[n] for n in main_nodes])
    main_tree = cKDTree(main_coords)

    for i, comp in enumerate(components[1:], start=2):
        comp_nodes = list(comp)
        comp_coords = np.array([node_coords[n] for n in comp_nodes])

        # Find minimum distance to main component
        min_dist = float('inf')
        best_pair = None

        for j, node in enumerate(comp_nodes):
            coord = comp_coords[j]
            dist, idx = main_tree.query(coord)
            if dist < min_dist:
                min_dist = dist
                best_pair = (main_nodes[idx], node)

        if best_pair and min_dist < 1000:  # Within 1km
            n1, n2 = best_pair
            c1 = node_coords[n1]
            c2 = node_coords[n2]
            lon1, lat1 = transformer.transform(c1[0], c1[1])
            lon2, lat2 = transformer.transform(c2[0], c2[1])

            # Check if there's a road edge between them
            has_road_connection = G.has_edge(n1, n2)

            gaps.append({
                'comp_num': i,
                'comp_size': len(comp),
                'distance': min_dist,
                'node1': n1, 'node2': n2,
                'point1': (lat1, lon1),
                'point2': (lat2, lon2),
                'midpoint': ((lat1+lat2)/2, (lon1+lon2)/2),
                'has_road': has_road_connection
            })

    # Sort by distance
    gaps.sort(key=lambda x: x['distance'])

    print(f"\n   Found {len(gaps)} gaps to main component (within 1km)")

    # Close gaps that could easily be connected
    close_gaps = [g for g in gaps if g['distance'] < 100]
    medium_gaps = [g for g in gaps if 100 <= g['distance'] < 300]
    far_gaps = [g for g in gaps if g['distance'] >= 300]

    print(f"\n   Close gaps (<100m): {len(close_gaps)}")
    print(f"   Medium gaps (100-300m): {len(medium_gaps)}")
    print(f"   Far gaps (>300m): {len(far_gaps)}")

    # Detail close gaps
    print("\n3. CLOSE GAPS (easy to fix):")
    for g in close_gaps[:15]:
        print(f"\n   Component {g['comp_num']} ({g['comp_size']} nodes)")
        print(f"   Gap distance: {g['distance']:.1f}m")
        print(f"   Road connection exists: {g['has_road']}")
        print(f"   Location 1: {g['point1'][0]:.5f}, {g['point1'][1]:.5f}")
        print(f"   Location 2: {g['point2'][0]:.5f}, {g['point2'][1]:.5f}")

    # Detail medium gaps
    print("\n4. MEDIUM GAPS (potential for new lanes):")
    for g in medium_gaps[:10]:
        print(f"\n   Component {g['comp_num']} ({g['comp_size']} nodes)")
        print(f"   Gap distance: {g['distance']:.1f}m")
        print(f"   Midpoint: {g['midpoint'][0]:.5f}, {g['midpoint'][1]:.5f}")

    # Find gaps between non-main components too
    print("\n5. Gaps between smaller components:")
    inter_gaps = []

    for i in range(1, min(10, len(components))):
        comp_i = components[i]
        coords_i = np.array([node_coords[n] for n in comp_i])
        tree_i = cKDTree(coords_i)

        for j in range(i+1, min(10, len(components))):
            comp_j = components[j]
            coords_j = np.array([node_coords[n] for n in comp_j])

            min_dist = float('inf')
            for k, coord in enumerate(coords_j):
                dist, _ = tree_i.query(coord)
                if dist < min_dist:
                    min_dist = dist

            if min_dist < 200:
                inter_gaps.append({
                    'comp1': i+1,
                    'comp2': j+1,
                    'size1': len(comp_i),
                    'size2': len(comp_j),
                    'distance': min_dist
                })

    inter_gaps.sort(key=lambda x: x['distance'])
    for g in inter_gaps[:10]:
        print(f"   Comp {g['comp1']} ({g['size1']} nodes) <-> Comp {g['comp2']} ({g['size2']} nodes): {g['distance']:.0f}m")

    # 6. Analyze which bike lane layers have the most isolated segments
    print("\n6. Isolated segments by layer:")

    # Re-process to track which layer each edge came from
    layer_names = ['completed', 'construction', 'plan', 'check', 'wishing']
    layer_gdfs = [completed, construction, plan, check, wishing]

    isolated_by_layer = {name: 0 for name in layer_names}

    # For each non-main component, check which layers contributed edges
    for i, comp in enumerate(components[1:], start=2):
        # Find edges in this component
        comp_edges = []
        for n1 in comp:
            for n2 in G_bike.neighbors(n1):
                if n2 in comp:
                    comp_edges.append((min(n1, n2), max(n1, n2)))

        # This is simplified - in reality we'd need to trace back to source layer
        # For now just count isolated components
        if len(comp) < 20:
            # Small isolated component
            pass

    print("   (Need more detailed layer tracking for this analysis)")

    # 7. Create output files
    print("\n7. Creating output files...")

    # GeoJSON of gaps
    gap_features = []
    for g in gaps:
        gap_features.append({
            "type": "Feature",
            "properties": {
                "type": "gap",
                "component": g['comp_num'],
                "comp_size": g['comp_size'],
                "distance": round(g['distance'], 1),
                "has_road": g['has_road']
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [g['point1'][1], g['point1'][0]],
                    [g['point2'][1], g['point2'][0]]
                ]
            }
        })

    with open(script_dir / "bike_gaps_detailed.geojson", 'w') as f:
        json.dump({"type": "FeatureCollection", "features": gap_features}, f, indent=2)

    print(f"   Saved {len(gap_features)} gaps to bike_gaps_detailed.geojson")

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_isolated_nodes = sum(len(c) for c in components[1:])
    main_pct = 100 * len(main_comp) / (len(main_comp) + total_isolated_nodes)

    print(f"\nBike network connectivity:")
    print(f"  - Main component: {len(main_comp)} nodes ({main_pct:.1f}%)")
    print(f"  - Isolated: {total_isolated_nodes} nodes in {len(components)-1} components")

    print(f"\nPotential quick wins (gaps < 100m):")
    for g in close_gaps[:5]:
        print(f"  - {g['distance']:.0f}m gap would connect {g['comp_size']} nodes")

    total_connectable = sum(g['comp_size'] for g in close_gaps)
    print(f"\n  Total easily connectable: {total_connectable} nodes")

    return gaps, components, G, node_coords


if __name__ == "__main__":
    gaps, components, G, node_coords = analyze_gaps()
