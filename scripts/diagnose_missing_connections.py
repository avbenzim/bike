#!/usr/bin/env python3
"""
Diagnose why bike lanes aren't connecting to roads.
Check if roads exist nearby or if the road network is missing.
"""
import sys
import os

import geopandas as gpd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.strtree import STRtree
from pyproj import Transformer
import fiona
import warnings

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
data_dir = script_dir.parent  # Input files are in parent directory
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326

def diagnose():
    print("=" * 70)
    print("DIAGNOSING MISSING BIKE-ROAD CONNECTIONS")
    print("=" * 70)

    # Load raw data
    areas = gpd.read_file(data_dir / "jer_areas.shp")
    areas = areas[areas['in_jeru'] == 1].copy()
    roads = gpd.read_file(data_dir / "jerusalem_roads.kml", driver='KML')
    completed = gpd.read_file(data_dir / "bike_lanes_completed.kml", driver='KML')

    roads_proj = roads.to_crs(TARGET_CRS)
    completed_proj = completed.to_crs(TARGET_CRS)

    transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)
    transformer_inv = Transformer.from_crs(WGS84, TARGET_CRS, always_xy=True)

    # Build road geometry index
    road_geoms = []
    for _, row in roads_proj.iterrows():
        geom = row.geometry
        if geom and not geom.is_empty and geom.geom_type == 'LineString':
            road_geoms.append(geom)

    road_tree = STRtree(road_geoms)
    print(f"Roads loaded: {len(road_geoms)}")

    # Check specific problem locations
    problem_locations = [
        ("Western area 1", 31.78065, 35.15645),  # 656m from node
        ("Western area 2", 31.77645, 35.15816),  # 449m from node
        ("Southwest", 31.75141, 35.15929),       # 259m from node
        ("Park trail", 31.74313, 35.18105),      # Construction 49m from node
        ("Northern plan", 31.86842, 35.20965),   # Plan 197m from node
    ]

    print("\n" + "=" * 70)
    print("CHECKING PROBLEM LOCATIONS")
    print("=" * 70)

    for name, lat, lon in problem_locations:
        x, y = transformer_inv.transform(lon, lat)
        pt = Point(x, y)

        print(f"\n{name} ({lat:.5f}, {lon:.5f}):")

        # Find nearest road
        nearest_idx = road_tree.nearest(pt)
        nearest_road = road_geoms[nearest_idx]
        road_dist = pt.distance(nearest_road)

        print(f"  Nearest road: {road_dist:.0f}m away")

        # Find all roads within 500m
        buffer = pt.buffer(500)
        nearby_roads = []
        for idx in road_tree.query(buffer):
            road = road_geoms[idx]
            dist = pt.distance(road)
            if dist <= 500:
                nearby_roads.append((dist, road.length))

        nearby_roads.sort()
        print(f"  Roads within 500m: {len(nearby_roads)}")
        if nearby_roads:
            print(f"  Closest 5 roads:")
            for dist, length in nearby_roads[:5]:
                print(f"    {dist:.0f}m (road length: {length:.0f}m)")

        # Check if there are ANY roads in this general area
        large_buffer = pt.buffer(1000)
        roads_in_1km = sum(1 for idx in road_tree.query(large_buffer)
                          if pt.distance(road_geoms[idx]) <= 1000)
        print(f"  Roads within 1km: {roads_in_1km}")

    # Check road network coverage
    print("\n" + "=" * 70)
    print("ROAD NETWORK COVERAGE ANALYSIS")
    print("=" * 70)

    # Sample grid across Jerusalem bounds
    areas_proj = areas.to_crs(TARGET_CRS)
    bounds = areas_proj.total_bounds  # minx, miny, maxx, maxy

    grid_size = 500  # 500m grid
    x_points = np.arange(bounds[0], bounds[2], grid_size)
    y_points = np.arange(bounds[1], bounds[3], grid_size)

    no_road_cells = []
    for x in x_points:
        for y in y_points:
            pt = Point(x, y)
            # Check if point is within Jerusalem
            in_jeru = any(areas_proj.geometry.contains(pt))
            if not in_jeru:
                continue

            # Find nearest road
            nearest_idx = road_tree.nearest(pt)
            dist = pt.distance(road_geoms[nearest_idx])

            if dist > 200:  # More than 200m from any road
                lon, lat = transformer.transform(x, y)
                no_road_cells.append({
                    'x': x, 'y': y,
                    'lat': lat, 'lon': lon,
                    'road_dist': dist
                })

    print(f"\nGrid cells (500m) with no road within 200m: {len(no_road_cells)}")
    if no_road_cells:
        no_road_cells.sort(key=lambda c: c['road_dist'], reverse=True)
        print("\nWorst coverage areas:")
        for cell in no_road_cells[:20]:
            print(f"  ({cell['lat']:.4f}, {cell['lon']:.4f}): {cell['road_dist']:.0f}m to nearest road")

    # Check where bike lanes exist but roads don't
    print("\n" + "=" * 70)
    print("BIKE LANES IN AREAS WITHOUT ROADS")
    print("=" * 70)

    bike_in_no_road_areas = []
    for idx, row in completed_proj.iterrows():
        geom = row.geometry
        name = row.get('Name', f'lane_{idx}')
        if geom is None or geom.is_empty:
            continue

        lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
        for line in lines:
            if line.length < 50:
                continue

            center = line.interpolate(0.5, normalized=True)
            nearest_idx = road_tree.nearest(center)
            road_dist = center.distance(road_geoms[nearest_idx])

            if road_dist > 100:  # Bike lane center is >100m from any road
                lon, lat = transformer.transform(center.x, center.y)
                bike_in_no_road_areas.append({
                    'name': name,
                    'length': line.length,
                    'road_dist': road_dist,
                    'lat': lat, 'lon': lon
                })

    bike_in_no_road_areas.sort(key=lambda x: x['road_dist'], reverse=True)
    print(f"\nCompleted bike lanes >100m from any road: {len(bike_in_no_road_areas)}")
    for lane in bike_in_no_road_areas[:20]:
        print(f"  {lane['name'][:40]} ({lane['length']:.0f}m)")
        print(f"    {lane['road_dist']:.0f}m from road at ({lane['lat']:.5f}, {lane['lon']:.5f})")

    # Conclusion
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    if no_road_cells:
        print("\n1. ROAD NETWORK GAPS: The jerusalem_roads.kml is missing roads in some areas")
        print("   These are primarily in western Jerusalem and around parks/trails")

    if bike_in_no_road_areas:
        print("\n2. OFF-ROAD BIKE PATHS: Many bike lanes are dedicated paths not on roads")
        print("   These need special handling (virtual connections or extended threshold)")


if __name__ == "__main__":
    diagnose()
