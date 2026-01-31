"""
Create gap map showing improvement from adding Bar Lev lane only.
Compares: baseline (no wishing list) vs baseline + Bar Lev
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy.spatial import cKDTree
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
SNAP_TOLERANCE = 50
K_PENALTY = 100
THETA = -1

fiona.drvsupport.supported_drivers['KML'] = 'rw'


def build_network(roads_gdf, bike_lanes_list, tolerance=15, snap_tolerance=50):
    """Build NetworkX graph with nearest-neighbor snapping for bike lanes."""
    G = nx.Graph()
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]

    def get_or_create_node_grid(x, y):
        key = (round(x / tolerance) * tolerance, round(y / tolerance) * tolerance)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        G.add_node(nid, x=x, y=y)
        return nid

    # Add road edges
    for idx, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms) if hasattr(geom, 'geoms') else []
        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            start = get_or_create_node_grid(coords[0][0], coords[0][1])
            end = get_or_create_node_grid(coords[-1][0], coords[-1][1])
            if start != end:
                length = line.length
                if not G.has_edge(start, end) or G[start][end]['length'] > length:
                    G.add_edge(start, end, length=length, has_bike_lane=False)

    # Build spatial index for snapping
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_array)

    def get_nearest_node(x, y):
        if len(coords_array) > 0:
            dist, idx = node_tree.query([x, y])
            if dist <= snap_tolerance:
                return node_ids[idx]
        return get_or_create_node_grid(x, y)

    # Add bike lanes with snapping
    for bl_gdf in bike_lanes_list:
        if bl_gdf is None:
            continue
        bl_proj = bl_gdf.to_crs(TARGET_CRS)
        for idx, row in bl_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms) if hasattr(geom, 'geoms') else []
            for line in lines:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue
                start = get_nearest_node(coords[0][0], coords[0][1])
                end = get_nearest_node(coords[-1][0], coords[-1][1])
                if start != end:
                    length = line.length
                    if G.has_edge(start, end):
                        G[start][end]['has_bike_lane'] = True
                    else:
                        G.add_edge(start, end, length=length, has_bike_lane=True)

    # Rebuild index
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_array)

    return G, node_coords, node_tree, node_ids


def apply_weights(G, k_penalty):
    G = G.copy()
    for u, v in G.edges():
        length = G[u][v]['length']
        has_bike = G[u][v].get('has_bike_lane', False)
        G[u][v]['weight'] = length if has_bike else length * k_penalty
    return G


def calculate_N_per_area(G, node_coords, node_tree, node_ids, areas_gdf, theta, k_penalty):
    """Calculate N_i for each origin area."""
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
    N_per_area = np.zeros(n_areas)

    for i in range(n_areas):
        if center_nodes[i] not in largest_cc:
            continue
        try:
            distances = nx.single_source_dijkstra_path_length(G, center_nodes[i], weight='weight')
        except:
            continue

        for j in range(n_areas):
            if i == j:
                continue
            if center_nodes[j] in distances:
                dist = distances[center_nodes[j]]
                tau = max(dist / 1000, 0.1)
                N_per_area[i] += pop[i] * emp[j] * (tau ** theta)

    return N_per_area


def create_bar_lev_gaps_map(areas_gdf, N_without, N_with, bar_lev_lane, all_lanes, output_file):
    """Create gaps map for Bar Lev lane."""
    fig, ax = plt.subplots(figsize=(14, 16))

    areas_plot = areas_gdf.copy()
    improvement = N_with - N_without
    pct_improvement = np.where(N_without > 0, 100 * improvement / N_without, 0)

    areas_plot['improvement'] = improvement
    areas_plot['pct_improvement'] = pct_improvement

    no_change = areas_plot[(areas_plot['improvement'] <= 0) | (N_without == 0)].copy()
    has_improvement = areas_plot[(areas_plot['improvement'] > 0) & (N_without > 0)].copy()

    # Green colormap for improvement
    cmap_improve = LinearSegmentedColormap.from_list('improve',
        ['#FFFFFF', '#90EE90', '#32CD32', '#228B22', '#006400'], N=256)

    if len(no_change) > 0:
        no_change.plot(ax=ax, color='#E0E0E0', edgecolor='#888888', linewidth=0.3)

    if len(has_improvement) > 0:
        has_improvement.plot(column='pct_improvement', ax=ax, cmap=cmap_improve,
                            edgecolor='#444444', linewidth=0.3,
                            legend=True,
                            legend_kwds={'label': 'Improvement (%)',
                                        'orientation': 'vertical',
                                        'shrink': 0.6})

    # Plot existing lanes in gray
    for bl_gdf in all_lanes:
        if bl_gdf is not None and len(bl_gdf) > 0:
            bl_proj = bl_gdf.to_crs(TARGET_CRS)
            bl_proj.plot(ax=ax, color='gray', linewidth=1, alpha=0.5)

    # Highlight Bar Lev in red
    bar_lev_proj = bar_lev_lane.to_crs(TARGET_CRS)
    bar_lev_proj.plot(ax=ax, color='red', linewidth=3, alpha=0.9, label='Bar Lev')

    ax.set_title('Accessibility Improvement from Bar Lev Lane\n(% increase in N per area)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (Israel TM Grid)')
    ax.set_ylabel('Y (Israel TM Grid)')

    total_improvement = N_with.sum() - N_without.sum()
    pct_total = 100 * total_improvement / N_without.sum() if N_without.sum() > 0 else 0
    n_improved = (improvement > 0).sum()
    max_pct = pct_improvement.max()

    stats_text = f'Bar Lev Lane Impact:\nTotal N improvement: {total_improvement:.2e} ({pct_total:.2f}%)\nAreas improved: {n_improved}\nMax improvement: {max_pct:.1f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def main():
    print("=" * 70)
    print("BAR LEV LANE IMPACT ANALYSIS")
    print(f"K = {K_PENALTY}, theta = {THETA}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
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
    wishing = gpd.read_file(BIKE_LANES_WISHING_LIST, driver='KML')

    # Find Bar Lev lane
    bar_lev = wishing[wishing['Name'].str.contains('בר לב', na=False)]
    print(f"  Bar Lev lane found: {len(bar_lev)} segment(s)")
    print(f"  Length: {bar_lev.to_crs(TARGET_CRS).geometry.length.sum():.0f}m")

    # Scenario 1: Without Bar Lev (baseline)
    print("\nScenario 1: Baseline (without Bar Lev)...")
    G1, nc1, nt1, ni1 = build_network(roads, [completed, construction], NODE_TOLERANCE, SNAP_TOLERANCE)
    N_without = calculate_N_per_area(G1, nc1, nt1, ni1, areas, THETA, K_PENALTY)
    print(f"  Total N: {N_without.sum():.4e}")

    # Scenario 2: With Bar Lev
    print("\nScenario 2: With Bar Lev...")
    G2, nc2, nt2, ni2 = build_network(roads, [completed, construction, bar_lev], NODE_TOLERANCE, SNAP_TOLERANCE)
    N_with = calculate_N_per_area(G2, nc2, nt2, ni2, areas, THETA, K_PENALTY)
    print(f"  Total N: {N_with.sum():.4e}")

    # Calculate improvement
    improvement = N_with.sum() - N_without.sum()
    pct = 100 * improvement / N_without.sum()
    print(f"\n  Improvement: {improvement:.2e} ({pct:.2f}%)")

    # Create map
    print("\nCreating gap map...")
    create_bar_lev_gaps_map(areas, N_without, N_with, bar_lev, [completed, construction],
                           str(script_dir / 'bar_lev_gaps.png'))

    print("\nDone!")


if __name__ == '__main__':
    main()
