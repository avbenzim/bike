"""
Generate accessibility heatmaps:
1. Without wishing list lanes
2. With wishing list lanes
3. Gaps/improvement between the two scenarios
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pathlib import Path
from shapely.geometry import LineString, Point
from scipy.spatial import cKDTree
import fiona
import warnings

warnings.filterwarnings('ignore')

# Configuration
script_dir = Path(__file__).parent
ROADS_FILE = script_dir / "jerusalem_roads.kml"  # Use new road network
BIKE_LANES_COMPLETED = script_dir / "bike_lanes_completed.kml"
BIKE_LANES_CONSTRUCTION = script_dir / "bike_lanes_construction.kml"
BIKE_LANES_WISHING_LIST = script_dir / "bike_lanes_wishing_list.kml"
AREAS_FILE = script_dir / "jer_areas.shp"

# Network parameters
TARGET_CRS = 2039  # Israel TM Grid (meters)
NODE_TOLERANCE = 15  # meters for node clustering

# Model parameters
K_PENALTY = 10      # Roads without bike lanes are K times slower
THETA = -1          # Distance decay parameter

# Enable KML driver
fiona.drvsupport.supported_drivers['KML'] = 'rw'


def load_roads_kml(roads_file, areas_gdf):
    """Load road network from KML file."""
    print("Loading road network from KML...")

    roads_gdf = gpd.read_file(roads_file, driver='KML')
    roads_gdf = roads_gdf.to_crs(TARGET_CRS)

    # Filter to Jerusalem area
    jeru_bounds = areas_gdf.total_bounds
    buffer = 1000
    roads_gdf = roads_gdf.cx[jeru_bounds[0]-buffer:jeru_bounds[2]+buffer,
                             jeru_bounds[1]-buffer:jeru_bounds[3]+buffer]

    print(f"  Roads loaded: {len(roads_gdf)}")
    return roads_gdf


def load_bike_lanes():
    """Load all bike lane files."""
    print("Loading bike lanes...")

    completed = gpd.read_file(BIKE_LANES_COMPLETED, driver='KML')
    construction = gpd.read_file(BIKE_LANES_CONSTRUCTION, driver='KML')
    wishing_list = gpd.read_file(BIKE_LANES_WISHING_LIST, driver='KML')

    print(f"  Completed: {len(completed)}")
    print(f"  Construction: {len(construction)}")
    print(f"  Wishing list: {len(wishing_list)}")

    return completed, construction, wishing_list


def load_areas():
    """Load Jerusalem areas with population and employment data."""
    print("Loading areas...")

    areas = gpd.read_file(AREAS_FILE)
    areas = areas[areas['in_jeru'] == 1].copy()
    areas['pop'] = areas['pop_2025'].fillna(0)
    areas['emp'] = areas['emp_2025'].fillna(0)

    print(f"  Areas: {len(areas)}")
    print(f"  Total population: {areas['pop'].sum():,.0f}")
    print(f"  Total employment: {areas['emp'].sum():,.0f}")

    return areas


def build_network(roads_gdf, bike_lanes_list, tolerance=15):
    """Build NetworkX graph from roads with bike lanes marked."""
    print("Building network graph...")

    G = nx.Graph()
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]

    def get_or_create_node(x, y):
        """Get existing node or create new one using grid-based clustering."""
        # Round to tolerance grid
        key = (round(x / tolerance) * tolerance, round(y / tolerance) * tolerance)
        if key in coord_to_node:
            return coord_to_node[key]
        # Create new node
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
            start = get_or_create_node(coords[0][0], coords[0][1])
            end = get_or_create_node(coords[-1][0], coords[-1][1])
            if start != end:
                length = line.length
                if not G.has_edge(start, end) or G[start][end]['length'] > length:
                    G.add_edge(start, end, length=length, has_bike_lane=False)

    # Add bike lane edges and mark existing edges
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
                start = get_or_create_node(coords[0][0], coords[0][1])
                end = get_or_create_node(coords[-1][0], coords[-1][1])
                if start != end:
                    length = line.length
                    if G.has_edge(start, end):
                        G[start][end]['has_bike_lane'] = True
                    else:
                        G.add_edge(start, end, length=length, has_bike_lane=True)

    # Build node lookup tree
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_array)

    # Check connectivity
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"  Connected components: {len(components)}, largest: {len(largest_cc)} nodes")

    return G, node_coords, node_tree, node_ids


def apply_weights(G, k_penalty):
    """Apply K penalty to roads without bike lanes."""
    for u, v in G.edges():
        length = G[u][v]['length']
        has_bike = G[u][v].get('has_bike_lane', False)
        G[u][v]['weight'] = length if has_bike else length * k_penalty
    return G


def calculate_accessibility_per_area(G, node_coords, node_tree, node_ids, areas_gdf, theta, k_penalty):
    """Calculate accessibility N_i = sum_j(pop_i * emp_j * tau_ij^theta) for each origin area."""

    G = apply_weights(G, k_penalty)

    centroids = areas_gdf.geometry.centroid
    n_areas = len(areas_gdf)

    # Find nearest node for each centroid
    center_nodes = []
    for centroid in centroids:
        _, idx = node_tree.query([centroid.x, centroid.y])
        center_nodes.append(node_ids[idx])

    pop = areas_gdf['pop'].values
    emp = areas_gdf['emp'].values

    # Calculate N_i for each origin
    N_per_area = np.zeros(n_areas)
    connected_count = 0

    # Get largest connected component
    largest_cc = max(nx.connected_components(G), key=len)

    for i in range(n_areas):
        if center_nodes[i] not in largest_cc:
            N_per_area[i] = 0
            continue

        connected_count += 1

        # Calculate shortest paths from this origin
        try:
            distances = nx.single_source_dijkstra_path_length(G, center_nodes[i], weight='weight')
        except:
            continue

        # Sum over all destinations
        N_i = 0
        for j in range(n_areas):
            if i == j:
                continue
            if center_nodes[j] in distances:
                dist = distances[center_nodes[j]]
                tau = max(dist / 1000, 0.1)  # km, min 100m
                N_ij = pop[i] * emp[j] * (tau ** theta)
                N_i += N_ij

        N_per_area[i] = N_i

    total_N = N_per_area.sum()

    return N_per_area, total_N, connected_count


def create_heatmap(areas_gdf, N_values, bike_lanes, title, output_file, scenario_label):
    """Create a heatmap showing accessibility values per area."""

    fig, ax = plt.subplots(figsize=(14, 16))

    # Prepare data
    areas_plot = areas_gdf.copy()
    areas_plot['N'] = N_values
    areas_plot['N_log'] = np.log10(np.maximum(N_values, 1))

    # Separate connected and disconnected
    connected = areas_plot[areas_plot['N'] > 0].copy()
    disconnected = areas_plot[areas_plot['N'] == 0].copy()

    # Create custom colormap matching reference image (spectral: red -> orange -> yellow -> green -> cyan -> blue)
    # Red = high accessibility, Blue = low accessibility
    cmap_custom = LinearSegmentedColormap.from_list('spectral_acc',
        ['#0000CD', '#1E90FF', '#00CED1', '#90EE90', '#FFFF00', '#FFA500', '#FF4500', '#DC143C'], N=256)

    # Plot disconnected areas in gray
    if len(disconnected) > 0:
        disconnected.plot(ax=ax, color='#BEBEBE', edgecolor='#888888', linewidth=0.3)

    # Plot connected areas with color
    if len(connected) > 0:
        vmin = connected['N_log'].min()
        vmax = connected['N_log'].max()
        connected.plot(column='N_log', ax=ax, cmap=cmap_custom,
                      edgecolor='#444444', linewidth=0.3,
                      legend=True,
                      legend_kwds={'label': f'Accessibility (log₁₀ N)\n{scenario_label}',
                                  'orientation': 'vertical',
                                  'shrink': 0.6})

    # Plot bike lanes in black
    for bl_gdf in bike_lanes:
        if bl_gdf is not None and len(bl_gdf) > 0:
            bl_proj = bl_gdf.to_crs(TARGET_CRS)
            bl_proj.plot(ax=ax, color='black', linewidth=1.5, alpha=0.8)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (Israel TM Grid)')
    ax.set_ylabel('Y (Israel TM Grid)')

    # Add stats
    total_N = N_values.sum()
    n_connected = (N_values > 0).sum()
    n_disconnected = (N_values == 0).sum()

    stats_text = f'Total N = {total_N:.2e}\nConnected: {n_connected}, Disconnected: {n_disconnected}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save PNG only
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
    return total_N


def create_gaps_heatmap(areas_gdf, N_without, N_with, bike_lanes_all, output_file):
    """Create a heatmap showing the improvement (gaps) between scenarios."""

    fig, ax = plt.subplots(figsize=(14, 16))

    # Calculate improvement
    areas_plot = areas_gdf.copy()
    improvement = N_with - N_without
    pct_improvement = np.where(N_without > 0, 100 * improvement / N_without, 0)

    areas_plot['improvement'] = improvement
    areas_plot['pct_improvement'] = pct_improvement

    # Separate by improvement level
    no_change = areas_plot[(areas_plot['improvement'] == 0) | (N_without == 0)].copy()
    has_improvement = areas_plot[(areas_plot['improvement'] > 0) & (N_without > 0)].copy()

    # Create colormap for improvement (white -> green -> dark green)
    cmap_improve = LinearSegmentedColormap.from_list('improve',
        ['#FFFFFF', '#90EE90', '#32CD32', '#228B22', '#006400'], N=256)

    # Plot no change areas in light gray
    if len(no_change) > 0:
        no_change.plot(ax=ax, color='#E0E0E0', edgecolor='#888888', linewidth=0.3)

    # Plot areas with improvement
    if len(has_improvement) > 0:
        # Use percentage improvement for coloring
        has_improvement['pct_log'] = np.log10(np.maximum(has_improvement['pct_improvement'], 0.01) + 1)
        has_improvement.plot(column='pct_improvement', ax=ax, cmap=cmap_improve,
                            edgecolor='#444444', linewidth=0.3,
                            legend=True,
                            legend_kwds={'label': 'Improvement (%)',
                                        'orientation': 'vertical',
                                        'shrink': 0.6})

    # Plot all bike lanes
    for bl_gdf in bike_lanes_all:
        if bl_gdf is not None and len(bl_gdf) > 0:
            bl_proj = bl_gdf.to_crs(TARGET_CRS)
            bl_proj.plot(ax=ax, color='black', linewidth=1.5, alpha=0.8)

    ax.set_title('Accessibility Improvement from Wishing List Lanes\n(% increase in N per area)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (Israel TM Grid)')
    ax.set_ylabel('Y (Israel TM Grid)')

    # Add stats
    total_improvement = N_with.sum() - N_without.sum()
    pct_total = 100 * total_improvement / N_without.sum() if N_without.sum() > 0 else 0
    n_improved = (improvement > 0).sum()
    max_pct = pct_improvement.max()

    stats_text = f'Total N improvement: {total_improvement:.2e} ({pct_total:.2f}%)\nAreas improved: {n_improved}\nMax improvement: {max_pct:.1f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save PNG only
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def main():
    print("=" * 70)
    print("ACCESSIBILITY HEATMAPS")
    print(f"K = {K_PENALTY}, theta = {THETA}")
    print("=" * 70)

    # Load data
    areas = load_areas()
    areas = areas.to_crs(TARGET_CRS)

    roads = load_roads_kml(ROADS_FILE, areas)
    completed, construction, wishing_list = load_bike_lanes()

    # ===== SCENARIO 1: WITHOUT WISHING LIST =====
    print("\n" + "=" * 70)
    print("SCENARIO 1: WITHOUT WISHING LIST")
    print("=" * 70)

    G1, node_coords1, node_tree1, node_ids1 = build_network(
        roads, [completed, construction], tolerance=NODE_TOLERANCE
    )

    N_without, total_without, connected_without = calculate_accessibility_per_area(
        G1, node_coords1, node_tree1, node_ids1, areas, THETA, K_PENALTY
    )

    print(f"  Total N: {total_without:.2e}")
    print(f"  Connected areas: {connected_without}")

    create_heatmap(areas, N_without, [completed, construction],
                  'Accessibility Heatmap - Without Wishing List',
                  str(script_dir / 'accessibility_heatmap_no_wishing.png'),
                  'Without Wishing List')

    # ===== SCENARIO 2: WITH WISHING LIST =====
    print("\n" + "=" * 70)
    print("SCENARIO 2: WITH WISHING LIST")
    print("=" * 70)

    G2, node_coords2, node_tree2, node_ids2 = build_network(
        roads, [completed, construction, wishing_list], tolerance=NODE_TOLERANCE
    )

    N_with, total_with, connected_with = calculate_accessibility_per_area(
        G2, node_coords2, node_tree2, node_ids2, areas, THETA, K_PENALTY
    )

    print(f"  Total N: {total_with:.2e}")
    print(f"  Connected areas: {connected_with}")

    create_heatmap(areas, N_with, [completed, construction, wishing_list],
                  'Accessibility Heatmap - With Wishing List',
                  str(script_dir / 'accessibility_heatmap_with_wishing.png'),
                  'With Wishing List')

    # ===== SCENARIO 3: GAPS (IMPROVEMENT) =====
    print("\n" + "=" * 70)
    print("SCENARIO 3: GAPS (IMPROVEMENT)")
    print("=" * 70)

    improvement = total_with - total_without
    pct_improvement = 100 * improvement / total_without if total_without > 0 else 0

    print(f"  Total N improvement: {improvement:.2e} ({pct_improvement:.2f}%)")

    create_gaps_heatmap(areas, N_without, N_with,
                       [completed, construction, wishing_list],
                       str(script_dir / 'accessibility_heatmap_gaps.png'))

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Without wishing list: N = {total_without:.2e} ({connected_without} connected)")
    print(f"With wishing list:    N = {total_with:.2e} ({connected_with} connected)")
    print(f"Improvement:          {improvement:.2e} ({pct_improvement:.2f}%)")
    print("\nFiles created:")
    print("  - accessibility_heatmap_no_wishing.png")
    print("  - accessibility_heatmap_with_wishing.png")
    print("  - accessibility_heatmap_gaps.png")


if __name__ == '__main__':
    main()
