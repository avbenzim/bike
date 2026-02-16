"""
Generate all figures and tables for the academic paper.
Outputs:
  - figures/: PDF figures
  - tables/: LaTeX table files

Optimized version: pre-builds network and efficiently computes lane contributions.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import json
import os
from pathlib import Path
from scipy.spatial import cKDTree
from pyproj import Transformer
import fiona
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from shapely.geometry import Point
from shapely.strtree import STRtree
import time
from copy import deepcopy

warnings.filterwarnings('ignore')

# Configure matplotlib for PDF output
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# Color scheme matching the HTML interactive map
# Spectral colormap: blue -> cyan -> yellow -> orange -> red
SPECTRAL_COLORS = [
    (0.0, '#0000CD'),   # Dark blue (low)
    (0.25, '#00CED1'),  # Cyan
    (0.5, '#FFFF00'),   # Yellow
    (0.75, '#FFA500'),  # Orange
    (1.0, '#DC143C'),   # Crimson red (high)
]

# Lane layer colors from HTML
LANE_COLORS = {
    'existing': '#1B5E20',      # Dark green
    'construction': '#81C784',  # Light green
    'planning': '#2196F3',      # Blue
    'checking': '#00BCD4',      # Cyan
    'wishing': '#FF9800',       # Orange
    'selected': '#9b59b6',      # Purple
    'user_drawn': '#E91E63',    # Pink
}

# Impact colors
IMPACT_COLORS = {
    'positive': '#4caf50',      # Green border
    'positive_bg': '#e8f5e9',   # Light green background
    'negative': '#f44336',      # Red border
    'negative_bg': '#ffebee',   # Light red background
    'neutral': '#BEBEBE',       # Gray
}

# UI colors
UI_COLORS = {
    'header': '#2c3e50',
    'accent': '#3498db',
    'success': '#27ae60',
    'origin_marker': '#27ae60',
    'dest_marker': '#e74c3c',
}

def create_spectral_colormap():
    """Create a custom colormap matching the HTML spectral gradient."""
    colors = []
    positions = []
    for pos, hexcolor in SPECTRAL_COLORS:
        positions.append(pos)
        rgb = mcolors.to_rgb(hexcolor)
        colors.append(rgb)

    cmap = LinearSegmentedColormap.from_list('spectral_custom', list(zip(positions, colors)))
    return cmap

# Create the spectral colormap for use in figures
SPECTRAL_CMAP = create_spectral_colormap()


def spectral_color(t):
    """Convert normalized value t (0-1) to RGB color matching HTML spectral gradient.

    This function replicates the exact color interpolation used in the interactive
    HTML map, ensuring visual consistency between the static paper figures and the
    dynamic web-based analysis tool.
    """
    stops = [
        (0.0, (0, 0, 205)),       # #0000CD dark blue
        (0.25, (0, 206, 209)),    # #00CED1 cyan
        (0.5, (255, 255, 0)),     # #FFFF00 yellow
        (0.75, (255, 165, 0)),    # #FFA500 orange
        (1.0, (220, 20, 60))      # #DC143C red
    ]

    t = max(0.0, min(1.0, t))

    i = 0
    while i < len(stops) - 1 and stops[i + 1][0] < t:
        i += 1

    if i >= len(stops) - 1:
        return tuple(c / 255.0 for c in stops[-1][1])

    t0, c0 = stops[i]
    t1, c1 = stops[i + 1]
    f = (t - t0) / (t1 - t0) if t1 != t0 else 0

    r = int(c0[0] + (c1[0] - c0[0]) * f)
    g = int(c0[1] + (c1[1] - c0[1]) * f)
    b = int(c0[2] + (c1[2] - c0[2]) * f)

    return (r / 255.0, g / 255.0, b / 255.0)


def plot_areas_with_values(ax, areas_gdf, values, mode='accessibility', title='', show_colorbar=True):
    """Plot statistical areas with colors matching HTML map exactly.

    This function replicates the exact rendering logic from the interactive HTML map,
    including logarithmic scaling, the spectral color gradient, and consistent styling
    for area boundaries and fill opacity. The resulting visualization is visually
    identical to what users see in the web-based analysis tool.

    The accessibility mode uses natural log scaling to normalize values, matching how
    the HTML interface transforms raw accessibility metrics into color mappings. The
    change mode applies log(v+1) scaling to percentage improvements, ensuring that
    small and large changes are both visible while maintaining proportional relationships.
    """
    areas_plot = areas_gdf.to_crs(WGS84).copy()
    areas_plot['value'] = values

    # Filter to positive values for log scaling (matching HTML behavior)
    pos_values = values[values > 0]

    if len(pos_values) == 0:
        # No positive values - all gray
        areas_plot.plot(ax=ax, color=IMPACT_COLORS['neutral'], edgecolor=UI_COLORS['header'],
                        linewidth=0.5, alpha=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold', color=UI_COLORS['header'])
        ax.set_aspect('equal')
        return

    if mode == 'accessibility':
        # Use natural log scaling (matching HTML: logVals = pos.map(v => Math.log(v)))
        log_values = np.log(pos_values)
        mn_log = log_values.min()
        mx_log = log_values.max()
        fill_opacity = 0.5
    else:  # change mode
        # Use log(v+1) scaling (matching HTML: logV = Math.log(v + 1))
        log_values = np.log(pos_values + 1)
        mx_log = log_values.max()
        mn_log = 0  # For change mode, min is always 0
        fill_opacity = 0.6

    # Plot each area with appropriate color
    for idx, row in areas_plot.iterrows():
        v = row['value']

        if v <= 0:
            # Zero or negative: gray (matching HTML: fillColor: "#BEBEBE")
            color = IMPACT_COLORS['neutral']
            alpha = 0.3 if mode == 'change' else 0.5
        else:
            # Compute normalized value using log scale
            if mode == 'accessibility':
                log_v = np.log(v)
                n = (log_v - mn_log) / (mx_log - mn_log) if mx_log > mn_log else 0
            else:
                log_v = np.log(v + 1)
                n = log_v / mx_log if mx_log > 0 else 0

            color = spectral_color(n)
            alpha = fill_opacity

        # Plot individual polygon
        if row.geometry is not None and not row.geometry.is_empty:
            if row.geometry.geom_type == 'Polygon':
                xs, ys = row.geometry.exterior.xy
                ax.fill(xs, ys, color=color, alpha=alpha, edgecolor=UI_COLORS['header'], linewidth=0.5)
            elif row.geometry.geom_type == 'MultiPolygon':
                for poly in row.geometry.geoms:
                    xs, ys = poly.exterior.xy
                    ax.fill(xs, ys, color=color, alpha=alpha, edgecolor=UI_COLORS['header'], linewidth=0.5)

    ax.set_title(title, fontsize=12, fontweight='bold', color=UI_COLORS['header'])
    ax.set_aspect('equal')

    # Add colorbar
    if show_colorbar:
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        # Create a ScalarMappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=SPECTRAL_CMAP, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)

        if mode == 'accessibility':
            # Show actual value range
            min_val = np.exp(mn_log)
            max_val = np.exp(mx_log)
            cbar.set_label('Accessibility (log scale)', fontsize=9)
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels([f'{min_val:.0f}', f'{np.exp((mn_log+mx_log)/2):.0f}', f'{max_val:.0f}'])
        else:
            # Show percentage range
            max_pct = np.exp(mx_log) - 1 if mx_log > 0 else 0
            cbar.set_label('Improvement % (log scale)', fontsize=9)
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(['0%', f'{max_pct/2:.1f}%', f'{max_pct:.1f}%'])

script_dir = Path(__file__).parent
data_dir = script_dir.parent  # Input files are in parent directory
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326
NODE_TOLERANCE = 15

# Output directories
FIGURES_DIR = script_dir / 'figures'
TABLES_DIR = script_dir / 'tables'

# Default parameters
DEFAULT_K = 5
DEFAULT_THETA = -1.0
DEFAULT_YEAR = 2025

# Parameter ranges for sensitivity analysis
K_VALUES = [2, 5, 10, 50, 100]
THETA_VALUES = [-0.5, -1.0, -1.5, -2.0, -3.0]
DATA_YEARS = [2020, 2025, 2030, 2035, 2040]

# Hebrew to English transliteration for lane names
LANE_TRANSLITERATION = {
    'האר״י-מטודלה': "HaAri-Metudela",
    'אגרון-רמב״ן‎': "Agron-Ramban",
    'קרן היסוד-קינג ג׳ורג׳': "Keren HaYesod-King George",
    'דרך חברון': "Derech Hebron",
    'ז׳בוטינסקי': "Jabotinsky",
    'ג׳ורג׳ אדם סמית׳ - לח״י‎‎': "George Adam Smith-Lehi",
    'הרצל‎': "Herzl",
    'בן זכאי-יהודה הנשיא': "Ben Zakai-Yehuda HaNasi",
    'רחל אמנו-חזקיהו המלך-דוסטאי': "Rachel Imenu-Hizkiyahu",
    'בר לב': "Bar Lev",
    'צבי יהודה': "Tzvi Yehuda",
    'הברון הירש-אליעזר הלוי': "Baron Hirsch-Eliezer HaLevi",
    'אלעזר המודעי-כובשי קטמון': "Elazar HaModa'i-Katamon",
    'הפלמ״ח': "HaPalmach",
    'בזק-בייט': "Bezek-Beit",
    'ירמיהו-בר אילן-לוי אשכול': "Yirmiyahu-Bar Ilan-Eshkol",
    'גולדה-שמואל הנביא': "Golda-Shmuel HaNavi",
    'שטראוס-יחזקאל': "Strauss-Yehezkel",
    'בגין (גבעת רם)': "Begin (Givat Ram)",
    'גולומב': "Golomb",
    'קוליץ': "Kolitz",
    'פייר קניג': "Pierre Koenig",
    'שמגר-אוהל יהושוע-שפע חיים': "Shamgar-Ohel Yehoshua",
    'בצלאל-רבין': "Bezalel-Rabin",
}

def transliterate_name(name):
    """Convert Hebrew lane name to English."""
    if name in LANE_TRANSLITERATION:
        return LANE_TRANSLITERATION[name]
    # If not found, return original
    return name


def load_data():
    """Load all data files."""
    print("Loading data...")
    areas = gpd.read_file(data_dir / "jer_areas.shp")
    areas = areas[areas['in_jeru'] == 1].copy()

    for year in DATA_YEARS:
        areas[f'pop_{year}'] = areas[f'pop_{year}'].fillna(0)
        areas[f'emp_{year}'] = areas[f'emp_{year}'].fillna(0)
    areas['pop'] = areas[f'pop_{DEFAULT_YEAR}'].fillna(0)
    areas['emp'] = areas[f'emp_{DEFAULT_YEAR}'].fillna(0)

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


class NetworkBuilder:
    """Efficient network builder with edge tracking for lane modifications."""

    def __init__(self, roads_proj, areas_proj, tolerance=NODE_TOLERANCE):
        self.tolerance = tolerance
        self.coord_to_node = {}
        self.node_coords = {}
        self.node_counter = 0
        self.G = nx.Graph()
        self.road_geoms = []
        self.road_edges = []
        self.edge_to_bike_lane = {}  # edge_key -> bool

        self._build_base_network(roads_proj, areas_proj)

    def _get_or_create_node(self, x, y):
        key = (round(x / self.tolerance) * self.tolerance, round(y / self.tolerance) * self.tolerance)
        if key in self.coord_to_node:
            return self.coord_to_node[key]
        nid = self.node_counter
        self.node_counter += 1
        self.coord_to_node[key] = nid
        self.node_coords[nid] = (x, y)
        return nid

    def _build_base_network(self, roads_proj, areas_proj):
        """Build the base road network."""
        for _, row in roads_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or geom.geom_type != 'LineString':
                continue
            coords = list(geom.coords)
            if len(coords) >= 2:
                s = self._get_or_create_node(coords[0][0], coords[0][1])
                e = self._get_or_create_node(coords[-1][0], coords[-1][1])
                if s != e:
                    self.G.add_edge(s, e, length=geom.length, has_bike_lane=False)
                    edge_key = (min(s, e), max(s, e))
                    self.road_geoms.append(geom)
                    self.road_edges.append(edge_key)
                    self.edge_to_bike_lane[edge_key] = False

        # Build spatial index
        self.road_tree = STRtree(self.road_geoms) if self.road_geoms else None

        # Connect area centroids
        if areas_proj is not None and len(self.road_geoms) > 0:
            road_tree_for_areas = STRtree(self.road_geoms)
            for _, area_row in areas_proj.iterrows():
                centroid = area_row.geometry.centroid
                centroid_pt = Point(centroid.x, centroid.y)
                nearest_idx = road_tree_for_areas.nearest(centroid_pt)
                nearest_road = self.road_geoms[nearest_idx]
                nearest_point_on_road = nearest_road.interpolate(nearest_road.project(centroid_pt))
                dist_to_road = centroid_pt.distance(nearest_point_on_road)
                road_node = self._get_or_create_node(nearest_point_on_road.x, nearest_point_on_road.y)
                if dist_to_road > self.tolerance:
                    centroid_node = self._get_or_create_node(centroid.x, centroid.y)
                    if centroid_node != road_node:
                        self.G.add_edge(centroid_node, road_node, length=dist_to_road, has_bike_lane=False)

        # Build node tree
        self.node_ids = list(self.node_coords.keys())
        coords_array = np.array([self.node_coords[n] for n in self.node_ids])
        self.node_tree = cKDTree(coords_array) if len(coords_array) > 0 else None

    def mark_bike_lanes(self, bike_lanes_list):
        """Mark edges covered by bike lanes."""
        BUFFER_DIST = 15
        marked_edges = set()

        for bl_gdf in bike_lanes_list:
            if bl_gdf is None or len(bl_gdf) == 0:
                continue
            bl_proj = bl_gdf.to_crs(TARGET_CRS) if bl_gdf.crs != TARGET_CRS else bl_gdf
            for _, row in bl_proj.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms) if geom.geom_type == 'MultiLineString' else []

                for line_geom in lines:
                    if self.road_tree is None:
                        continue
                    buffered = line_geom.buffer(BUFFER_DIST)
                    candidate_indices = self.road_tree.query(buffered)
                    for idx in candidate_indices:
                        road_geom = self.road_geoms[idx]
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
                                edge_key = self.road_edges[idx]
                                marked_edges.add(edge_key)
                        except:
                            pass

        return marked_edges

    def get_graph_with_lanes(self, lane_edges):
        """Get a copy of the graph with specified edges marked as bike lanes."""
        Gw = self.G.copy()
        for edge_key in lane_edges:
            s, e = edge_key
            if Gw.has_edge(s, e):
                Gw[s][e]['has_bike_lane'] = True
        return Gw


def compute_accessibility(G, node_coords, node_tree, node_ids, areas_proj, theta, k, year=DEFAULT_YEAR):
    """Compute accessibility metric."""
    Gw = G.copy()
    for u, v in Gw.edges():
        l = Gw[u][v]['length']
        Gw[u][v]['weight'] = l if Gw[u][v].get('has_bike_lane') else l * k

    centroids = areas_proj.geometry.centroid
    n = len(areas_proj)
    center_nodes = [node_ids[node_tree.query([c.x, c.y])[1]] for c in centroids]

    pop = areas_proj[f'pop_{year}'].values
    emp = areas_proj[f'emp_{year}'].values

    # Get largest connected component
    if nx.is_connected(Gw):
        largest_cc = set(Gw.nodes())
    else:
        largest_cc = max(nx.connected_components(Gw), key=len)

    acc_orig = np.zeros(n)
    acc_dest = np.zeros(n)
    total_N = 0.0

    for i in range(n):
        if center_nodes[i] not in largest_cc:
            continue
        try:
            dists = nx.single_source_dijkstra_path_length(Gw, center_nodes[i], weight='weight')
        except:
            continue
        for j in range(n):
            if i != j and center_nodes[j] in dists:
                tau = max(dists[center_nodes[j]] / 1000, 0.1)
                decay = tau ** theta
                contribution = pop[i] * emp[j] * decay
                total_N += contribution
                acc_orig[i] += emp[j] * decay
                acc_dest[j] += pop[i] * decay

    return acc_orig, acc_dest, total_N


def compute_lane_rankings(network, areas_proj, wishing, baseline_edges, theta, k, year=DEFAULT_YEAR):
    """Efficiently compute lane rankings by reusing network structure."""
    print(f"  Computing rankings for K={k}, theta={theta}, year={year}...")

    wishing_proj = wishing.to_crs(TARGET_CRS)

    # Compute baseline accessibility
    G_base = network.get_graph_with_lanes(baseline_edges)
    _, _, baseline_N = compute_accessibility(
        G_base, network.node_coords, network.node_tree, network.node_ids,
        areas_proj, theta, k, year
    )

    results = []

    for idx in range(len(wishing_proj)):
        row = wishing_proj.iloc[idx]
        lane_name_raw = row.get('Name', f'Lane {idx}')
        lane_name = transliterate_name(lane_name_raw)
        geom = row.geometry
        lane_length = geom.length if geom else 0

        # Get edges covered by this lane
        lane_edges = network.mark_bike_lanes([wishing_proj.iloc[[idx]]])
        all_edges = baseline_edges | lane_edges

        # Compute accessibility with this lane
        G_with = network.get_graph_with_lanes(all_edges)
        _, _, new_N = compute_accessibility(
            G_with, network.node_coords, network.node_tree, network.node_ids,
            areas_proj, theta, k, year
        )

        improvement = new_N - baseline_N
        improvement_pct = (improvement / baseline_N * 100) if baseline_N > 0 else 0

        results.append({
            'name': lane_name,
            'name_raw': lane_name_raw,
            'length': lane_length,
            'improvement_pct': improvement_pct,
        })

    # Sort by improvement
    results.sort(key=lambda x: x['improvement_pct'], reverse=True)

    # Add ranks and cumulative
    cumulative = 0.0
    for i, r in enumerate(results):
        r['rank'] = i + 1
        cumulative += r['improvement_pct']
        r['cumulative_pct'] = cumulative

    return results, baseline_N


# Table generation functions
def generate_table_1_theta():
    """Generate Table 1: Distance Decay Parameter Interpretation."""
    return r"""\begin{table}[H]
\centering
\caption{Distance Decay Parameter Interpretation}
\label{tab:theta_values}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter Value} & \textbf{Interpretation} \\
\midrule
$\theta = -0.5$ & Slow decay; long trips acceptable \\
$\theta = -1.0$ & Moderate decay (default) \\
$\theta = -2.0$ & Fast decay; only nearby destinations matter \\
$\theta = -3.0$ & Very fast decay; highly localized trips \\
\bottomrule
\end{tabular}
\end{table}
"""


def generate_table_2_k():
    """Generate Table 2: K-Penalty Parameter Interpretation."""
    return r"""\begin{table}[H]
\centering
\caption{K-Penalty Parameter Interpretation}
\label{tab:k_values}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{K Value} & \textbf{Interpretation} \\
\midrule
$K = 2$ & Mild penalty; cyclists tolerate mixed traffic \\
$K = 5$ & Moderate preference (default) \\
$K = 10$ & Strong preference for bike lanes \\
$K = 50$ & Very strong preference \\
$K = 100$ & Near-exclusive use of bike lanes \\
\bottomrule
\end{tabular}
\end{table}
"""


def generate_table_3_data_sources():
    """Generate Table 3: Data Sources."""
    return r"""\begin{table}[H]
\centering
\caption{Data Sources}
\label{tab:data_sources}
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Data} & \textbf{Source} & \textbf{Description} \\
\midrule
Statistical Areas & JTMP Team & 98 geographic units \\
Population & JTMP Team & Residents per area, 2020--2040 \\
Employment & JTMP Team & Jobs per area, 2020--2040 \\
Completed Bike Lanes & JTMP Team & Existing infrastructure \\
Under Construction & JTMP Team & Lanes being built \\
Wishing List Lanes & Author & 24 proposed future lanes \\
Road Network & OpenStreetMap & Complete road network \\
\bottomrule
\end{tabular}
\end{table}
"""


def generate_table_4_wishing_list(wishing):
    """Generate Table 4: Wishing List Bike Lanes."""
    wishing_proj = wishing.to_crs(TARGET_CRS)

    rows = []
    for idx in range(len(wishing_proj)):
        row = wishing_proj.iloc[idx]
        name_raw = row.get('Name', f'Lane {idx+1}')
        name = transliterate_name(name_raw)
        length = row.geometry.length if row.geometry else 0
        location = "Jerusalem"
        rows.append((idx + 1, name, location, int(length)))

    table_rows = []
    for r in rows:
        name = str(r[1]).replace('&', r'\&').replace('_', r'\_')
        table_rows.append(f"{r[0]} & {name} & {r[2]} & {r[3]:,}")

    return r"""\begin{table}[H]
\centering
\caption{Wishing List Bike Lanes}
\label{tab:wishing_list}
\small
\begin{tabular}{@{}clll@{}}
\toprule
\textbf{ID} & \textbf{Lane Name} & \textbf{Location} & \textbf{Length (m)} \\
\midrule
""" + " \\\\\n".join(table_rows) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""


def generate_table_5_rankings(rankings):
    """Generate Table 5: Lane Rankings - Additive Mode."""
    table_rows = []
    for r in rankings:
        name = str(r['name']).replace('&', r'\&').replace('_', r'\_')
        table_rows.append(f"{r['rank']} & {name} & {r['improvement_pct']:.2f} & {r['cumulative_pct']:.2f}")

    return r"""\begin{table}[H]
\centering
\caption{Lane Rankings -- Additive Mode ($K=5$, $\theta=-1.0$)}
\label{tab:lane_rankings}
\begin{tabular}{@{}clcc@{}}
\toprule
\textbf{Rank} & \textbf{Lane Name} & \textbf{\% Improvement} & \textbf{Cumulative \%} \\
\midrule
""" + " \\\\\n".join(table_rows) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""


def generate_table_6_sensitivity_k(rankings_by_k):
    """Generate Table 6: Top 5 Lane Rankings by K Value."""
    k_values = [2, 5, 10, 50, 100]

    rows = []
    for rank in range(1, 6):
        row_data = [str(rank)]
        for k in k_values:
            if k in rankings_by_k and len(rankings_by_k[k]) >= rank:
                name = str(rankings_by_k[k][rank-1]['name'])
                if len(name) > 15:
                    name = name[:12] + "..."
                name = name.replace('&', r'\&').replace('_', r'\_')
                row_data.append(name)
            else:
                row_data.append("--")
        rows.append(" & ".join(row_data))

    return r"""\begin{table}[H]
\centering
\caption{Top 5 Lane Rankings by K Value}
\label{tab:sensitivity_k}
\begin{tabular}{@{}clllll@{}}
\toprule
\textbf{Rank} & \textbf{K=2} & \textbf{K=5} & \textbf{K=10} & \textbf{K=50} & \textbf{K=100} \\
\midrule
""" + " \\\\\n".join(rows) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""


def generate_table_7_sensitivity_theta(rankings_by_theta):
    """Generate Table 7: Top 5 Lane Rankings by Theta Value."""
    theta_values = [-0.5, -1.0, -1.5, -2.0, -3.0]

    rows = []
    for rank in range(1, 6):
        row_data = [str(rank)]
        for theta in theta_values:
            if theta in rankings_by_theta and len(rankings_by_theta[theta]) >= rank:
                name = str(rankings_by_theta[theta][rank-1]['name'])
                if len(name) > 12:
                    name = name[:10] + "..."
                name = name.replace('&', r'\&').replace('_', r'\_')
                row_data.append(name)
            else:
                row_data.append("--")
        rows.append(" & ".join(row_data))

    return r"""\begin{table}[H]
\centering
\caption{Top 5 Lane Rankings by $\theta$ Value}
\label{tab:sensitivity_theta}
\begin{tabular}{@{}clllll@{}}
\toprule
\textbf{Rank} & $\theta=-0.5$ & $\theta=-1.0$ & $\theta=-1.5$ & $\theta=-2.0$ & $\theta=-3.0$ \\
\midrule
""" + " \\\\\n".join(rows) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""


def generate_table_8_temporal(rankings_by_year):
    """Generate Table 8: Top 5 Rankings by Year."""
    years = [2020, 2025, 2030, 2035, 2040]

    rows = []
    for rank in range(1, 6):
        row_data = [str(rank)]
        for year in years:
            if year in rankings_by_year and len(rankings_by_year[year]) >= rank:
                name = str(rankings_by_year[year][rank-1]['name'])
                if len(name) > 12:
                    name = name[:10] + "..."
                name = name.replace('&', r'\&').replace('_', r'\_')
                row_data.append(name)
            else:
                row_data.append("--")
        rows.append(" & ".join(row_data))

    return r"""\begin{table}[H]
\centering
\caption{Top 5 Rankings by Year}
\label{tab:temporal}
\begin{tabular}{@{}clllll@{}}
\toprule
\textbf{Rank} & \textbf{2020} & \textbf{2025} & \textbf{2030} & \textbf{2035} & \textbf{2040} \\
\midrule
""" + " \\\\\n".join(rows) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""


def generate_table_9_phases(rankings):
    """Generate Table 9: Cumulative Accessibility Improvement by Phase."""
    def escape(s):
        return str(s).replace('&', r'\&').replace('_', r'\_')

    # Phase 1: Top 3
    phase1_cum = sum(r['improvement_pct'] for r in rankings[:3])
    phase1_lanes = ", ".join(str(r['name']) for r in rankings[:3])

    # Phase 2: +3 (4-6)
    phase2_cum = sum(r['improvement_pct'] for r in rankings[:6])
    phase2_lanes = "+ " + ", ".join(str(r['name']) for r in rankings[3:6])

    # Phase 3: +3 (7-9)
    phase3_cum = sum(r['improvement_pct'] for r in rankings[:9])
    phase3_lanes = "+ " + ", ".join(str(r['name']) for r in rankings[6:9])

    # Phase 4: +3 (10-12)
    phase4_cum = sum(r['improvement_pct'] for r in rankings[:12])
    phase4_lanes = "+ " + ", ".join(str(r['name']) for r in rankings[9:12])

    # Complete
    complete_cum = sum(r['improvement_pct'] for r in rankings)

    return r"""\begin{table}[H]
\centering
\caption{Cumulative Accessibility Improvement by Implementation Phase}
\label{tab:phases}
\begin{tabular}{@{}llc@{}}
\toprule
\textbf{Phase} & \textbf{Lanes Added} & \textbf{Cumulative \%} \\
\midrule
""" + f"""Phase 1 (Top 3) & {escape(phase1_lanes[:50])} & {phase1_cum:.2f} \\\\
Phase 2 (+3) & {escape(phase2_lanes[:50])} & {phase2_cum:.2f} \\\\
Phase 3 (+3) & {escape(phase3_lanes[:50])} & {phase3_cum:.2f} \\\\
Phase 4 (+3) & {escape(phase4_lanes[:50])} & {phase4_cum:.2f} \\\\
Complete (All {len(rankings)}) & All proposed lanes & {complete_cum:.2f} \\\\
""" + r"""\bottomrule
\end{tabular}
\end{table}
"""


def generate_table_10_performance():
    """Generate Table 10: Computational Performance."""
    return r"""\begin{table}[H]
\centering
\caption{Computational Performance}
\label{tab:performance}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Operation} & \textbf{Runtime} \\
\midrule
Network construction & 15--30 seconds \\
Single accessibility calculation & 30--60 seconds \\
Lane ranking (25 lanes) & 10--15 minutes \\
Path finding (single pair) & $<$100 milliseconds \\
\bottomrule
\end{tabular}
\end{table}
"""


def generate_table_11_network_stats(G, n_areas):
    """Generate Table 11: Network Statistics."""
    return r"""\begin{table}[H]
\centering
\caption{Network Statistics}
\label{tab:network_stats}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
""" + f"""Total nodes & {G.number_of_nodes():,} \\\\
Total edges & {G.number_of_edges():,} \\\\
Statistical areas & {n_areas} \\\\
Node merge tolerance & 15 meters \\\\
Gap connection threshold & 50 meters \\\\
""" + r"""\bottomrule
\end{tabular}
\end{table}
"""


def generate_figure_1_baseline_accessibility(areas, acc_orig, acc_dest, output_path):
    """Generate Figure 1: Baseline Accessibility map showing both origin and destination.

    This figure presents the baseline accessibility landscape across Jerusalem's statistical
    areas before any proposed bike lanes from the wishing list are added to the network.
    The visualization uses a spectral color gradient that ranges from dark blue representing
    areas with low accessibility values through cyan, yellow, and orange, culminating in
    crimson red for areas with the highest accessibility. This color scheme maintains
    consistency with the interactive web-based analysis tool, ensuring that researchers
    and planners can seamlessly transition between the static figures and the dynamic
    exploration capabilities of the HTML interface.

    The origin accessibility metric captures how well residents of each statistical area
    can reach employment opportunities throughout the city via the bike network. Areas
    with higher origin accessibility values indicate neighborhoods where cycling provides
    effective access to jobs, commercial centers, and other destinations. Conversely,
    areas with low origin accessibility represent neighborhoods that would benefit most
    from improved cycling infrastructure connecting them to the broader urban fabric.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Origin accessibility (left panel)
    plot_areas_with_values(axes[0], areas, acc_orig, mode='accessibility',
                           title='Origin Accessibility', show_colorbar=True)
    axes[0].set_xlabel('Longitude', fontsize=10)
    axes[0].set_ylabel('Latitude', fontsize=10)

    # Destination accessibility (right panel)
    plot_areas_with_values(axes[1], areas, acc_dest, mode='accessibility',
                           title='Destination Accessibility', show_colorbar=True)
    axes[1].set_xlabel('Longitude', fontsize=10)
    axes[1].set_ylabel('Latitude', fontsize=10)

    fig.suptitle('Baseline Accessibility by Statistical Area', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def generate_figure_2_improvement(areas, improvement_orig, improvement_dest, top_lanes, output_path):
    """Generate Figure 2: Accessibility Improvement with Top 5 Lanes for both origin and destination.

    This figure illustrates the transformative potential of strategic bike lane investments
    by visualizing how the top five ranked lanes from the wishing list would improve
    accessibility across Jerusalem. The analysis considers both origin accessibility,
    which measures how residents benefit from improved connections to destinations, and
    destination accessibility, which captures how employment centers and services become
    more reachable from residential areas throughout the city.

    The improvement percentages shown in this figure represent the relative change in
    accessibility values compared to the baseline network configuration. Areas displayed
    in warmer colors toward the red end of the spectrum experience the most significant
    improvements, indicating that these neighborhoods would see substantial benefits from
    the proposed infrastructure investments. The spatial distribution of improvements
    reveals important patterns about which communities stand to gain the most from the
    prioritized lane construction sequence.

    Understanding both origin and destination perspectives is essential for comprehensive
    transportation planning. A neighborhood might show modest origin accessibility gains
    but substantial destination improvements if it serves as an employment hub that becomes
    better connected to residential areas. Conversely, residential neighborhoods often
    show stronger origin improvements as new lanes connect them to commercial and employment
    centers elsewhere in the city.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Origin improvement (left panel)
    plot_areas_with_values(axes[0], areas, improvement_orig, mode='change',
                           title='Origin Accessibility Improvement', show_colorbar=True)
    axes[0].set_xlabel('Longitude', fontsize=10)
    axes[0].set_ylabel('Latitude', fontsize=10)

    # Destination improvement (right panel)
    plot_areas_with_values(axes[1], areas, improvement_dest, mode='change',
                           title='Destination Accessibility Improvement', show_colorbar=True)
    axes[1].set_xlabel('Longitude', fontsize=10)
    axes[1].set_ylabel('Latitude', fontsize=10)

    # Create title with lane names
    lane_names = ', '.join(top_lanes[:3]) + (' and others' if len(top_lanes) > 3 else '')
    fig.suptitle(f'Accessibility Improvement (%) with Top {len(top_lanes)} Lanes\n({lane_names})',
                 fontsize=13, fontweight='bold', y=1.04)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_lane_geometry(ax, lane_geom, color, linewidth=4):
    """Plot a lane geometry on an axis, handling LineString and MultiLineString types."""
    if lane_geom is None:
        return

    if lane_geom.geom_type == 'LineString':
        coords = list(lane_geom.coords)
        xs, ys = zip(*[(c[0], c[1]) for c in coords])
        ax.plot(xs, ys, color=color, linewidth=linewidth, solid_capstyle='round', zorder=10)
    elif lane_geom.geom_type == 'MultiLineString':
        for line in lane_geom.geoms:
            coords = list(line.coords)
            xs, ys = zip(*[(c[0], c[1]) for c in coords])
            ax.plot(xs, ys, color=color, linewidth=linewidth, solid_capstyle='round', zorder=10)


def generate_figure_3_single_lane_impact(areas, lane_name, imp_orig, imp_dest, lane_geom, output_path):
    """Generate a figure showing the impact of a single lane on both origin and destination accessibility.

    This figure provides a detailed examination of how one specific proposed bike lane
    would affect accessibility patterns throughout Jerusalem. By isolating the contribution
    of a single infrastructure element, planners and decision-makers can better understand
    the spatial reach and magnitude of benefits that each lane provides. The visualization
    includes the lane geometry itself, rendered prominently on the map to show its physical
    location and extent within the urban fabric.

    The origin accessibility panel on the left reveals which residential areas would see
    improved connections to the rest of the city if this lane were constructed. Higher
    improvement values in specific statistical areas indicate neighborhoods whose residents
    would benefit from shorter effective cycling distances to employment, services, and
    amenities. The spatial pattern of origin improvements often radiates outward from
    the lane location, with the strongest effects observed in areas directly served by
    or adjacent to the proposed infrastructure.

    The destination accessibility panel on the right shows how the same lane affects the
    reachability of different areas as destinations. Employment centers, commercial districts,
    and service locations that become more accessible to cyclists throughout the city
    appear with higher improvement values. This perspective is particularly valuable for
    economic development planning, as it identifies which areas would see increased
    potential customer or worker catchments from improved cycling connectivity.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Origin improvement (left panel)
    plot_areas_with_values(axes[0], areas, imp_orig, mode='change',
                           title='Origin Accessibility Impact', show_colorbar=True)
    plot_lane_geometry(axes[0], lane_geom, LANE_COLORS['wishing'], linewidth=4)
    axes[0].set_xlabel('Longitude', fontsize=10)
    axes[0].set_ylabel('Latitude', fontsize=10)

    # Add legend for lane
    lane_line = Line2D([0], [0], color=LANE_COLORS['wishing'], linewidth=4, label='Proposed Lane')
    axes[0].legend(handles=[lane_line], loc='lower left', fontsize=9)

    # Destination improvement (right panel)
    plot_areas_with_values(axes[1], areas, imp_dest, mode='change',
                           title='Destination Accessibility Impact', show_colorbar=True)
    plot_lane_geometry(axes[1], lane_geom, LANE_COLORS['wishing'], linewidth=4)
    axes[1].set_xlabel('Longitude', fontsize=10)
    axes[1].set_ylabel('Latitude', fontsize=10)
    axes[1].legend(handles=[lane_line], loc='lower left', fontsize=9)

    fig.suptitle(f'Accessibility Impact of {lane_name}', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def generate_figure_4_top_lanes_comparison(areas, lane_impacts, output_path):
    """Generate a multi-panel figure comparing the top ranked lanes side by side.

    This comprehensive visualization presents the accessibility impacts of the top
    three ranked lanes in a format that facilitates direct comparison. Each row
    corresponds to one proposed lane, while the columns separate origin and destination
    accessibility perspectives. By arranging the information in this grid format,
    planners can quickly identify similarities and differences in how each lane
    affects the city's cycling accessibility landscape.

    The consistent color scale across all panels ensures that comparisons are meaningful
    and accurate. A lane showing intense red coloration in multiple statistical areas
    delivers more concentrated benefits than one with predominantly blue or cyan tones.
    The spectral color gradient, matching the interactive HTML analysis tool, provides
    intuitive interpretation where warmer colors indicate stronger improvements and
    cooler colors represent more modest gains.

    Examining the spatial patterns across lanes reveals important insights about
    infrastructure complementarity. Lanes that improve accessibility in different
    parts of the city may together provide broader coverage than lanes with overlapping
    impact zones. This visualization supports the development of phased implementation
    strategies that maximize cumulative benefits while ensuring geographic equity in
    infrastructure investments across Jerusalem's diverse neighborhoods.
    """
    n_lanes = min(3, len(lane_impacts))
    fig, axes = plt.subplots(n_lanes, 2, figsize=(14, 5 * n_lanes))

    if n_lanes == 1:
        axes = axes.reshape(1, 2)

    for i, impact in enumerate(lane_impacts[:n_lanes]):
        # Origin panel
        ax1 = axes[i, 0]
        plot_areas_with_values(ax1, areas, impact['imp_orig'], mode='change',
                               title=f'{impact["name"]} - Origin', show_colorbar=(i == 0))
        plot_lane_geometry(ax1, impact.get('geom_wgs84'), LANE_COLORS['wishing'], linewidth=3)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Destination panel
        ax2 = axes[i, 1]
        plot_areas_with_values(ax2, areas, impact['imp_dest'], mode='change',
                               title=f'{impact["name"]} - Destination', show_colorbar=(i == 0))
        plot_lane_geometry(ax2, impact.get('geom_wgs84'), LANE_COLORS['wishing'], linewidth=3)
        ax2.set_xticks([])
        ax2.set_yticks([])

    # Add lane legend
    lane_line = Line2D([0], [0], color=LANE_COLORS['wishing'], linewidth=4, label='Proposed Lane')
    fig.legend(handles=[lane_line], loc='lower center', ncol=1, fontsize=10, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle('Comparison of Top Ranked Lanes: Origin vs Destination Impact',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def generate_figure_5_cumulative_impact(areas, cumulative_impacts, output_path):
    """Generate a figure showing cumulative impact as lanes are added progressively.

    This visualization demonstrates the principle of diminishing marginal returns in
    bike lane infrastructure investment. As successive lanes are added to the network,
    each additional lane typically contributes less improvement than the previous one,
    assuming lanes are constructed in order of their individual rankings. This pattern
    emerges because the highest-impact lanes address the most significant gaps in the
    existing network, while subsequent lanes fill progressively smaller connectivity
    deficiencies.

    The four-panel layout presents snapshots of cumulative accessibility improvement
    after adding one, two, three, and five lanes respectively. This progression allows
    planners to visualize how benefits accumulate spatially and to identify threshold
    points where additional investment yields substantially diminished returns. The
    consistent color scale across panels enables direct comparison of improvement
    magnitudes at each stage of implementation.

    Understanding cumulative impacts is essential for budget allocation and phased
    implementation planning. A municipality with limited resources might choose to
    implement only the first three lanes if subsequent additions provide minimal
    additional benefit. Alternatively, the spatial distribution of cumulative benefits
    might reveal that early-phase investments concentrate improvements in certain
    neighborhoods, suggesting that later phases should prioritize geographic equity
    by targeting underserved areas even if absolute accessibility gains are smaller.
    """
    n_stages = min(4, len(cumulative_impacts))
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    stage_labels = ['After Top 1 Lane', 'After Top 2 Lanes', 'After Top 3 Lanes', 'After Top 5 Lanes']
    stages_to_show = [0, 1, 2, 4] if len(cumulative_impacts) >= 5 else list(range(n_stages))

    for i, stage_idx in enumerate(stages_to_show[:4]):
        if stage_idx >= len(cumulative_impacts):
            continue

        impact = cumulative_impacts[stage_idx]
        ax = axes[i]

        plot_areas_with_values(ax, areas, impact['imp_orig'], mode='change',
                               title=f'{stage_labels[i]}\n({impact["lane_names"]})', show_colorbar=True)

        # Plot all lane geometries up to this stage
        for j in range(stage_idx + 1):
            lane_geom = cumulative_impacts[j].get('geom_wgs84')
            color = LANE_COLORS['wishing'] if j == stage_idx else LANE_COLORS['existing']
            plot_lane_geometry(ax, lane_geom, color, linewidth=2.5)

        ax.set_xticks([])
        ax.set_yticks([])

    # Add legend
    new_lane = Line2D([0], [0], color=LANE_COLORS['wishing'], linewidth=4, label='Newly Added Lane')
    prev_lanes = Line2D([0], [0], color=LANE_COLORS['existing'], linewidth=4, label='Previously Added Lanes')
    fig.legend(handles=[new_lane, prev_lanes], loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle('Cumulative Origin Accessibility Improvement by Implementation Phase',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def main():
    """Main function to generate all figures and tables."""
    start_time = time.time()

    # Create output directories
    FIGURES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)
    print(f"Output directories: {FIGURES_DIR}, {TABLES_DIR}")

    # Load data
    areas, roads, completed, construction, plan, check, wishing = load_data()

    areas_proj = areas.to_crs(TARGET_CRS)
    roads_proj = roads.to_crs(TARGET_CRS)

    print(f"Loaded {len(areas)} areas, {len(wishing)} wishing list lanes")

    # Build network
    print("Building network...")
    network = NetworkBuilder(roads_proj, areas_proj)
    print(f"  Network: {network.G.number_of_nodes()} nodes, {network.G.number_of_edges()} edges")

    # Mark baseline bike lanes (completed + construction)
    print("Marking baseline bike lanes...")
    baseline_edges = network.mark_bike_lanes([completed, construction])
    print(f"  Marked {len(baseline_edges)} baseline bike lane edges")

    # Compute baseline accessibility
    print("Computing baseline accessibility...")
    G_base = network.get_graph_with_lanes(baseline_edges)
    acc_orig_base, acc_dest_base, baseline_N = compute_accessibility(
        G_base, network.node_coords, network.node_tree, network.node_ids,
        areas_proj, DEFAULT_THETA, DEFAULT_K, DEFAULT_YEAR
    )
    print(f"  Baseline N = {baseline_N:.2e}")

    # Generate Figure 1: Baseline Accessibility (now includes both origin and destination)
    print("Generating Figure 1: Baseline Accessibility...")
    generate_figure_1_baseline_accessibility(areas, acc_orig_base, acc_dest_base, FIGURES_DIR / 'figure1_baseline_accessibility.pdf')

    # Compute lane rankings for default parameters
    print("Computing lane rankings...")
    rankings_default, _ = compute_lane_rankings(
        network, areas_proj, wishing, baseline_edges,
        DEFAULT_THETA, DEFAULT_K, DEFAULT_YEAR
    )

    # Compute rankings for sensitivity analysis (K values)
    print("Computing K sensitivity analysis...")
    rankings_by_k = {}
    for k in K_VALUES:
        rankings, _ = compute_lane_rankings(
            network, areas_proj, wishing, baseline_edges,
            DEFAULT_THETA, k, DEFAULT_YEAR
        )
        rankings_by_k[k] = rankings

    # Compute rankings for sensitivity analysis (theta values)
    print("Computing theta sensitivity analysis...")
    rankings_by_theta = {}
    for theta in THETA_VALUES:
        rankings, _ = compute_lane_rankings(
            network, areas_proj, wishing, baseline_edges,
            theta, DEFAULT_K, DEFAULT_YEAR
        )
        rankings_by_theta[theta] = rankings

    # Compute rankings for temporal analysis
    print("Computing temporal analysis...")
    rankings_by_year = {}
    for year in DATA_YEARS:
        rankings, _ = compute_lane_rankings(
            network, areas_proj, wishing, baseline_edges,
            DEFAULT_THETA, DEFAULT_K, year
        )
        rankings_by_year[year] = rankings

    # Compute individual lane impacts for all top lanes
    print("Computing individual lane impacts for figures...")
    wishing_proj = wishing.to_crs(TARGET_CRS)
    wishing_wgs84 = wishing.to_crs(WGS84)

    lane_impacts = []
    cumulative_edges = set(baseline_edges)
    cumulative_impacts = []

    for rank_idx, r in enumerate(rankings_default[:5]):
        # Find the lane index by raw name
        for idx in range(len(wishing_proj)):
            raw_name = wishing_proj.iloc[idx].get('Name')
            if raw_name == r.get('name_raw') or transliterate_name(raw_name) == r['name']:
                # Compute impact of this single lane
                single_lane_edges = network.mark_bike_lanes([wishing_proj.iloc[[idx]]])
                edges_with_single = baseline_edges | single_lane_edges

                G_single = network.get_graph_with_lanes(edges_with_single)
                acc_orig_single, acc_dest_single, _ = compute_accessibility(
                    G_single, network.node_coords, network.node_tree, network.node_ids,
                    areas_proj, DEFAULT_THETA, DEFAULT_K, DEFAULT_YEAR
                )

                # Compute improvements
                imp_orig = np.zeros(len(areas_proj))
                imp_dest = np.zeros(len(areas_proj))
                for i in range(len(areas_proj)):
                    if acc_orig_base[i] > 0:
                        imp_orig[i] = ((acc_orig_single[i] - acc_orig_base[i]) / acc_orig_base[i]) * 100
                    if acc_dest_base[i] > 0:
                        imp_dest[i] = ((acc_dest_single[i] - acc_dest_base[i]) / acc_dest_base[i]) * 100

                lane_impacts.append({
                    'name': r['name'],
                    'name_raw': r.get('name_raw'),
                    'rank': rank_idx + 1,
                    'imp_orig': imp_orig.copy(),
                    'imp_dest': imp_dest.copy(),
                    'geom_wgs84': wishing_wgs84.iloc[idx].geometry,
                    'geom_proj': wishing_proj.iloc[idx].geometry,
                })

                # Compute cumulative impact
                cumulative_edges |= single_lane_edges
                G_cumul = network.get_graph_with_lanes(cumulative_edges)
                acc_orig_cumul, acc_dest_cumul, _ = compute_accessibility(
                    G_cumul, network.node_coords, network.node_tree, network.node_ids,
                    areas_proj, DEFAULT_THETA, DEFAULT_K, DEFAULT_YEAR
                )

                cumul_imp_orig = np.zeros(len(areas_proj))
                cumul_imp_dest = np.zeros(len(areas_proj))
                for i in range(len(areas_proj)):
                    if acc_orig_base[i] > 0:
                        cumul_imp_orig[i] = ((acc_orig_cumul[i] - acc_orig_base[i]) / acc_orig_base[i]) * 100
                    if acc_dest_base[i] > 0:
                        cumul_imp_dest[i] = ((acc_dest_cumul[i] - acc_dest_base[i]) / acc_dest_base[i]) * 100

                lane_names = ', '.join([li['name'] for li in lane_impacts])
                cumulative_impacts.append({
                    'imp_orig': cumul_imp_orig.copy(),
                    'imp_dest': cumul_imp_dest.copy(),
                    'lane_names': lane_names,
                    'geom_wgs84': wishing_wgs84.iloc[idx].geometry,
                })
                break

    # Generate Figure 2: Improvement with Top 5 Lanes (both origin and destination)
    print("Generating Figure 2: Top 5 Lanes Improvement...")
    top_lane_names = [li['name'] for li in lane_impacts[:5]]
    generate_figure_2_improvement(
        areas,
        cumulative_impacts[-1]['imp_orig'] if cumulative_impacts else np.zeros(len(areas_proj)),
        cumulative_impacts[-1]['imp_dest'] if cumulative_impacts else np.zeros(len(areas_proj)),
        top_lane_names,
        FIGURES_DIR / 'figure2_improvement_top5.pdf'
    )

    # Generate Figure 3: Individual lane impact for top 3 lanes
    print("Generating Figure 3: Individual Lane Impacts...")
    for i, impact in enumerate(lane_impacts[:3]):
        output_path = FIGURES_DIR / f'figure3_{i+1}_lane_impact_{impact["name"].replace(" ", "_").replace("/", "-")[:20]}.pdf'
        generate_figure_3_single_lane_impact(
            areas,
            impact['name'],
            impact['imp_orig'],
            impact['imp_dest'],
            impact['geom_wgs84'],
            output_path
        )

    # Generate Figure 4: Top 3 lanes comparison
    print("Generating Figure 4: Top Lanes Comparison...")
    generate_figure_4_top_lanes_comparison(areas, lane_impacts[:3], FIGURES_DIR / 'figure4_top_lanes_comparison.pdf')

    # Generate Figure 5: Cumulative impact progression
    print("Generating Figure 5: Cumulative Impact...")
    generate_figure_5_cumulative_impact(areas, cumulative_impacts, FIGURES_DIR / 'figure5_cumulative_impact.pdf')

    # Generate all tables
    print("Generating tables...")

    tables = [
        ('table1_theta.tex', generate_table_1_theta()),
        ('table2_k.tex', generate_table_2_k()),
        ('table3_data_sources.tex', generate_table_3_data_sources()),
        ('table4_wishing_list.tex', generate_table_4_wishing_list(wishing)),
        ('table5_rankings.tex', generate_table_5_rankings(rankings_default)),
        ('table6_sensitivity_k.tex', generate_table_6_sensitivity_k(rankings_by_k)),
        ('table7_sensitivity_theta.tex', generate_table_7_sensitivity_theta(rankings_by_theta)),
        ('table8_temporal.tex', generate_table_8_temporal(rankings_by_year)),
        ('table9_phases.tex', generate_table_9_phases(rankings_default)),
        ('table10_performance.tex', generate_table_10_performance()),
        ('table11_network_stats.tex', generate_table_11_network_stats(network.G, len(areas))),
    ]

    for filename, content in tables:
        with open(TABLES_DIR / filename, 'w') as f:
            f.write(content)
        print(f"  Saved {filename}")

    elapsed = time.time() - start_time
    print(f"\nDone! Total time: {elapsed:.1f} seconds")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == '__main__':
    main()
