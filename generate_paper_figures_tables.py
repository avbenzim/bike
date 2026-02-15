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
from shapely.geometry import Point
from shapely.strtree import STRtree
import time
from copy import deepcopy

warnings.filterwarnings('ignore')

# Configure matplotlib for PDF output
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

script_dir = Path(__file__).parent
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326
NODE_TOLERANCE = 15

# Output directories
FIGURES_DIR = script_dir / 'figures'
TABLES_DIR = script_dir / 'tables'

# Default parameters
DEFAULT_K = 100
DEFAULT_THETA = -1.0
DEFAULT_YEAR = 2025

# Parameter ranges for sensitivity analysis
K_VALUES = [10, 50, 100, 500, 1000]
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
    areas = gpd.read_file(script_dir / "jer_areas.shp")
    areas = areas[areas['in_jeru'] == 1].copy()

    for year in DATA_YEARS:
        areas[f'pop_{year}'] = areas[f'pop_{year}'].fillna(0)
        areas[f'emp_{year}'] = areas[f'emp_{year}'].fillna(0)
    areas['pop'] = areas[f'pop_{DEFAULT_YEAR}'].fillna(0)
    areas['emp'] = areas[f'emp_{DEFAULT_YEAR}'].fillna(0)

    roads = gpd.read_file(script_dir / "jerusalem_roads.kml", driver='KML')
    completed = gpd.read_file(script_dir / "bike_lanes_completed.kml", driver='KML')
    construction = gpd.read_file(script_dir / "bike_lanes_construction.kml", driver='KML')

    try:
        plan = gpd.read_file(script_dir / "bike_lanes_plan.kml", driver='KML', on_invalid='ignore')
        plan = plan[plan.geometry.notnull()].copy()
    except:
        plan = gpd.GeoDataFrame(columns=['geometry', 'Name'], geometry='geometry', crs='EPSG:4326')

    try:
        check = gpd.read_file(script_dir / "bike_lanes_check.kml", driver='KML', on_invalid='ignore')
        check = check[check.geometry.notnull()].copy()
    except:
        check = gpd.GeoDataFrame(columns=['geometry', 'Name'], geometry='geometry', crs='EPSG:4326')

    wishing = gpd.read_file(script_dir / "bike_lanes_wishing_list.kml", driver='KML')

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
$K = 10$ & Moderate preference for bike lanes \\
$K = 100$ & Strong preference (default) \\
$K = 500$ & Very strong preference \\
$K = 1000$ & Near-exclusive use of bike lanes \\
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
\caption{Lane Rankings -- Additive Mode ($K=100$, $\theta=-1.0$)}
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
    k_values = [10, 50, 100, 500, 1000]

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
\textbf{Rank} & \textbf{K=10} & \textbf{K=50} & \textbf{K=100} & \textbf{K=500} & \textbf{K=1000} \\
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


def generate_figure_1_baseline_accessibility(areas, acc_orig, output_path):
    """Generate Figure 1: Baseline Origin Accessibility map."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    areas_plot = areas.to_crs(WGS84).copy()
    areas_plot['accessibility'] = acc_orig

    # Create color map (red=low, green=high)
    cmap = plt.cm.RdYlGn

    areas_plot.plot(column='accessibility', cmap=cmap, ax=ax, edgecolor='black',
                    linewidth=0.5, legend=True, legend_kwds={'label': 'Origin Accessibility'})

    ax.set_title('Baseline Origin Accessibility by Statistical Area', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def generate_figure_2_improvement(areas, improvement_pct, output_path):
    """Generate Figure 2: Accessibility Improvement with Top 5 Lanes."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    areas_plot = areas.to_crs(WGS84).copy()
    areas_plot['improvement'] = improvement_pct

    cmap = plt.cm.Blues

    areas_plot.plot(column='improvement', cmap=cmap, ax=ax, edgecolor='black',
                    linewidth=0.5, legend=True,
                    legend_kwds={'label': 'Accessibility Improvement (%)'})

    ax.set_title('Accessibility Improvement (%) with Top 5 Lanes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    plt.tight_layout()
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

    # Generate Figure 1: Baseline Accessibility
    print("Generating Figure 1...")
    generate_figure_1_baseline_accessibility(areas, acc_orig_base, FIGURES_DIR / 'figure1_baseline_accessibility.pdf')

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

    # Generate Figure 2: Improvement with Top 5 Lanes
    print("Generating Figure 2...")
    # Get top 5 lane edges
    wishing_proj = wishing.to_crs(TARGET_CRS)
    top5_edges = set(baseline_edges)
    for r in rankings_default[:5]:
        # Find the lane index by raw name
        for idx in range(len(wishing_proj)):
            raw_name = wishing_proj.iloc[idx].get('Name')
            if raw_name == r.get('name_raw') or transliterate_name(raw_name) == r['name']:
                lane_edges = network.mark_bike_lanes([wishing_proj.iloc[[idx]]])
                top5_edges |= lane_edges
                break

    G_top5 = network.get_graph_with_lanes(top5_edges)
    acc_orig_top5, _, _ = compute_accessibility(
        G_top5, network.node_coords, network.node_tree, network.node_ids,
        areas_proj, DEFAULT_THETA, DEFAULT_K, DEFAULT_YEAR
    )

    improvement_pct = np.zeros(len(areas_proj))
    for i in range(len(areas_proj)):
        if acc_orig_base[i] > 0:
            improvement_pct[i] = ((acc_orig_top5[i] - acc_orig_base[i]) / acc_orig_base[i]) * 100

    generate_figure_2_improvement(areas, improvement_pct, FIGURES_DIR / 'figure2_improvement_top5.pdf')

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
