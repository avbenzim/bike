"""
Generate high-quality PDF figures for the academic paper using matplotlib with contextily basemaps.
This version produces figures that closely match the HTML map style.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import json
import os
from pathlib import Path
from scipy.spatial import cKDTree
import fiona
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from shapely.geometry import Point
from shapely.strtree import STRtree
import time
import contextily as ctx

warnings.filterwarnings('ignore')

# Configure matplotlib for high-quality output
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

script_dir = Path(__file__).parent
data_dir = script_dir.parent  # Input files are in parent directory
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039  # Israel TM
WGS84 = 4326
WEB_MERCATOR = 3857
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

# Spectral colormap matching HTML exactly
SPECTRAL_COLORS = [
    (0.0, '#0000CD'),   # Dark blue
    (0.25, '#00CED1'),  # Cyan
    (0.5, '#FFFF00'),   # Yellow
    (0.75, '#FFA500'),  # Orange
    (1.0, '#DC143C'),   # Crimson red
]

LANE_COLORS = {
    'existing': '#1B5E20',
    'construction': '#81C784',
    'wishing': '#FF9800',
}

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
    return LANE_TRANSLITERATION.get(name, name)


def create_spectral_colormap():
    """Create a custom colormap matching the HTML spectral gradient."""
    colors = []
    positions = []
    for pos, hexcolor in SPECTRAL_COLORS:
        positions.append(pos)
        rgb = mcolors.to_rgb(hexcolor)
        colors.append(rgb)
    return LinearSegmentedColormap.from_list('spectral_custom', list(zip(positions, colors)))


SPECTRAL_CMAP = create_spectral_colormap()


def spectral_color(t):
    """Convert normalized value t (0-1) to RGB color."""
    stops = [
        (0.0, (0, 0, 205)),
        (0.25, (0, 206, 209)),
        (0.5, (255, 255, 0)),
        (0.75, (255, 165, 0)),
        (1.0, (220, 20, 60))
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
    wishing = gpd.read_file(data_dir / "bike_lanes_wishing_list.kml", driver='KML')

    return areas, roads, completed, construction, wishing


class NetworkBuilder:
    def __init__(self, roads_proj, areas_proj, tolerance=NODE_TOLERANCE):
        self.tolerance = tolerance
        self.coord_to_node = {}
        self.node_coords = {}
        self.node_counter = 0
        self.G = nx.Graph()
        self.road_geoms = []
        self.road_edges = []
        self.edge_to_bike_lane = {}
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

        self.road_tree = STRtree(self.road_geoms) if self.road_geoms else None

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

        self.node_ids = list(self.node_coords.keys())
        coords_array = np.array([self.node_coords[n] for n in self.node_ids])
        self.node_tree = cKDTree(coords_array) if len(coords_array) > 0 else None

    def mark_bike_lanes(self, bike_lanes_list):
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
                            should_mark = (overlap_ratio > 0.5 or (road_geom.length < 50 and overlap_ratio > 0.3) or intersection.length > 20)
                            if should_mark:
                                edge_key = self.road_edges[idx]
                                marked_edges.add(edge_key)
                        except:
                            pass
        return marked_edges

    def get_graph_with_lanes(self, lane_edges):
        Gw = self.G.copy()
        for edge_key in lane_edges:
            s, e = edge_key
            if Gw.has_edge(s, e):
                Gw[s][e]['has_bike_lane'] = True
        return Gw


def compute_accessibility(G, node_coords, node_tree, node_ids, areas_proj, theta, k, year=DEFAULT_YEAR):
    Gw = G.copy()
    for u, v in Gw.edges():
        l = Gw[u][v]['length']
        Gw[u][v]['weight'] = l if Gw[u][v].get('has_bike_lane') else l * k
    centroids = areas_proj.geometry.centroid
    n = len(areas_proj)
    center_nodes = [node_ids[node_tree.query([c.x, c.y])[1]] for c in centroids]
    pop = areas_proj[f'pop_{year}'].values
    emp = areas_proj[f'emp_{year}'].values
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
    print(f"  Computing rankings for K={k}, theta={theta}, year={year}...")
    wishing_proj = wishing.to_crs(TARGET_CRS)
    G_base = network.get_graph_with_lanes(baseline_edges)
    _, _, baseline_N = compute_accessibility(G_base, network.node_coords, network.node_tree, network.node_ids, areas_proj, theta, k, year)
    results = []
    for idx in range(len(wishing_proj)):
        row = wishing_proj.iloc[idx]
        lane_name_raw = row.get('Name', f'Lane {idx}')
        lane_name = transliterate_name(lane_name_raw)
        geom = row.geometry
        lane_length = geom.length if geom else 0
        lane_edges = network.mark_bike_lanes([wishing_proj.iloc[[idx]]])
        all_edges = baseline_edges | lane_edges
        G_with = network.get_graph_with_lanes(all_edges)
        _, _, new_N = compute_accessibility(G_with, network.node_coords, network.node_tree, network.node_ids, areas_proj, theta, k, year)
        improvement = new_N - baseline_N
        improvement_pct = (improvement / baseline_N * 100) if baseline_N > 0 else 0
        results.append({'name': lane_name, 'name_raw': lane_name_raw, 'length': lane_length, 'improvement_pct': improvement_pct})
    results.sort(key=lambda x: x['improvement_pct'], reverse=True)
    cumulative = 0.0
    for i, r in enumerate(results):
        r['rank'] = i + 1
        cumulative += r['improvement_pct']
        r['cumulative_pct'] = cumulative
    return results, baseline_N


def plot_areas_with_basemap(ax, areas_gdf, values, mode='accessibility', title='',
                             show_colorbar=True, bike_lanes=None, highlight_lanes=None):
    """Plot areas with contextily basemap for a more professional look."""
    # Convert to Web Mercator for basemap
    areas_wm = areas_gdf.to_crs(WEB_MERCATOR).copy()
    areas_wm['value'] = values

    pos_values = values[values > 0]

    if len(pos_values) == 0:
        areas_wm.plot(ax=ax, color='#BEBEBE', edgecolor='#2c3e50', linewidth=0.5, alpha=0.5)
    else:
        if mode == 'accessibility':
            log_values = np.log(pos_values)
            mn_log = log_values.min()
            mx_log = log_values.max()
            fill_opacity = 0.65
        else:
            log_values = np.log(pos_values + 1)
            mx_log = log_values.max()
            mn_log = 0
            fill_opacity = 0.75

        colors = []
        for v in values:
            if v <= 0:
                colors.append('#BEBEBE')
            else:
                if mode == 'accessibility':
                    log_v = np.log(v)
                    n = (log_v - mn_log) / (mx_log - mn_log) if mx_log > mn_log else 0
                else:
                    log_v = np.log(v + 1)
                    n = log_v / mx_log if mx_log > 0 else 0
                colors.append(spectral_color(n))

        areas_wm.plot(ax=ax, color=colors, edgecolor='#34495e', linewidth=0.4, alpha=fill_opacity)

    # Add basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=13, alpha=0.4)
    except Exception as e:
        print(f"    Warning: Could not add basemap: {e}")

    # Plot bike lanes
    if bike_lanes:
        for layer_name, layer_gdf in bike_lanes:
            if layer_gdf is None or len(layer_gdf) == 0:
                continue
            layer_wm = layer_gdf.to_crs(WEB_MERCATOR)
            color = LANE_COLORS.get(layer_name, '#888888')
            layer_wm.plot(ax=ax, color=color, linewidth=2.5, alpha=0.9, zorder=5)

    # Plot highlighted lanes
    if highlight_lanes:
        for lane_gdf, color in highlight_lanes:
            if lane_gdf is None or len(lane_gdf) == 0:
                continue
            lane_wm = lane_gdf.to_crs(WEB_MERCATOR)
            lane_wm.plot(ax=ax, color=color, linewidth=4, alpha=1.0, zorder=10)

    ax.set_title(title, fontsize=13, fontweight='bold', color='#2c3e50', pad=10)
    ax.set_axis_off()

    # Add colorbar
    if show_colorbar and len(pos_values) > 0:
        sm = ScalarMappable(cmap=SPECTRAL_CMAP, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=20)
        if mode == 'accessibility':
            cbar.set_label('Accessibility (log scale)', fontsize=10)
        else:
            cbar.set_label('Improvement % (log scale)', fontsize=10)
        cbar.ax.tick_params(labelsize=9)


def generate_figure_1(areas, acc_orig, acc_dest, completed, construction, output_path):
    """Generate Figure 1: Baseline Accessibility."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    bike_lanes = [('existing', completed), ('construction', construction)]

    plot_areas_with_basemap(axes[0], areas, acc_orig, mode='accessibility',
                             title='Origin Accessibility', show_colorbar=True,
                             bike_lanes=bike_lanes)

    plot_areas_with_basemap(axes[1], areas, acc_dest, mode='accessibility',
                             title='Destination Accessibility', show_colorbar=True,
                             bike_lanes=bike_lanes)

    fig.suptitle(f'Baseline Accessibility (K={DEFAULT_K}, θ={DEFAULT_THETA})',
                 fontsize=16, fontweight='bold', color='#2c3e50', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def generate_figure_2(areas, imp_orig, imp_dest, top_lane_names, completed, construction, top_lanes_gdf, output_path):
    """Generate Figure 2: Top 5 Lanes Improvement."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    bike_lanes = [('existing', completed), ('construction', construction)]
    highlight = [(gdf, LANE_COLORS['wishing']) for gdf in top_lanes_gdf]

    plot_areas_with_basemap(axes[0], areas, imp_orig, mode='change',
                             title='Origin Accessibility Improvement (%)', show_colorbar=True,
                             bike_lanes=bike_lanes, highlight_lanes=highlight)

    plot_areas_with_basemap(axes[1], areas, imp_dest, mode='change',
                             title='Destination Accessibility Improvement (%)', show_colorbar=True,
                             bike_lanes=bike_lanes, highlight_lanes=highlight)

    names_str = ', '.join(top_lane_names[:3]) + ('...' if len(top_lane_names) > 3 else '')
    fig.suptitle(f'Accessibility Improvement with Top {len(top_lane_names)} Lanes\n({names_str})',
                 fontsize=15, fontweight='bold', color='#2c3e50', y=0.99)

    # Add legend for lanes
    existing_line = Line2D([0], [0], color=LANE_COLORS['existing'], linewidth=3, label='Existing Lanes')
    construction_line = Line2D([0], [0], color=LANE_COLORS['construction'], linewidth=3, label='Under Construction')
    wishing_line = Line2D([0], [0], color=LANE_COLORS['wishing'], linewidth=4, label='Top Ranked Lanes')
    fig.legend(handles=[existing_line, construction_line, wishing_line],
               loc='lower center', ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def generate_figure_3(areas, lane_name, imp_orig, imp_dest, lane_gdf, completed, construction, output_path):
    """Generate Figure 3: Individual Lane Impact."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    bike_lanes = [('existing', completed), ('construction', construction)]
    highlight = [(lane_gdf, LANE_COLORS['wishing'])]

    plot_areas_with_basemap(axes[0], areas, imp_orig, mode='change',
                             title='Origin Accessibility Impact (%)', show_colorbar=True,
                             bike_lanes=bike_lanes, highlight_lanes=highlight)

    plot_areas_with_basemap(axes[1], areas, imp_dest, mode='change',
                             title='Destination Accessibility Impact (%)', show_colorbar=True,
                             bike_lanes=bike_lanes, highlight_lanes=highlight)

    fig.suptitle(f'Accessibility Impact of {lane_name}',
                 fontsize=15, fontweight='bold', color='#2c3e50', y=0.98)

    wishing_line = Line2D([0], [0], color=LANE_COLORS['wishing'], linewidth=4, label='Proposed Lane')
    fig.legend(handles=[wishing_line], loc='lower center', fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def generate_figure_4(areas, lane_impacts, completed, construction, output_path):
    """Generate Figure 4: Top 3 Lanes Comparison (3x2 grid)."""
    n_lanes = min(3, len(lane_impacts))
    fig, axes = plt.subplots(n_lanes, 2, figsize=(16, 6 * n_lanes))

    if n_lanes == 1:
        axes = axes.reshape(1, 2)

    bike_lanes = [('existing', completed), ('construction', construction)]

    for i, impact in enumerate(lane_impacts[:n_lanes]):
        highlight = [(impact['lane_gdf'], LANE_COLORS['wishing'])]

        plot_areas_with_basemap(axes[i, 0], areas, impact['imp_orig'], mode='change',
                                 title=f"#{impact['rank']}. {impact['name']} - Origin",
                                 show_colorbar=(i == 0),
                                 bike_lanes=bike_lanes, highlight_lanes=highlight)

        plot_areas_with_basemap(axes[i, 1], areas, impact['imp_dest'], mode='change',
                                 title=f"#{impact['rank']}. {impact['name']} - Destination",
                                 show_colorbar=False,
                                 bike_lanes=bike_lanes, highlight_lanes=highlight)

    fig.suptitle('Comparison of Top Ranked Lanes: Origin vs Destination Impact',
                 fontsize=16, fontweight='bold', color='#2c3e50', y=0.99)

    wishing_line = Line2D([0], [0], color=LANE_COLORS['wishing'], linewidth=4, label='Proposed Lane')
    fig.legend(handles=[wishing_line], loc='lower center', fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def generate_figure_5(areas, cumulative_stages, completed, construction, output_path):
    """Generate Figure 5: Cumulative Impact Progression (2x2 grid)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    bike_lanes = [('existing', completed), ('construction', construction)]
    stage_labels = ['After Top 1 Lane', 'After Top 2 Lanes', 'After Top 3 Lanes', 'After Top 5 Lanes']

    for i, stage in enumerate(cumulative_stages[:4]):
        highlight = [(gdf, LANE_COLORS['wishing']) for gdf in stage['lane_gdfs']]

        plot_areas_with_basemap(axes[i], areas, stage['imp_orig'], mode='change',
                                 title=stage_labels[i], show_colorbar=(i == 0),
                                 bike_lanes=bike_lanes, highlight_lanes=highlight)

    fig.suptitle('Cumulative Origin Accessibility Improvement by Implementation Phase',
                 fontsize=16, fontweight='bold', color='#2c3e50', y=0.99)

    wishing_line = Line2D([0], [0], color=LANE_COLORS['wishing'], linewidth=4, label='Added Lanes')
    existing_line = Line2D([0], [0], color=LANE_COLORS['existing'], linewidth=3, label='Existing Lanes')
    fig.legend(handles=[existing_line, wishing_line], loc='lower center', ncol=2,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


# Table generation functions
def generate_table_5_rankings(rankings):
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


def main():
    start_time = time.time()

    FIGURES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)
    print(f"Output directories: {FIGURES_DIR}, {TABLES_DIR}")

    # Load data
    areas, roads, completed, construction, wishing = load_data()
    areas_proj = areas.to_crs(TARGET_CRS)
    roads_proj = roads.to_crs(TARGET_CRS)
    print(f"Loaded {len(areas)} areas, {len(wishing)} wishing list lanes")

    # Build network
    print("Building network...")
    network = NetworkBuilder(roads_proj, areas_proj)
    print(f"  Network: {network.G.number_of_nodes()} nodes, {network.G.number_of_edges()} edges")

    # Mark baseline bike lanes
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

    # Generate Figure 1
    print("\nGenerating Figure 1: Baseline Accessibility...")
    generate_figure_1(areas, acc_orig_base, acc_dest_base, completed, construction,
                       FIGURES_DIR / 'figure1_baseline_accessibility.pdf')

    # Compute lane rankings
    print("\nComputing lane rankings...")
    rankings_default, _ = compute_lane_rankings(network, areas_proj, wishing, baseline_edges,
                                                 DEFAULT_THETA, DEFAULT_K, DEFAULT_YEAR)

    print("\n  Top 5 lanes:")
    for r in rankings_default[:5]:
        print(f"    {r['rank']}. {r['name']}: +{r['improvement_pct']:.4f}%")

    # Compute sensitivity analysis
    print("\nComputing K sensitivity analysis...")
    rankings_by_k = {}
    for k in K_VALUES:
        rankings, _ = compute_lane_rankings(network, areas_proj, wishing, baseline_edges, DEFAULT_THETA, k, DEFAULT_YEAR)
        rankings_by_k[k] = rankings

    # Compute individual lane impacts
    wishing_proj = wishing.to_crs(TARGET_CRS)
    wishing_wgs84 = wishing.to_crs(WGS84)

    lane_impacts = []
    cumulative_edges = set(baseline_edges)
    cumulative_stages = []

    for rank_idx, r in enumerate(rankings_default[:5]):
        print(f"\nComputing impact for lane {rank_idx+1}: {r['name']}...")

        for idx in range(len(wishing_proj)):
            raw_name = wishing_proj.iloc[idx].get('Name')
            if raw_name == r.get('name_raw') or transliterate_name(raw_name) == r['name']:
                single_lane_edges = network.mark_bike_lanes([wishing_proj.iloc[[idx]]])
                edges_with_single = baseline_edges | single_lane_edges

                G_single = network.get_graph_with_lanes(edges_with_single)
                acc_orig_single, acc_dest_single, _ = compute_accessibility(
                    G_single, network.node_coords, network.node_tree, network.node_ids,
                    areas_proj, DEFAULT_THETA, DEFAULT_K, DEFAULT_YEAR
                )

                imp_orig = np.zeros(len(areas_proj))
                imp_dest = np.zeros(len(areas_proj))
                for i in range(len(areas_proj)):
                    if acc_orig_base[i] > 0:
                        imp_orig[i] = ((acc_orig_single[i] - acc_orig_base[i]) / acc_orig_base[i]) * 100
                    if acc_dest_base[i] > 0:
                        imp_dest[i] = ((acc_dest_single[i] - acc_dest_base[i]) / acc_dest_base[i]) * 100

                lane_impacts.append({
                    'name': r['name'],
                    'rank': rank_idx + 1,
                    'imp_orig': imp_orig,
                    'imp_dest': imp_dest,
                    'lane_gdf': wishing_wgs84.iloc[[idx]],
                })

                cumulative_edges |= single_lane_edges

                # Compute cumulative impact
                G_cumul = network.get_graph_with_lanes(cumulative_edges)
                acc_orig_cumul, acc_dest_cumul, _ = compute_accessibility(
                    G_cumul, network.node_coords, network.node_tree, network.node_ids,
                    areas_proj, DEFAULT_THETA, DEFAULT_K, DEFAULT_YEAR
                )

                cumul_imp_orig = np.zeros(len(areas_proj))
                for i in range(len(areas_proj)):
                    if acc_orig_base[i] > 0:
                        cumul_imp_orig[i] = ((acc_orig_cumul[i] - acc_orig_base[i]) / acc_orig_base[i]) * 100

                cumulative_stages.append({
                    'imp_orig': cumul_imp_orig,
                    'lane_gdfs': [li['lane_gdf'] for li in lane_impacts],
                })
                break

    # Generate Figure 2
    print("\nGenerating Figure 2: Top 5 lanes improvement...")
    top_lane_names = [li['name'] for li in lane_impacts[:5]]
    top_lanes_gdf = [li['lane_gdf'] for li in lane_impacts[:5]]

    final_cumul = cumulative_stages[-1] if cumulative_stages else {'imp_orig': np.zeros(len(areas_proj))}

    # Compute final cumulative destination
    G_cumul_final = network.get_graph_with_lanes(cumulative_edges)
    _, acc_dest_cumul_final, _ = compute_accessibility(
        G_cumul_final, network.node_coords, network.node_tree, network.node_ids,
        areas_proj, DEFAULT_THETA, DEFAULT_K, DEFAULT_YEAR
    )
    cumul_imp_dest = np.zeros(len(areas_proj))
    for i in range(len(areas_proj)):
        if acc_dest_base[i] > 0:
            cumul_imp_dest[i] = ((acc_dest_cumul_final[i] - acc_dest_base[i]) / acc_dest_base[i]) * 100

    generate_figure_2(areas, final_cumul['imp_orig'], cumul_imp_dest, top_lane_names,
                       completed, construction, top_lanes_gdf,
                       FIGURES_DIR / 'figure2_improvement_top5.pdf')

    # Generate Figure 3: Individual lane impacts (top 3)
    print("\nGenerating Figure 3: Individual lane impacts...")
    for i, impact in enumerate(lane_impacts[:3]):
        safe_name = impact['name'].replace(' ', '_').replace('/', '-')[:20]
        generate_figure_3(areas, impact['name'], impact['imp_orig'], impact['imp_dest'],
                           impact['lane_gdf'], completed, construction,
                           FIGURES_DIR / f'figure3_{i+1}_lane_impact_{safe_name}.pdf')

    # Generate Figure 4: Top 3 comparison
    print("\nGenerating Figure 4: Top 3 comparison...")
    generate_figure_4(areas, lane_impacts[:3], completed, construction,
                       FIGURES_DIR / 'figure4_top_lanes_comparison.pdf')

    # Generate Figure 5: Cumulative progression
    print("\nGenerating Figure 5: Cumulative progression...")
    # Select stages: 1, 2, 3, 5 lanes
    stages_for_fig5 = []
    for stage_idx in [0, 1, 2, 4]:
        if stage_idx < len(cumulative_stages):
            stages_for_fig5.append(cumulative_stages[stage_idx])

    if stages_for_fig5:
        generate_figure_5(areas, stages_for_fig5, completed, construction,
                           FIGURES_DIR / 'figure5_cumulative_impact.pdf')

    # Generate tables
    print("\nGenerating tables...")

    with open(TABLES_DIR / 'table5_rankings.tex', 'w') as f:
        f.write(generate_table_5_rankings(rankings_default))
    print("  Saved table5_rankings.tex")

    with open(TABLES_DIR / 'table6_sensitivity_k.tex', 'w') as f:
        f.write(generate_table_6_sensitivity_k(rankings_by_k))
    print("  Saved table6_sensitivity_k.tex")

    elapsed = time.time() - start_time
    print(f"\nDone! Total time: {elapsed:.1f} seconds")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == '__main__':
    main()
