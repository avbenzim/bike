"""
Generate a figure with 4 subfigures showing the effect of the top 4 ranked lanes
on accessibility value. Uses the same visualization approach as the interactive HTML:
- Log-transformed values for normalization
- Spectral color scale (blue-cyan-yellow-orange-red)
- Shows both origin and destination accessibility changes

Output: A 4x2 grid (4 lanes x 2 panels: origin/destination)
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import json
import os
import folium
from pathlib import Path
from scipy.spatial import cKDTree
import fiona
import warnings
from shapely.geometry import Point
from shapely.strtree import STRtree
import time

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
data_dir = script_dir.parent
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326
NODE_TOLERANCE = 15

# Output directories
FIGURES_DIR = script_dir / 'figures'
HTML_DIR = script_dir / 'html_figures'

# Default parameters matching HTML
DEFAULT_K = 5
DEFAULT_THETA = -1.0
DEFAULT_YEAR = 2025

# Colors matching HTML exactly
LANE_COLORS = {
    'existing': '#1B5E20',
    'construction': '#81C784',
    'wishing': '#FF9800',
    'highlight': '#E91E63',  # Pink for highlighted lane
}

# Hebrew to English transliteration
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


def spectral_color(t):
    """
    Convert normalized value t (0-1) to hex color using spectral gradient.
    This matches the JavaScript implementation in the interactive HTML exactly.

    Color stops:
    - 0.00: Blue (#0000CD) - Medium Blue
    - 0.25: Cyan (#00CED1) - Dark Turquoise
    - 0.50: Yellow (#FFFF00)
    - 0.75: Orange (#FFA500)
    - 1.00: Red (#DC143C) - Crimson
    """
    stops = [
        (0.0, (0, 0, 205)),      # Medium Blue
        (0.25, (0, 206, 209)),   # Dark Turquoise
        (0.5, (255, 255, 0)),    # Yellow
        (0.75, (255, 165, 0)),   # Orange
        (1.0, (220, 20, 60))     # Crimson
    ]
    t = max(0.0, min(1.0, t))
    i = 0
    while i < len(stops) - 1 and stops[i + 1][0] < t:
        i += 1
    if i >= len(stops) - 1:
        r, g, b = stops[-1][1]
        return f'#{r:02x}{g:02x}{b:02x}'
    t0, c0 = stops[i]
    t1, c1 = stops[i + 1]
    f = (t - t0) / (t1 - t0) if t1 != t0 else 0
    r = int(c0[0] + (c1[0] - c0[0]) * f)
    g = int(c0[1] + (c1[1] - c0[1]) * f)
    b = int(c0[2] + (c1[2] - c0[2]) * f)
    return f'#{r:02x}{g:02x}{b:02x}'


def load_data():
    """Load all geographic data files."""
    print("Loading data...")
    areas = gpd.read_file(data_dir / "jer_areas.shp")
    areas = areas[areas['in_jeru'] == 1].copy()
    for year in [2020, 2025, 2030, 2035, 2040]:
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
    """Build and manage the road network graph with bike lane information."""

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
    """
    Compute accessibility using the gravity model formula:
    N = Σᵢ Σⱼ Pᵢ × Eⱼ × τᵢⱼ^θ

    where:
    - Pᵢ = population at origin i
    - Eⱼ = employment at destination j
    - τᵢⱼ = travel cost (shortest path with K-penalty)
    - θ = distance decay parameter
    """
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


def create_folium_map(areas_wgs84, values, mode='change', title='',
                      bike_lanes=None, highlight_lane=None, center=None, zoom=12.3,
                      show_legend=True, panel_label=None):
    """
    Create a Folium map with styling matching the interactive HTML.

    Uses log transformation for value normalization:
    - For 'change' mode: log(value + 1), min=0
    - For 'accessibility' mode: log(value), using actual min/max
    """
    if center is None:
        bounds = areas_wgs84.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=zoom, tiles='cartodbpositron',
                   width='100%', height='100%', zoom_control=False,
                   scrollWheelZoom=False, dragging=False)

    pos_values = values[values > 0]

    if len(pos_values) == 0:
        for idx, row in areas_wgs84.iterrows():
            if row.geometry is not None and not row.geometry.is_empty:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x: {'fillColor': '#BEBEBE', 'fillOpacity': 0.3, 'weight': 1, 'opacity': 0.5, 'color': '#2c3e50'}
                ).add_to(m)
    else:
        # Log transformation matching HTML implementation
        if mode == 'accessibility':
            log_values = np.log(pos_values)
            mn_log = log_values.min()
            mx_log = log_values.max()
            fill_opacity = 0.6
        else:  # mode == 'change'
            # For change/improvement values, use log(v+1) with min=0
            log_values = np.log(pos_values + 1)
            mx_log = log_values.max()
            mn_log = 0
            fill_opacity = 0.7

        for idx, row in areas_wgs84.iterrows():
            v = values[idx] if idx < len(values) else 0
            if v <= 0:
                color = '#BEBEBE'
                alpha = 0.3 if mode == 'change' else 0.5
            else:
                if mode == 'accessibility':
                    log_v = np.log(v)
                    n = (log_v - mn_log) / (mx_log - mn_log) if mx_log > mn_log else 0
                else:
                    log_v = np.log(v + 1)
                    n = log_v / mx_log if mx_log > 0 else 0
                color = spectral_color(n)
                alpha = fill_opacity

            if row.geometry is not None and not row.geometry.is_empty:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x, c=color, a=alpha: {'fillColor': c, 'fillOpacity': a, 'weight': 1, 'opacity': 0.7, 'color': '#2c3e50'}
                ).add_to(m)

    # Add existing bike lanes
    if bike_lanes is not None:
        for layer_name, layer_gdf in bike_lanes:
            if layer_gdf is None or len(layer_gdf) == 0:
                continue
            color = LANE_COLORS.get(layer_name, '#888888')
            layer_wgs84 = layer_gdf.to_crs(WGS84) if layer_gdf.crs != WGS84 else layer_gdf
            for _, row in layer_wgs84.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms) if geom.geom_type == 'MultiLineString' else []
                for line in lines:
                    coords = [(c[1], c[0]) for c in line.coords]
                    folium.PolyLine(coords, color=color, weight=3, opacity=0.8).add_to(m)

    # Highlight the specific lane being analyzed
    if highlight_lane is not None:
        lane_gdf, lane_color = highlight_lane
        if lane_gdf is not None and len(lane_gdf) > 0:
            layer_wgs84 = lane_gdf.to_crs(WGS84) if lane_gdf.crs != WGS84 else lane_gdf
            for _, row in layer_wgs84.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms) if geom.geom_type == 'MultiLineString' else []
                for line in lines:
                    coords = [(c[1], c[0]) for c in line.coords]
                    folium.PolyLine(coords, color=lane_color, weight=6, opacity=1.0).add_to(m)

    # Add panel label
    if panel_label:
        label_html = f'''
        <div style="position: fixed; top: 8px; left: 8px;
                    background: rgba(255,255,255,0.95); padding: 6px 12px; border-radius: 4px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3); z-index: 1000;
                    font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; font-weight: 600;
                    color: #2c3e50;">
            {panel_label}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(label_html))

    # Add legend (only for first panel in each row)
    if show_legend:
        legend_html = '''
        <div style="position: fixed; bottom: 15px; right: 8px;
                    background: rgba(255,255,255,0.95); padding: 8px; border-radius: 4px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3); z-index: 1000;
                    font-family: 'Segoe UI', Arial, sans-serif; font-size: 10px;">
            <div style="font-weight: 600; margin-bottom: 4px; color: #2c3e50;">Improvement %</div>
            <div style="display: flex; align-items: center;">
                <span style="color: #666;">Low</span>
                <div style="width: 60px; height: 10px; margin: 0 4px;
                     background: linear-gradient(to right, #0000CD, #00CED1, #FFFF00, #FFA500, #DC143C);
                     border-radius: 2px;"></div>
                <span style="color: #666;">High</span>
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

    return m


def create_top4_figure_html(lane_impacts, areas_wgs84, bike_lane_layers, output_path):
    """
    Create the main figure: 4 rows (one per lane) x 2 columns (origin/destination).
    Each panel shows the accessibility improvement (%) from adding that lane.
    """

    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                background: #f5f5f5;
            }
            .container {
                width: 1400px;
                display: flex;
                flex-direction: column;
                background: white;
            }
            .title {
                text-align: center;
                padding: 16px;
                font-size: 18px;
                font-weight: 600;
                color: #2c3e50;
                background: white;
                border-bottom: 2px solid #3498db;
            }
            .subtitle {
                text-align: center;
                padding: 8px;
                font-size: 13px;
                color: #7f8c8d;
                background: #ecf0f1;
                border-bottom: 1px solid #ddd;
            }
            .header-row {
                display: flex;
                background: #3498db;
                color: white;
            }
            .header-cell {
                flex: 1;
                text-align: center;
                padding: 12px;
                font-weight: 600;
                font-size: 14px;
            }
            .header-cell:first-child { border-right: 1px solid rgba(255,255,255,0.3); }
            .grid {
                display: flex;
                flex-direction: column;
            }
            .row {
                display: flex;
                height: 320px;
                border-bottom: 1px solid #ddd;
            }
            .row:last-child { border-bottom: none; }
            .cell {
                flex: 1;
                position: relative;
            }
            .cell:first-child { border-right: 1px solid #ddd; }
            .row-label {
                position: absolute;
                left: 8px;
                top: 8px;
                background: rgba(255,255,255,0.95);
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 600;
                color: #2c3e50;
                z-index: 1000;
                box-shadow: 0 2px 6px rgba(0,0,0,0.25);
                border-left: 4px solid #E91E63;
            }
            .improvement-badge {
                position: absolute;
                right: 8px;
                top: 8px;
                background: #27ae60;
                color: white;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
                z-index: 1000;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            .footer {
                padding: 12px;
                background: #ecf0f1;
                font-size: 11px;
                color: #7f8c8d;
                text-align: center;
                border-top: 1px solid #ddd;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="title">Effect of Top 4 Ranked Bike Lanes on Accessibility Value</div>
            <div class="subtitle">Log-scaled visualization showing percentage improvement in accessibility (K=5, θ=-1.0)</div>
            <div class="header-row">
                <div class="header-cell">Origin Accessibility (Jobs Reachable)</div>
                <div class="header-cell">Destination Accessibility (People Reaching)</div>
            </div>
            <div class="grid">
    '''

    for i, impact in enumerate(lane_impacts[:4]):
        m_left = create_folium_map(
            areas_wgs84,
            impact['imp_orig'],
            mode='change',
            bike_lanes=bike_lane_layers,
            highlight_lane=(impact['lane_gdf'], LANE_COLORS['highlight']),
            panel_label=None,
            show_legend=(i == 0)
        )
        m_right = create_folium_map(
            areas_wgs84,
            impact['imp_dest'],
            mode='change',
            bike_lanes=bike_lane_layers,
            highlight_lane=(impact['lane_gdf'], LANE_COLORS['highlight']),
            panel_label=None,
            show_legend=False
        )

        html_content += f'''
            <div class="row">
                <div class="cell">
                    <div class="row-label">#{impact['rank']}. {impact['name']}</div>
                    <div class="improvement-badge">+{impact['improvement_pct']:.2f}%</div>
                    {m_left._repr_html_()}
                </div>
                <div class="cell">
                    {m_right._repr_html_()}
                </div>
            </div>
        '''

    html_content += '''
            </div>
            <div class="footer">
                Colors show log-transformed accessibility improvement.
                Pink line indicates the added lane.
                Green = existing bike infrastructure.
            </div>
        </div>
    </body>
    </html>
    '''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path


def capture_html_to_png(html_path, png_path, width=1400, height=1400):
    """Capture HTML as PNG using Playwright."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': width, 'height': height})
            page.goto(f'file://{html_path.absolute()}')
            page.wait_for_timeout(3000)
            page.screenshot(path=str(png_path), full_page=True)
            browser.close()
        print(f"  Generated PNG: {png_path}")
    except Exception as e:
        print(f"  Warning: Could not generate PNG ({e}). HTML file was generated successfully.")


def main():
    start_time = time.time()

    FIGURES_DIR.mkdir(exist_ok=True)
    HTML_DIR.mkdir(exist_ok=True)
    print(f"Output directories: {FIGURES_DIR}, {HTML_DIR}")

    # Load data
    areas, roads, completed, construction, wishing = load_data()
    areas_proj = areas.to_crs(TARGET_CRS)
    areas_wgs84 = areas.to_crs(WGS84)
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

    bike_lane_layers = [('existing', completed), ('construction', construction)]

    # Load rankings
    rankings_path = HTML_DIR / 'rankings.json'
    if rankings_path.exists():
        with open(rankings_path) as f:
            rankings = json.load(f)
        print(f"Loaded rankings from {rankings_path}")
    else:
        print("Rankings file not found, computing...")
        # Compute rankings if not available
        from generate_html_figures import compute_lane_rankings
        rankings, _ = compute_lane_rankings(network, areas_proj, wishing, baseline_edges,
                                            DEFAULT_THETA, DEFAULT_K, DEFAULT_YEAR)

    print("\nTop 4 lanes:")
    for r in rankings[:4]:
        print(f"  {r['rank']}. {r['name']}: +{r['improvement_pct']:.4f}%")

    # Compute individual lane impacts for top 4
    wishing_proj = wishing.to_crs(TARGET_CRS)
    wishing_wgs84 = wishing.to_crs(WGS84)

    lane_impacts = []

    for rank_idx, r in enumerate(rankings[:4]):
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

                # Calculate percentage improvement for each area
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
                    'improvement_pct': r['improvement_pct'],
                    'length': r['length'],
                    'imp_orig': imp_orig,
                    'imp_dest': imp_dest,
                    'lane_gdf': wishing_wgs84.iloc[[idx]],
                })

                print(f"  Max origin improvement: {imp_orig.max():.2f}%")
                print(f"  Max destination improvement: {imp_dest.max():.2f}%")
                break

    # Generate the 4x2 figure
    print("\n" + "="*60)
    print("Generating Figure: Top 4 Lanes Effect on Accessibility Value")
    print("="*60)

    html_path = HTML_DIR / 'figure_top4_lanes_effect.html'
    create_top4_figure_html(lane_impacts, areas_wgs84, bike_lane_layers, html_path)
    print(f"  Saved HTML: {html_path}")

    # Capture to PNG
    png_path = FIGURES_DIR / 'figure_top4_lanes_effect.png'
    capture_html_to_png(html_path, png_path, width=1400, height=1480)

    # Also generate a PDF-ready version
    pdf_path = FIGURES_DIR / 'figure_top4_lanes_effect.pdf'
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': 1400, 'height': 1480})
            page.goto(f'file://{html_path.absolute()}')
            page.wait_for_timeout(3000)
            page.pdf(path=str(pdf_path), width='1400px', height='1480px', print_background=True)
            browser.close()
        print(f"  Generated PDF: {pdf_path}")
    except Exception as e:
        print(f"  Warning: Could not generate PDF ({e})")

    elapsed = time.time() - start_time
    print(f"\nDone! Total time: {elapsed:.1f} seconds")
    print(f"\nOutput files:")
    print(f"  HTML: {html_path}")
    print(f"  PNG:  {png_path}")
    print(f"  PDF:  {pdf_path}")


if __name__ == '__main__':
    main()
