"""
Generate figures using Folium to match the exact visual appearance of bike_analysis.html.
This script creates interactive HTML maps that look identical to the analysis tool.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import json
import os
import folium
from folium.plugins import FloatImage
from pathlib import Path
from scipy.spatial import cKDTree
from pyproj import Transformer
import fiona
import warnings
from shapely.geometry import Point, LineString, mapping
from shapely.strtree import STRtree
import time
from branca.colormap import LinearColormap
from branca.element import Template, MacroElement

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326
NODE_TOLERANCE = 15

# Output directories
FIGURES_DIR = script_dir / 'figures'
TABLES_DIR = script_dir / 'tables'
HTML_DIR = script_dir / 'html_figures'

# Default parameters (matching the HTML defaults)
DEFAULT_K = 100
DEFAULT_THETA = -1.0
DEFAULT_YEAR = 2025

# Colors matching HTML exactly
LANE_COLORS = {
    'existing': '#1B5E20',      # Dark green
    'construction': '#81C784',  # Light green
    'planning': '#2196F3',      # Blue
    'checking': '#00BCD4',      # Cyan
    'wishing': '#FF9800',       # Orange
    'selected': '#9b59b6',      # Purple
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
    """Convert Hebrew lane name to English."""
    return LANE_TRANSLITERATION.get(name, name)


def spectral_color(t):
    """Convert normalized value t (0-1) to hex color matching HTML spectral gradient exactly."""
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
    """Load all data files."""
    print("Loading data...")
    areas = gpd.read_file(script_dir / "jer_areas.shp")
    areas = areas[areas['in_jeru'] == 1].copy()

    for year in [2020, 2025, 2030, 2035, 2040]:
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
    """Compute accessibility metric - matching HTML exactly."""
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
    """Compute lane rankings - matching HTML exactly."""
    print(f"  Computing rankings for K={k}, theta={theta}, year={year}...")

    wishing_proj = wishing.to_crs(TARGET_CRS)

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

        lane_edges = network.mark_bike_lanes([wishing_proj.iloc[[idx]]])
        all_edges = baseline_edges | lane_edges

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

    results.sort(key=lambda x: x['improvement_pct'], reverse=True)

    cumulative = 0.0
    for i, r in enumerate(results):
        r['rank'] = i + 1
        cumulative += r['improvement_pct']
        r['cumulative_pct'] = cumulative

    return results, baseline_N


def create_folium_map(areas_wgs84, values, mode='accessibility', title='',
                      bike_lanes=None, highlight_lanes=None, center=None, zoom=12):
    """Create a Folium map matching HTML exactly."""

    if center is None:
        bounds = areas_wgs84.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=zoom, tiles='cartodbpositron')

    # Filter to positive values for log scaling
    pos_values = values[values > 0]

    if len(pos_values) == 0:
        # All gray
        for idx, row in areas_wgs84.iterrows():
            if row.geometry is not None and not row.geometry.is_empty:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': '#BEBEBE',
                        'fillOpacity': 0.3,
                        'weight': 1,
                        'opacity': 0.5,
                        'color': '#2c3e50'
                    }
                ).add_to(m)
    else:
        if mode == 'accessibility':
            log_values = np.log(pos_values)
            mn_log = log_values.min()
            mx_log = log_values.max()
            fill_opacity = 0.5
        else:  # change mode
            log_values = np.log(pos_values + 1)
            mx_log = log_values.max()
            mn_log = 0
            fill_opacity = 0.6

        # Add each area
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
                    style_function=lambda x, c=color, a=alpha: {
                        'fillColor': c,
                        'fillOpacity': a,
                        'weight': 1,
                        'opacity': 0.5,
                        'color': '#2c3e50'
                    }
                ).add_to(m)

    # Add bike lanes
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
                    folium.PolyLine(
                        coords,
                        color=color,
                        weight=4,
                        opacity=0.8
                    ).add_to(m)

    # Highlight specific lanes (e.g., wishing lanes being analyzed)
    if highlight_lanes is not None:
        for lane_gdf, lane_color in highlight_lanes:
            if lane_gdf is None or len(lane_gdf) == 0:
                continue
            layer_wgs84 = lane_gdf.to_crs(WGS84) if lane_gdf.crs != WGS84 else lane_gdf

            for _, row in layer_wgs84.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms) if geom.geom_type == 'MultiLineString' else []
                for line in lines:
                    coords = [(c[1], c[0]) for c in line.coords]
                    folium.PolyLine(
                        coords,
                        color=lane_color,
                        weight=6,
                        opacity=1.0
                    ).add_to(m)

    # Add title
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                background: white; padding: 10px 20px; border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3); z-index: 1000;
                font-family: Arial, sans-serif; font-size: 14px; font-weight: bold;">
        {title}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Add legend for spectral colormap
    if mode == 'accessibility' and len(pos_values) > 0:
        legend_html = f'''
        <div style="position: fixed; bottom: 30px; right: 10px;
                    background: white; padding: 10px; border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.3); z-index: 1000;
                    font-family: Arial, sans-serif; font-size: 12px;">
            <b>Accessibility (log scale)</b><br>
            <div style="display: flex; align-items: center; margin-top: 5px;">
                <span style="margin-right: 5px;">Low</span>
                <div style="width: 100px; height: 15px;
                     background: linear-gradient(to right, #0000CD, #00CED1, #FFFF00, #FFA500, #DC143C);"></div>
                <span style="margin-left: 5px;">High</span>
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
    elif mode == 'change':
        legend_html = '''
        <div style="position: fixed; bottom: 30px; right: 10px;
                    background: white; padding: 10px; border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.3); z-index: 1000;
                    font-family: Arial, sans-serif; font-size: 12px;">
            <b>Improvement % (log scale)</b><br>
            <div style="display: flex; align-items: center; margin-top: 5px;">
                <span style="margin-right: 5px;">0%</span>
                <div style="width: 100px; height: 15px;
                     background: linear-gradient(to right, #0000CD, #00CED1, #FFFF00, #FFA500, #DC143C);"></div>
                <span style="margin-left: 5px;">Max</span>
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

    return m


def main():
    """Main function to generate all figures and tables."""
    start_time = time.time()

    # Create output directories
    FIGURES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)
    HTML_DIR.mkdir(exist_ok=True)
    print(f"Output directories: {FIGURES_DIR}, {TABLES_DIR}, {HTML_DIR}")

    # Load data
    areas, roads, completed, construction, plan, check, wishing = load_data()
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

    # Define bike lane layers for visualization
    bike_lane_layers = [
        ('existing', completed),
        ('construction', construction),
    ]

    # === FIGURE 1: Baseline Accessibility ===
    print("\nGenerating Figure 1: Baseline Accessibility...")

    # Origin accessibility
    m1_orig = create_folium_map(
        areas_wgs84, acc_orig_base, mode='accessibility',
        title=f'Baseline Origin Accessibility (K={DEFAULT_K}, θ={DEFAULT_THETA})',
        bike_lanes=bike_lane_layers
    )
    m1_orig.save(HTML_DIR / 'figure1_baseline_origin.html')
    print(f"  Saved figure1_baseline_origin.html")

    # Destination accessibility
    m1_dest = create_folium_map(
        areas_wgs84, acc_dest_base, mode='accessibility',
        title=f'Baseline Destination Accessibility (K={DEFAULT_K}, θ={DEFAULT_THETA})',
        bike_lanes=bike_lane_layers
    )
    m1_dest.save(HTML_DIR / 'figure1_baseline_dest.html')
    print(f"  Saved figure1_baseline_dest.html")

    # === COMPUTE LANE RANKINGS ===
    print("\nComputing lane rankings...")
    rankings_default, _ = compute_lane_rankings(
        network, areas_proj, wishing, baseline_edges,
        DEFAULT_THETA, DEFAULT_K, DEFAULT_YEAR
    )

    # Save rankings to JSON for reference
    with open(HTML_DIR / 'rankings.json', 'w') as f:
        json.dump(rankings_default, f, indent=2)
    print(f"  Saved rankings.json")

    # Print top 5
    print("\n  Top 5 lanes:")
    for r in rankings_default[:5]:
        print(f"    {r['rank']}. {r['name']}: +{r['improvement_pct']:.4f}%")

    # === FIGURES 2-5: Individual and cumulative lane impacts ===
    wishing_proj = wishing.to_crs(TARGET_CRS)
    wishing_wgs84 = wishing.to_crs(WGS84)

    lane_impacts = []
    cumulative_edges = set(baseline_edges)

    for rank_idx, r in enumerate(rankings_default[:5]):
        print(f"\nComputing impact for lane {rank_idx+1}: {r['name']}...")

        # Find lane geometry
        for idx in range(len(wishing_proj)):
            raw_name = wishing_proj.iloc[idx].get('Name')
            if raw_name == r.get('name_raw') or transliterate_name(raw_name) == r['name']:
                # Compute single lane impact
                single_lane_edges = network.mark_bike_lanes([wishing_proj.iloc[[idx]]])
                edges_with_single = baseline_edges | single_lane_edges

                G_single = network.get_graph_with_lanes(edges_with_single)
                acc_orig_single, acc_dest_single, _ = compute_accessibility(
                    G_single, network.node_coords, network.node_tree, network.node_ids,
                    areas_proj, DEFAULT_THETA, DEFAULT_K, DEFAULT_YEAR
                )

                # Compute improvements (percentage)
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

                # Generate individual lane impact figure
                m_lane = create_folium_map(
                    areas_wgs84, imp_orig, mode='change',
                    title=f"Impact of {r['name']} - Origin Accessibility Change (%)",
                    bike_lanes=bike_lane_layers,
                    highlight_lanes=[(wishing_wgs84.iloc[[idx]], LANE_COLORS['wishing'])]
                )
                safe_name = r['name'].replace(' ', '_').replace('/', '-')[:30]
                m_lane.save(HTML_DIR / f'figure3_{rank_idx+1}_{safe_name}_origin.html')
                print(f"  Saved figure3_{rank_idx+1} (origin)")

                m_lane_dest = create_folium_map(
                    areas_wgs84, imp_dest, mode='change',
                    title=f"Impact of {r['name']} - Destination Accessibility Change (%)",
                    bike_lanes=bike_lane_layers,
                    highlight_lanes=[(wishing_wgs84.iloc[[idx]], LANE_COLORS['wishing'])]
                )
                m_lane_dest.save(HTML_DIR / f'figure3_{rank_idx+1}_{safe_name}_dest.html')
                print(f"  Saved figure3_{rank_idx+1} (dest)")

                # Update cumulative edges
                cumulative_edges |= single_lane_edges
                break

    # === FIGURE 2: Cumulative Impact of Top 5 ===
    print("\nGenerating Figure 2: Cumulative impact of top 5 lanes...")
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

    # Collect all top 5 lane geometries
    top5_lanes = []
    for li in lane_impacts[:5]:
        top5_lanes.append((li['lane_gdf'], LANE_COLORS['wishing']))

    top_lane_names = ', '.join([li['name'] for li in lane_impacts[:3]])
    m2_orig = create_folium_map(
        areas_wgs84, cumul_imp_orig, mode='change',
        title=f'Cumulative Origin Improvement with Top 5 Lanes ({top_lane_names}...)',
        bike_lanes=bike_lane_layers,
        highlight_lanes=top5_lanes
    )
    m2_orig.save(HTML_DIR / 'figure2_top5_origin.html')
    print(f"  Saved figure2_top5_origin.html")

    m2_dest = create_folium_map(
        areas_wgs84, cumul_imp_dest, mode='change',
        title=f'Cumulative Destination Improvement with Top 5 Lanes',
        bike_lanes=bike_lane_layers,
        highlight_lanes=top5_lanes
    )
    m2_dest.save(HTML_DIR / 'figure2_top5_dest.html')
    print(f"  Saved figure2_top5_dest.html")

    # === Generate LaTeX tables ===
    print("\nGenerating LaTeX tables...")

    # Table 5: Lane rankings
    table_rows = []
    for r in rankings_default:
        name = str(r['name']).replace('&', r'\&').replace('_', r'\_')
        table_rows.append(f"{r['rank']} & {name} & {r['improvement_pct']:.4f} & {r['cumulative_pct']:.4f}")

    table5 = r"""\begin{table}[H]
\centering
\caption{Lane Rankings -- Additive Mode ($K=""" + str(DEFAULT_K) + r"""$, $\theta=""" + str(DEFAULT_THETA) + r"""$)}
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
    with open(TABLES_DIR / 'table5_rankings.tex', 'w') as f:
        f.write(table5)
    print("  Saved table5_rankings.tex")

    # Save summary
    summary = {
        'baseline_N': baseline_N,
        'k': DEFAULT_K,
        'theta': DEFAULT_THETA,
        'year': DEFAULT_YEAR,
        'n_areas': len(areas),
        'n_lanes': len(wishing),
        'top5': [{'name': r['name'], 'improvement_pct': r['improvement_pct']} for r in rankings_default[:5]]
    }
    with open(HTML_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nDone! Total time: {elapsed:.1f} seconds")
    print(f"HTML figures saved to: {HTML_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"\nOpen the HTML files in a browser to see visualizations identical to bike_analysis.html")


if __name__ == '__main__':
    main()
