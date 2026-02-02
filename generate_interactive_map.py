"""Generate interactive HTML map for bike lane analysis."""
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import json
from pathlib import Path
from scipy.spatial import cKDTree
import fiona
import warnings

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326

def load_data():
    """Load all required data."""
    areas = gpd.read_file(script_dir / "jer_areas.shp")
    areas = areas[areas['in_jeru'] == 1].copy()
    areas['pop'] = areas['pop_2025'].fillna(0)
    areas['emp'] = areas['emp_2025'].fillna(0)

    roads = gpd.read_file(script_dir / "jerusalem_roads.kml", driver='KML')
    completed = gpd.read_file(script_dir / "bike_lanes_completed.kml", driver='KML')
    construction = gpd.read_file(script_dir / "bike_lanes_construction.kml", driver='KML')
    wishing = gpd.read_file(script_dir / "bike_lanes_wishing_list.kml", driver='KML')

    return areas, roads, completed, construction, wishing


def build_network(roads_gdf, bike_lanes_list, tolerance=15):
    """Build network with bike lanes."""
    roads_proj = roads_gdf.to_crs(TARGET_CRS)
    G = nx.Graph()
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]

    def get_or_create_node(x, y):
        key = (round(x / tolerance) * tolerance, round(y / tolerance) * tolerance)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        G.add_node(nid, x=x, y=y)
        return nid

    # Add roads
    for _, row in roads_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) >= 2:
            start = get_or_create_node(coords[0][0], coords[0][1])
            end = get_or_create_node(coords[-1][0], coords[-1][1])
            if start != end:
                G.add_edge(start, end, length=geom.length, has_bike_lane=False)

    # Add bike lanes
    for bl_gdf in bike_lanes_list:
        if bl_gdf is None or len(bl_gdf) == 0:
            continue
        bl_proj = bl_gdf.to_crs(TARGET_CRS)
        for _, row in bl_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or geom.geom_type != 'LineString':
                continue
            coords = list(geom.coords)
            if len(coords) >= 2:
                start = get_or_create_node(coords[0][0], coords[0][1])
                end = get_or_create_node(coords[-1][0], coords[-1][1])
                if start != end:
                    if G.has_edge(start, end):
                        G[start][end]['has_bike_lane'] = True
                    else:
                        G.add_edge(start, end, length=geom.length, has_bike_lane=True)

    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_array) if len(coords_array) > 0 else None

    return G, node_coords, node_tree, node_ids


def compute_area_accessibility(G, node_coords, node_tree, node_ids, areas_gdf, theta, k):
    """Compute accessibility for each area."""
    G = G.copy()
    for u, v in G.edges():
        length = G[u][v]['length']
        has_bike = G[u][v].get('has_bike_lane', False)
        G[u][v]['weight'] = length if has_bike else length * k

    areas_proj = areas_gdf.to_crs(TARGET_CRS)
    centroids = areas_proj.geometry.centroid
    n_areas = len(areas_proj)

    center_nodes = []
    for centroid in centroids:
        if node_tree is not None:
            _, idx = node_tree.query([centroid.x, centroid.y])
            center_nodes.append(node_ids[idx])
        else:
            center_nodes.append(None)

    pop = areas_proj['pop'].values
    emp = areas_proj['emp'].values

    largest_cc = max(nx.connected_components(G), key=len) if len(G) > 0 else set()

    # Compute accessibility for each area
    accessibility = np.zeros(n_areas)

    for i in range(n_areas):
        if center_nodes[i] is None or center_nodes[i] not in largest_cc:
            continue
        try:
            distances = nx.single_source_dijkstra_path_length(G, center_nodes[i], weight='weight')
        except:
            continue

        for j in range(n_areas):
            if i != j and center_nodes[j] in distances:
                tau = max(distances[center_nodes[j]] / 1000, 0.1)
                # Accessibility from i to j
                accessibility[i] += emp[j] * (tau ** theta)

    return accessibility


def get_shortest_path(G, node_coords, node_tree, node_ids, areas_gdf, origin_idx, dest_idx, k):
    """Get shortest path between two areas."""
    G = G.copy()
    for u, v in G.edges():
        length = G[u][v]['length']
        has_bike = G[u][v].get('has_bike_lane', False)
        G[u][v]['weight'] = length if has_bike else length * k

    areas_proj = areas_gdf.to_crs(TARGET_CRS)
    centroids = areas_proj.geometry.centroid

    if node_tree is None:
        return None

    _, origin_node_idx = node_tree.query([centroids.iloc[origin_idx].x, centroids.iloc[origin_idx].y])
    _, dest_node_idx = node_tree.query([centroids.iloc[dest_idx].x, centroids.iloc[dest_idx].y])

    origin_node = node_ids[origin_node_idx]
    dest_node = node_ids[dest_node_idx]

    try:
        path = nx.shortest_path(G, origin_node, dest_node, weight='weight')
        path_coords = [node_coords[n] for n in path]
        return path_coords
    except:
        return None


def main():
    print("Loading data...")
    areas, roads, completed, construction, wishing = load_data()

    # Convert to WGS84 for web display
    areas_wgs = areas.to_crs(WGS84)
    completed_wgs = completed.to_crs(WGS84) if len(completed) > 0 else completed
    construction_wgs = construction.to_crs(WGS84) if len(construction) > 0 else construction
    wishing_wgs = wishing.to_crs(WGS84) if len(wishing) > 0 else wishing

    # Export areas to GeoJSON
    print("Exporting areas...")
    areas_export = areas_wgs[['geometry', 'pop', 'emp', 'name']].copy()
    areas_export['area_id'] = range(len(areas_export))
    areas_export.to_file(script_dir / 'areas.geojson', driver='GeoJSON')

    # Export bike lanes to GeoJSON
    print("Exporting bike lanes...")
    if len(completed_wgs) > 0:
        completed_export = completed_wgs[['geometry', 'Name']].copy()
        completed_export.to_file(script_dir / 'completed.geojson', driver='GeoJSON')

    if len(construction_wgs) > 0:
        construction_export = construction_wgs[['geometry', 'Name']].copy()
        construction_export.to_file(script_dir / 'construction.geojson', driver='GeoJSON')

    if len(wishing_wgs) > 0:
        wishing_export = wishing_wgs[['geometry', 'Name']].copy()
        wishing_export['lane_id'] = range(len(wishing_export))
        wishing_export.to_file(script_dir / 'wishing.geojson', driver='GeoJSON')

    # Load sensitivity analysis results
    print("Loading sensitivity analysis results...")
    sens_df = pd.read_csv(script_dir / 'sensitivity_analysis.csv')

    # Create sensitivity data JSON
    sens_data = {}
    for _, row in sens_df.iterrows():
        key = f"{row['K']}_{row['theta']}"
        if key not in sens_data:
            sens_data[key] = {}
        sens_data[key][row['lane']] = row['improvement_pct']

    with open(script_dir / 'sensitivity_data.json', 'w', encoding='utf-8') as f:
        json.dump(sens_data, f, ensure_ascii=False)

    # Pre-compute area accessibility for baseline (no wishing lanes)
    print("Computing baseline accessibility...")
    areas_proj = areas.to_crs(TARGET_CRS)

    # Build baseline network
    G_base, nc, nt, ni = build_network(roads.to_crs(TARGET_CRS), [completed, construction])

    # Compute accessibility for each K/theta combination
    K_VALUES = [10, 50, 100, 200, 500]
    THETA_VALUES = [-0.5, -1, -1.5, -2, -3]

    accessibility_data = {}
    for K in K_VALUES:
        for theta in THETA_VALUES:
            print(f"  Computing K={K}, theta={theta}...")
            acc = compute_area_accessibility(G_base, nc, nt, ni, areas, theta, K)
            key = f"{K}_{theta}"
            accessibility_data[key] = acc.tolist()

    with open(script_dir / 'accessibility_data.json', 'w', encoding='utf-8') as f:
        json.dump(accessibility_data, f)

    # Create area centroids for path finding
    centroids_wgs = areas_wgs.geometry.centroid
    centroids_data = [[c.x, c.y] for c in centroids_wgs]
    with open(script_dir / 'centroids.json', 'w', encoding='utf-8') as f:
        json.dump(centroids_data, f)

    # Store node data for path finding (in WGS84)
    print("Exporting network data for path finding...")
    from pyproj import Transformer
    transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)

    # Convert node coords to WGS84
    node_coords_wgs = {}
    for nid, (x, y) in nc.items():
        lon, lat = transformer.transform(x, y)
        node_coords_wgs[nid] = [lon, lat]

    # Export edges
    edges_data = []
    for u, v, data in G_base.edges(data=True):
        edges_data.append({
            'from': u,
            'to': v,
            'length': data['length'],
            'has_bike_lane': data.get('has_bike_lane', False)
        })

    network_data = {
        'nodes': node_coords_wgs,
        'edges': edges_data,
        'node_ids': ni
    }
    with open(script_dir / 'network_data.json', 'w') as f:
        json.dump(network_data, f)

    # Generate HTML
    print("Generating HTML...")
    generate_html()

    print("Done! Open bike_analysis.html in a browser.")


def generate_html():
    """Generate the interactive HTML file."""
    html_content = '''<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ניתוח שבילי אופניים - ירושלים</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { margin: 0; font-size: 1.5em; }
        .controls {
            background: #34495e;
            padding: 10px 20px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .control-group label {
            color: white;
            font-weight: bold;
        }
        select, button {
            padding: 8px 12px;
            border: none;
            border-radius: 4px;
            font-size: 14px;
        }
        select { background: white; }
        button {
            background: #3498db;
            color: white;
            cursor: pointer;
        }
        button:hover { background: #2980b9; }
        button.active { background: #27ae60; }
        .main-content {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        .map-container {
            flex: 1;
            position: relative;
        }
        .map {
            width: 100%;
            height: 100%;
        }
        .sidebar {
            width: 350px;
            background: #ecf0f1;
            overflow-y: auto;
            padding: 15px;
        }
        .sidebar h3 {
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }
        .lane-item {
            padding: 8px;
            margin: 5px 0;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .lane-item:hover { background: #d5dbdb; }
        .lane-item.selected { background: #a9dfbf; border: 2px solid #27ae60; }
        .lane-improvement {
            font-weight: bold;
            color: #27ae60;
        }
        .legend {
            position: absolute;
            bottom: 30px;
            right: 10px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            z-index: 1000;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 5px 0;
        }
        .legend-color {
            width: 20px;
            height: 4px;
            margin-left: 10px;
        }
        .info-panel {
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            padding: 10px 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 300px;
        }
        .path-controls {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .path-controls select {
            width: 100%;
            margin: 5px 0;
        }
        .tab-buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        .tab-btn {
            flex: 1;
            padding: 10px;
            text-align: center;
        }
        .tab-btn.active {
            background: #27ae60;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .baseline-info {
            background: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>ניתוח שבילי אופניים - ירושלים</h1>
        <div>
            <span id="totalImprovement">שיפור כולל: 0%</span>
        </div>
    </div>

    <div class="controls">
        <div class="control-group">
            <label>K (עונש לכבישים ללא שביל):</label>
            <select id="kSelect">
                <option value="10">10</option>
                <option value="50">50</option>
                <option value="100" selected>100</option>
                <option value="200">200</option>
                <option value="500">500</option>
            </select>
        </div>
        <div class="control-group">
            <label>θ (פרמטר ריקבון מרחק):</label>
            <select id="thetaSelect">
                <option value="-0.5">-0.5</option>
                <option value="-1" selected>-1</option>
                <option value="-1.5">-1.5</option>
                <option value="-2">-2</option>
                <option value="-3">-3</option>
            </select>
        </div>
        <div class="control-group">
            <button id="clearSelection">נקה בחירה</button>
        </div>
    </div>

    <div class="main-content">
        <div class="map-container">
            <div id="map" class="map"></div>
            <div class="legend">
                <strong>מקרא</strong>
                <div class="legend-item">
                    <span>שבילים קיימים</span>
                    <div class="legend-color" style="background: #27ae60;"></div>
                </div>
                <div class="legend-item">
                    <span>בהקמה</span>
                    <div class="legend-color" style="background: #f39c12;"></div>
                </div>
                <div class="legend-item">
                    <span>רשימת משאלות</span>
                    <div class="legend-color" style="background: #e74c3c;"></div>
                </div>
                <div class="legend-item">
                    <span>נבחר</span>
                    <div class="legend-color" style="background: #9b59b6; height: 6px;"></div>
                </div>
            </div>
            <div class="info-panel" id="infoPanel" style="display: none;">
                <strong id="infoPanelTitle"></strong>
                <p id="infoPanelContent"></p>
            </div>
        </div>

        <div class="sidebar">
            <div class="tab-buttons">
                <button class="tab-btn active" data-tab="lanes">שבילים</button>
                <button class="tab-btn" data-tab="paths">מסלולים</button>
            </div>

            <div id="lanesTab" class="tab-content active">
                <h3>רשימת משאלות</h3>
                <div class="baseline-info">
                    לחץ על שביל במפה או ברשימה כדי לראות את השיפור הצפוי בנגישות
                </div>
                <div id="lanesList"></div>
            </div>

            <div id="pathsTab" class="tab-content">
                <h3>מסלול אופטימלי</h3>
                <div class="path-controls">
                    <label>מוצא:</label>
                    <select id="originSelect">
                        <option value="">בחר אזור מוצא...</option>
                    </select>
                    <label>יעד:</label>
                    <select id="destSelect">
                        <option value="">בחר אזור יעד...</option>
                    </select>
                    <button id="showPath" style="width: 100%; margin-top: 10px;">הצג מסלול</button>
                </div>
                <div id="pathInfo"></div>
            </div>
        </div>
    </div>

    <script>
        // Initialize map
        const map = L.map('map').setView([31.78, 35.22], 12);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        // Data storage
        let areasData = null;
        let wishingData = null;
        let completedData = null;
        let constructionData = null;
        let sensitivityData = null;
        let accessibilityData = null;
        let networkData = null;
        let centroidsData = null;

        // Layers
        let areasLayer = null;
        let completedLayer = null;
        let constructionLayer = null;
        let wishingLayer = null;
        let pathLayer = null;

        // State
        let selectedLanes = new Set();

        // Load all data
        async function loadData() {
            try {
                const [areas, wishing, completed, construction, sensitivity, accessibility, network, centroids] = await Promise.all([
                    fetch('areas.geojson').then(r => r.json()),
                    fetch('wishing.geojson').then(r => r.json()),
                    fetch('completed.geojson').then(r => r.json()).catch(() => ({type: 'FeatureCollection', features: []})),
                    fetch('construction.geojson').then(r => r.json()).catch(() => ({type: 'FeatureCollection', features: []})),
                    fetch('sensitivity_data.json').then(r => r.json()),
                    fetch('accessibility_data.json').then(r => r.json()),
                    fetch('network_data.json').then(r => r.json()),
                    fetch('centroids.json').then(r => r.json())
                ]);

                areasData = areas;
                wishingData = wishing;
                completedData = completed;
                constructionData = construction;
                sensitivityData = sensitivity;
                accessibilityData = accessibility;
                networkData = network;
                centroidsData = centroids;

                initializeLayers();
                populateLanesList();
                populateAreaSelects();
            } catch (error) {
                console.error('Error loading data:', error);
                alert('שגיאה בטעינת הנתונים');
            }
        }

        function initializeLayers() {
            // Areas layer
            areasLayer = L.geoJSON(areasData, {
                style: function(feature) {
                    return {
                        fillColor: '#3498db',
                        weight: 1,
                        opacity: 0.5,
                        color: '#2c3e50',
                        fillOpacity: 0.1
                    };
                },
                onEachFeature: function(feature, layer) {
                    const props = feature.properties;
                    layer.bindPopup(`
                        <strong>${props.name || 'אזור ' + props.area_id}</strong><br>
                        אוכלוסייה: ${Math.round(props.pop).toLocaleString()}<br>
                        תעסוקה: ${Math.round(props.emp).toLocaleString()}
                    `);
                }
            }).addTo(map);

            // Completed bike lanes
            if (completedData.features.length > 0) {
                completedLayer = L.geoJSON(completedData, {
                    style: { color: '#27ae60', weight: 3, opacity: 0.8 },
                    onEachFeature: function(feature, layer) {
                        layer.bindPopup(`<strong>שביל קיים:</strong><br>${feature.properties.Name || ''}`);
                    }
                }).addTo(map);
            }

            // Construction bike lanes
            if (constructionData.features.length > 0) {
                constructionLayer = L.geoJSON(constructionData, {
                    style: { color: '#f39c12', weight: 3, opacity: 0.8 },
                    onEachFeature: function(feature, layer) {
                        layer.bindPopup(`<strong>בהקמה:</strong><br>${feature.properties.Name || ''}`);
                    }
                }).addTo(map);
            }

            // Wishing list lanes
            wishingLayer = L.geoJSON(wishingData, {
                style: function(feature) {
                    const isSelected = selectedLanes.has(feature.properties.Name);
                    return {
                        color: isSelected ? '#9b59b6' : '#e74c3c',
                        weight: isSelected ? 5 : 3,
                        opacity: 0.8
                    };
                },
                onEachFeature: function(feature, layer) {
                    const laneName = feature.properties.Name;
                    layer.on('click', function() {
                        toggleLaneSelection(laneName);
                    });
                    layer.on('mouseover', function() {
                        showLaneInfo(laneName);
                    });
                    layer.on('mouseout', function() {
                        hideInfo();
                    });
                }
            }).addTo(map);
        }

        function populateLanesList() {
            const container = document.getElementById('lanesList');
            const k = document.getElementById('kSelect').value;
            const theta = document.getElementById('thetaSelect').value;
            const key = `${k}_${theta}`;
            const data = sensitivityData[key] || {};

            // Sort lanes by improvement
            const lanes = wishingData.features.map(f => ({
                name: f.properties.Name,
                improvement: data[f.properties.Name] || 0
            })).sort((a, b) => b.improvement - a.improvement);

            container.innerHTML = lanes.map(lane => `
                <div class="lane-item ${selectedLanes.has(lane.name) ? 'selected' : ''}"
                     data-lane="${lane.name}"
                     onclick="toggleLaneSelection('${lane.name.replace(/'/g, "\\'")}')">
                    <span>${lane.name}</span>
                    <span class="lane-improvement">+${lane.improvement.toFixed(2)}%</span>
                </div>
            `).join('');
        }

        function populateAreaSelects() {
            const originSelect = document.getElementById('originSelect');
            const destSelect = document.getElementById('destSelect');

            const options = areasData.features.map((f, i) =>
                `<option value="${i}">${f.properties.name || 'אזור ' + i}</option>`
            ).join('');

            originSelect.innerHTML = '<option value="">בחר אזור מוצא...</option>' + options;
            destSelect.innerHTML = '<option value="">בחר אזור יעד...</option>' + options;
        }

        function toggleLaneSelection(laneName) {
            if (selectedLanes.has(laneName)) {
                selectedLanes.delete(laneName);
            } else {
                selectedLanes.add(laneName);
            }
            updateDisplay();
        }

        function updateDisplay() {
            // Update lane styling
            if (wishingLayer) {
                wishingLayer.setStyle(function(feature) {
                    const isSelected = selectedLanes.has(feature.properties.Name);
                    return {
                        color: isSelected ? '#9b59b6' : '#e74c3c',
                        weight: isSelected ? 5 : 3,
                        opacity: 0.8
                    };
                });
            }

            // Update lanes list
            populateLanesList();

            // Calculate total improvement
            const k = document.getElementById('kSelect').value;
            const theta = document.getElementById('thetaSelect').value;
            const key = `${k}_${theta}`;
            const data = sensitivityData[key] || {};

            let totalImprovement = 0;
            selectedLanes.forEach(name => {
                totalImprovement += data[name] || 0;
            });

            document.getElementById('totalImprovement').textContent =
                `שיפור כולל (משוער): +${totalImprovement.toFixed(2)}%`;

            // Update area colors based on accessibility
            updateAreaColors();
        }

        function updateAreaColors() {
            const k = document.getElementById('kSelect').value;
            const theta = document.getElementById('thetaSelect').value;
            const key = `${k}_${theta}`;
            const accessibility = accessibilityData[key] || [];

            if (accessibility.length === 0) return;

            const maxAcc = Math.max(...accessibility.filter(a => a > 0));
            const minAcc = Math.min(...accessibility.filter(a => a > 0));

            areasLayer.eachLayer(function(layer) {
                const areaId = layer.feature.properties.area_id;
                const acc = accessibility[areaId] || 0;

                // Normalize to 0-1
                const normalized = maxAcc > minAcc ? (acc - minAcc) / (maxAcc - minAcc) : 0;

                // Color from red (low) to green (high)
                const r = Math.round(255 * (1 - normalized));
                const g = Math.round(255 * normalized);
                const color = `rgb(${r}, ${g}, 100)`;

                layer.setStyle({
                    fillColor: color,
                    fillOpacity: 0.4,
                    weight: 1,
                    opacity: 0.5,
                    color: '#2c3e50'
                });
            });
        }

        function showLaneInfo(laneName) {
            const k = document.getElementById('kSelect').value;
            const theta = document.getElementById('thetaSelect').value;
            const key = `${k}_${theta}`;
            const improvement = sensitivityData[key]?.[laneName] || 0;

            const panel = document.getElementById('infoPanel');
            document.getElementById('infoPanelTitle').textContent = laneName;
            document.getElementById('infoPanelContent').textContent =
                `שיפור צפוי: +${improvement.toFixed(3)}%`;
            panel.style.display = 'block';
        }

        function hideInfo() {
            document.getElementById('infoPanel').style.display = 'none';
        }

        function showPath() {
            const originIdx = document.getElementById('originSelect').value;
            const destIdx = document.getElementById('destSelect').value;

            if (!originIdx || !destIdx) {
                alert('נא לבחור אזור מוצא ויעד');
                return;
            }

            // Remove existing path
            if (pathLayer) {
                map.removeLayer(pathLayer);
            }

            // Use Dijkstra on the network
            const k = parseInt(document.getElementById('kSelect').value);
            const path = findShortestPath(parseInt(originIdx), parseInt(destIdx), k);

            if (path && path.length > 0) {
                pathLayer = L.polyline(path.map(p => [p[1], p[0]]), {
                    color: '#9b59b6',
                    weight: 6,
                    opacity: 0.8
                }).addTo(map);

                map.fitBounds(pathLayer.getBounds());

                document.getElementById('pathInfo').innerHTML = `
                    <p><strong>מסלול נמצא!</strong></p>
                    <p>מספר נקודות: ${path.length}</p>
                `;
            } else {
                document.getElementById('pathInfo').innerHTML = `
                    <p style="color: #e74c3c;">לא נמצא מסלול בין האזורים</p>
                `;
            }
        }

        function findShortestPath(originIdx, destIdx, k) {
            if (!networkData || !centroidsData) return null;

            const nodes = networkData.nodes;
            const edges = networkData.edges;
            const nodeIds = networkData.node_ids;

            // Find nearest nodes to centroids
            const originCentroid = centroidsData[originIdx];
            const destCentroid = centroidsData[destIdx];

            let originNode = null;
            let destNode = null;
            let minDistO = Infinity;
            let minDistD = Infinity;

            for (const [nodeId, coords] of Object.entries(nodes)) {
                const distO = Math.sqrt(Math.pow(coords[0] - originCentroid[0], 2) + Math.pow(coords[1] - originCentroid[1], 2));
                const distD = Math.sqrt(Math.pow(coords[0] - destCentroid[0], 2) + Math.pow(coords[1] - destCentroid[1], 2));

                if (distO < minDistO) {
                    minDistO = distO;
                    originNode = nodeId;
                }
                if (distD < minDistD) {
                    minDistD = distD;
                    destNode = nodeId;
                }
            }

            if (!originNode || !destNode) return null;

            // Build adjacency list with weights
            const adj = {};
            for (const edge of edges) {
                const weight = edge.has_bike_lane ? edge.length : edge.length * k;

                if (!adj[edge.from]) adj[edge.from] = [];
                if (!adj[edge.to]) adj[edge.to] = [];

                adj[edge.from].push({node: edge.to, weight: weight});
                adj[edge.to].push({node: edge.from, weight: weight});
            }

            // Dijkstra
            const dist = {};
            const prev = {};
            const visited = new Set();
            const pq = [{node: originNode, dist: 0}];

            dist[originNode] = 0;

            while (pq.length > 0) {
                // Get node with min distance
                pq.sort((a, b) => a.dist - b.dist);
                const {node: current, dist: currentDist} = pq.shift();

                if (visited.has(current)) continue;
                visited.add(current);

                if (current == destNode) break;

                const neighbors = adj[current] || [];
                for (const {node: neighbor, weight} of neighbors) {
                    if (visited.has(neighbor)) continue;

                    const newDist = currentDist + weight;
                    if (dist[neighbor] === undefined || newDist < dist[neighbor]) {
                        dist[neighbor] = newDist;
                        prev[neighbor] = current;
                        pq.push({node: neighbor, dist: newDist});
                    }
                }
            }

            // Reconstruct path
            if (dist[destNode] === undefined) return null;

            const path = [];
            let current = destNode;
            while (current) {
                path.unshift(nodes[current]);
                current = prev[current];
            }

            return path;
        }

        // Event listeners
        document.getElementById('kSelect').addEventListener('change', updateDisplay);
        document.getElementById('thetaSelect').addEventListener('change', updateDisplay);
        document.getElementById('clearSelection').addEventListener('click', function() {
            selectedLanes.clear();
            updateDisplay();
        });
        document.getElementById('showPath').addEventListener('click', showPath);

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

                this.classList.add('active');
                document.getElementById(this.dataset.tab + 'Tab').classList.add('active');
            });
        });

        // Initialize
        loadData();
    </script>
</body>
</html>'''

    with open(script_dir / 'bike_analysis.html', 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == '__main__':
    main()
