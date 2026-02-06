"""Generate a single self-contained interactive HTML map for bike lane analysis.
All data is embedded in the HTML - no external file loading needed.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import json
from pathlib import Path
from scipy.spatial import cKDTree
from pyproj import Transformer
import fiona
import warnings

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326
NODE_TOLERANCE = 15

K_VALUES = [10, 50, 100, 200, 500]
THETA_VALUES = [-0.5, -1.0, -1.5, -2.0, -3.0]


def load_data():
    areas = gpd.read_file(script_dir / "jer_areas.shp")
    areas = areas[areas['in_jeru'] == 1].copy()
    areas['pop'] = areas['pop_2025'].fillna(0)
    areas['emp'] = areas['emp_2025'].fillna(0)

    roads = gpd.read_file(script_dir / "jerusalem_roads.kml", driver='KML')
    completed = gpd.read_file(script_dir / "bike_lanes_completed.kml", driver='KML')
    construction = gpd.read_file(script_dir / "bike_lanes_construction.kml", driver='KML')
    wishing = gpd.read_file(script_dir / "bike_lanes_wishing_list.kml", driver='KML')

    return areas, roads, completed, construction, wishing


def build_network(roads_proj, bike_lanes_list, tolerance=NODE_TOLERANCE):
    from shapely.strtree import STRtree

    G = nx.Graph()
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]
    edge_to_geom = {}  # (s, e) -> road geometry for spatial matching

    def get_or_create_node(x, y):
        key = (round(x / tolerance) * tolerance, round(y / tolerance) * tolerance)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        return nid

    # First, add all road segments and keep track of geometries
    road_geoms = []
    road_edges = []
    for _, row in roads_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) >= 2:
            s = get_or_create_node(coords[0][0], coords[0][1])
            e = get_or_create_node(coords[-1][0], coords[-1][1])
            if s != e:
                G.add_edge(s, e, length=geom.length, has_bike_lane=False)
                edge_key = (min(s, e), max(s, e))
                edge_to_geom[edge_key] = geom
                road_geoms.append(geom)
                road_edges.append(edge_key)

    # Build spatial index for road geometries
    road_tree_spatial = STRtree(road_geoms) if road_geoms else None

    # Mark road edges that have bike lanes running along them
    BUFFER_DIST = 15  # meters - bike lane must be within 15m of road

    def mark_bike_lane_roads(line_geom):
        """Find and mark all road edges that this bike lane runs along."""
        if road_tree_spatial is None:
            return
        # Buffer the bike lane to find nearby roads
        buffered = line_geom.buffer(BUFFER_DIST)
        # Find candidate road geometries
        candidate_indices = road_tree_spatial.query(buffered)
        for idx in candidate_indices:
            road_geom = road_geoms[idx]
            # Check if the road segment significantly overlaps with the bike lane
            # Use intersection length as a measure
            try:
                intersection = road_geom.intersection(buffered)
                if intersection.is_empty:
                    continue
                # If most of the road segment is within the buffer, mark it
                overlap_ratio = intersection.length / road_geom.length if road_geom.length > 0 else 0
                if overlap_ratio > 0.5:  # At least 50% of road segment covered
                    edge_key = road_edges[idx]
                    s, e = edge_key
                    if G.has_edge(s, e):
                        G[s][e]['has_bike_lane'] = True
            except:
                pass

    for bl_gdf in bike_lanes_list:
        if bl_gdf is None or len(bl_gdf) == 0:
            continue
        bl_proj = bl_gdf.to_crs(TARGET_CRS)
        for _, row in bl_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            # Handle both LineString and MultiLineString
            if geom.geom_type == 'LineString':
                mark_bike_lane_roads(geom)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    mark_bike_lane_roads(line)

    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_array) if len(coords_array) > 0 else None
    return G, node_coords, node_tree, node_ids, edge_to_geom


def compute_area_accessibility(G, node_coords, node_tree, node_ids, areas_proj, theta, k):
    """Compute origin and destination accessibility for each area.
    origin: acc_orig[i] = sum_j E_j * tau_ij^theta  (how many jobs area i can reach)
    dest:   acc_dest[j] = sum_i P_i * tau_ij^theta  (how many people can reach area j)
    """
    Gw = G.copy()
    for u, v in Gw.edges():
        l = Gw[u][v]['length']
        Gw[u][v]['weight'] = l if Gw[u][v].get('has_bike_lane') else l * k

    centroids = areas_proj.geometry.centroid
    n = len(areas_proj)
    center_nodes = [node_ids[node_tree.query([c.x, c.y])[1]] for c in centroids]
    pop = areas_proj['pop'].values
    emp = areas_proj['emp'].values
    largest_cc = max(nx.connected_components(Gw), key=len)

    acc_orig = np.zeros(n)
    acc_dest = np.zeros(n)
    for i in range(n):
        if center_nodes[i] not in largest_cc:
            continue
        try:
            dists = nx.single_source_dijkstra_path_length(Gw, center_nodes[i], weight='weight')
        except Exception:
            continue
        for j in range(n):
            if i != j and center_nodes[j] in dists:
                tau = max(dists[center_nodes[j]] / 1000, 0.1)
                decay = tau ** theta
                acc_orig[i] += emp[j] * decay
                acc_dest[j] += pop[i] * decay
    return acc_orig, acc_dest


def geojson_from_gdf(gdf, props_list):
    """Convert GeoDataFrame to GeoJSON dict, only keeping specified properties."""
    gdf_wgs = gdf.to_crs(WGS84)
    features = []
    for idx, row in gdf_wgs.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = {}
        for p in props_list:
            val = row.get(p, None)
            if val is not None:
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = round(float(val), 2)
                props[p] = val
        features.append({
            "type": "Feature",
            "geometry": json.loads(gdf_wgs.loc[[idx]].geometry.to_json())["features"][0]["geometry"],
            "properties": props
        })
    return {"type": "FeatureCollection", "features": features}


def main():
    print("Loading data...")
    areas, roads, completed, construction, wishing = load_data()

    areas_proj = areas.to_crs(TARGET_CRS)
    roads_proj = roads.to_crs(TARGET_CRS)

    # Prepare area names and IDs
    areas['area_id'] = range(len(areas))
    areas_proj['area_id'] = range(len(areas_proj))
    area_names = []
    for _, row in areas.iterrows():
        name = row.get('name', None)
        if name is None or (isinstance(name, float) and np.isnan(name)):
            name = f"Area {row['area_id']}"
        area_names.append(str(name))

    # Build GeoJSON for display
    print("Building GeoJSON layers...")
    areas_geojson = geojson_from_gdf(areas[['geometry', 'pop', 'emp', 'area_id']], ['pop', 'emp', 'area_id'])
    # attach area_id to features properly
    for i, feat in enumerate(areas_geojson['features']):
        feat['properties']['area_id'] = i

    completed_geojson = geojson_from_gdf(completed[['geometry', 'Name']], ['Name']) if len(completed) > 0 else {"type": "FeatureCollection", "features": []}
    construction_geojson = geojson_from_gdf(construction[['geometry', 'Name']], ['Name']) if len(construction) > 0 else {"type": "FeatureCollection", "features": []}

    # Wishing list - use integer lane_id for identification
    wishing['lane_id'] = range(len(wishing))
    wishing_geojson = geojson_from_gdf(wishing[['geometry', 'Name', 'lane_id']], ['Name', 'lane_id'])
    lane_names = [wishing.iloc[i]['Name'] for i in range(len(wishing))]

    # Load sensitivity results
    print("Loading sensitivity data...")
    sens_df = pd.read_csv(script_dir / 'sensitivity_analysis.csv')

    # Build sensitivity dict indexed by lane_id (integer)
    # Map lane name -> lane_id
    name_to_id = {name: i for i, name in enumerate(lane_names)}

    sens_data = {}  # key: "K_theta" -> list of improvement_pct ordered by lane_id
    for K in K_VALUES:
        for theta in THETA_VALUES:
            key = f"{K}_{theta}"
            improvements = [0.0] * len(lane_names)
            sub = sens_df[(sens_df['K'] == K) & (sens_df['theta'] == theta)]
            for _, row in sub.iterrows():
                lid = name_to_id.get(row['lane'])
                if lid is not None:
                    improvements[lid] = round(row['improvement_pct'], 4)
            sens_data[key] = improvements

    # Compute accessibility
    print("Building network and computing accessibility...")
    G_base, nc, nt, ni, edge_geoms = build_network(roads_proj, [completed, construction])
    print(f"  Network: {G_base.number_of_nodes()} nodes, {G_base.number_of_edges()} edges")

    acc_orig_data = {}
    acc_dest_data = {}
    for K in K_VALUES:
        for theta in THETA_VALUES:
            key = f"{K}_{theta}"
            print(f"  Accessibility K={K}, theta={theta}...")
            acc_orig, acc_dest = compute_area_accessibility(G_base, nc, nt, ni, areas_proj, theta, K)
            acc_orig_data[key] = [round(float(v), 2) for v in acc_orig]
            acc_dest_data[key] = [round(float(v), 2) for v in acc_dest]

    # Build network data for path finding (WGS84 coords)
    print("Exporting network for path finding...")
    transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)
    nodes_wgs = {}
    for nid, (x, y) in nc.items():
        lon, lat = transformer.transform(x, y)
        nodes_wgs[str(nid)] = [round(lon, 6), round(lat, 6)]

    edges_list = []
    for u, v, d in G_base.edges(data=True):
        edges_list.append([u, v, round(d['length'], 1), 1 if d.get('has_bike_lane') else 0])

    # Export edge geometries for accurate path drawing
    edge_geoms_wgs = {}
    for (s, e), geom in edge_geoms.items():
        # Transform geometry coords to WGS84
        coords_wgs = []
        for x, y in geom.coords:
            lon, lat = transformer.transform(x, y)
            coords_wgs.append([round(lon, 6), round(lat, 6)])
        edge_key = f"{min(s,e)}_{max(s,e)}"
        edge_geoms_wgs[edge_key] = coords_wgs

    # Pre-compute edges for each wishing list lane (for online path calculation)
    # Find which road edges each wishing lane covers (same approach as build_network)
    print("Computing wishing lane edges for online path finding...")
    from shapely.strtree import STRtree

    wishing_proj = wishing.to_crs(TARGET_CRS)
    wishing_wgs = wishing.to_crs(WGS84)
    wishing_edges = {}  # lane_id -> [[nodeA, nodeB], ...] - road edges this lane covers
    wishing_geoms = {}  # lane_id -> [[lon, lat], ...] for visualization

    # Build road geometry index from roads_proj (need to rebuild for wishing lane matching)
    road_geoms_list = []
    road_edges_list = []
    for _, row in roads_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) >= 2:
            # Find the nodes for this road segment
            sx, sy = coords[0][0], coords[0][1]
            ex, ey = coords[-1][0], coords[-1][1]
            _, s_idx = nt.query([sx, sy])
            _, e_idx = nt.query([ex, ey])
            s, e = ni[s_idx], ni[e_idx]
            if s != e:
                edge_key = (min(s, e), max(s, e))
                road_geoms_list.append(geom)
                road_edges_list.append(edge_key)

    road_tree_spatial = STRtree(road_geoms_list) if road_geoms_list else None
    BUFFER_DIST = 15  # meters

    def get_linestrings(geom):
        """Extract LineStrings from any geometry type."""
        if geom.geom_type == 'LineString':
            return [geom]
        elif geom.geom_type == 'MultiLineString':
            return list(geom.geoms)
        return []

    for lid in range(len(wishing_proj)):
        geom = wishing_proj.iloc[lid].geometry
        geom_wgs = wishing_wgs.iloc[lid].geometry
        if geom is None or geom.is_empty:
            wishing_edges[lid] = []
            wishing_geoms[lid] = []
            continue

        # Store WGS84 coordinates for visualization
        lines_wgs = get_linestrings(geom_wgs)
        all_coords = []
        for line in lines_wgs:
            all_coords.extend([[round(c[0], 6), round(c[1], 6)] for c in line.coords])
        wishing_geoms[lid] = all_coords if all_coords else []

        # Find road edges this wishing lane covers
        covered_edges = set()
        for line in get_linestrings(geom):
            if road_tree_spatial is None:
                continue
            buffered = line.buffer(BUFFER_DIST)
            candidate_indices = road_tree_spatial.query(buffered)
            for idx in candidate_indices:
                road_geom = road_geoms_list[idx]
                try:
                    intersection = road_geom.intersection(buffered)
                    if intersection.is_empty:
                        continue
                    overlap_ratio = intersection.length / road_geom.length if road_geom.length > 0 else 0
                    if overlap_ratio > 0.5:
                        covered_edges.add(road_edges_list[idx])
                except:
                    pass

        # Store as list of [nodeA, nodeB] pairs
        wishing_edges[lid] = [[e[0], e[1]] for e in covered_edges]

    # Area centroids in WGS84
    areas_wgs = areas.to_crs(WGS84)
    centroids_wgs = [[round(c.x, 6), round(c.y, 6)] for c in areas_wgs.geometry.centroid]

    # Generate HTML
    print("Generating HTML...")
    # Extract area data for online computation
    area_pop = [round(float(v), 0) for v in areas_proj['pop'].values]
    area_emp = [round(float(v), 0) for v in areas_proj['emp'].values]

    # Pre-compute area center nodes (which network node is closest to each area centroid)
    centroids_proj = [[round(c.x, 1), round(c.y, 1)] for c in areas_proj.geometry.centroid]
    area_center_nodes = []
    for c in areas_proj.geometry.centroid:
        _, idx = nt.query([c.x, c.y])
        area_center_nodes.append(ni[idx])

    html = generate_html(
        areas_geojson=areas_geojson,
        completed_geojson=completed_geojson,
        construction_geojson=construction_geojson,
        wishing_geojson=wishing_geojson,
        lane_names=lane_names,
        area_names=area_names,
        sens_data=sens_data,
        acc_orig_data=acc_orig_data,
        acc_dest_data=acc_dest_data,
        area_pop=area_pop,
        area_emp=area_emp,
        area_center_nodes=area_center_nodes,
        nodes_wgs=nodes_wgs,
        edges_list=edges_list,
        edge_geoms_wgs=edge_geoms_wgs,
        wishing_edges=wishing_edges,
        wishing_geoms=wishing_geoms,
        centroids_wgs=centroids_wgs,
    )

    out_path = script_dir / 'bike_analysis.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Done! Open {out_path}")


def generate_html(*, areas_geojson, completed_geojson, construction_geojson,
                  wishing_geojson, lane_names, area_names, sens_data,
                  acc_orig_data, acc_dest_data, area_pop, area_emp, area_center_nodes,
                  nodes_wgs, edges_list, edge_geoms_wgs, wishing_edges, wishing_geoms, centroids_wgs):

    # Serialize data compactly
    def js_json(obj):
        return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jerusalem Bike Lane Analysis</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{{box-sizing:border-box}}
body{{font-family:Arial,sans-serif;margin:0;display:flex;flex-direction:column;height:100vh}}
.header{{background:#2c3e50;color:#fff;padding:10px 20px;display:flex;justify-content:space-between;align-items:center}}
.header h1{{margin:0;font-size:1.4em}}
.controls{{background:#34495e;padding:10px 20px;display:flex;gap:20px;flex-wrap:wrap;align-items:center}}
.cg{{display:flex;align-items:center;gap:8px}}
.cg label{{color:#fff;font-weight:700;font-size:.9em}}
select,button{{padding:7px 12px;border:none;border-radius:4px;font-size:13px}}
button{{background:#3498db;color:#fff;cursor:pointer}}
button:hover{{background:#2980b9}}
.radio-group{{display:flex;gap:12px;align-items:center}}
.radio-group label{{color:#fff;font-weight:400;cursor:pointer;display:flex;align-items:center;gap:4px}}
.radio-group input{{cursor:pointer}}
.main{{display:flex;flex:1;overflow:hidden}}
.map-wrap{{flex:1;position:relative}}
#map{{width:100%;height:100%}}
.sidebar{{width:380px;background:#ecf0f1;overflow-y:auto;padding:12px;font-size:.9em}}
.sidebar h3{{margin:0 0 8px;color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:4px}}
.tabs{{display:flex;gap:6px;margin-bottom:10px}}
.tabs button{{flex:1;padding:8px;text-align:center}}
.tabs button.act{{background:#27ae60}}
.tc{{display:none}}.tc.act{{display:block}}
.lane{{padding:6px 8px;margin:4px 0;background:#fff;border-radius:4px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;border:2px solid transparent}}
.lane:hover{{background:#d5dbdb}}
.lane.sel{{background:#a9dfbf;border-color:#27ae60}}
.lane .pct{{font-weight:700;color:#27ae60;white-space:nowrap;margin-left:8px}}
.legend{{position:absolute;bottom:30px;right:10px;background:#fff;padding:10px;border-radius:5px;box-shadow:0 2px 5px rgba(0,0,0,.3);z-index:1000;font-size:.85em}}
.legend-item{{display:flex;align-items:center;gap:8px;margin:4px 0}}
.legend-line{{width:24px;height:4px;border-radius:2px}}
.info{{position:absolute;top:10px;left:10px;background:#fff;padding:10px 14px;border-radius:5px;box-shadow:0 2px 5px rgba(0,0,0,.3);z-index:1000;max-width:300px;display:none}}
.path-ctl{{background:#f8f9fa;padding:10px;border-radius:5px;margin-bottom:12px}}
.path-ctl select{{width:100%;margin:4px 0}}
.path-ctl button{{width:100%;margin-top:8px}}
.note{{background:#fff3cd;padding:8px;border-radius:4px;margin-bottom:10px;font-size:.85em}}
#totalImp{{font-size:.95em}}
.formula{{background:#1a252f;color:#ccc;padding:6px 20px;font-size:.85em;display:flex;align-items:center;gap:15px}}
.formula .math{{color:#fff;font-family:'Times New Roman',serif;font-size:1.1em}}
.formula .math .var{{color:#f1c40f}}
.path-stats{{background:#fff;border-radius:5px;padding:10px;margin-top:10px}}
.path-stats .bar{{height:20px;border-radius:3px;display:flex;overflow:hidden;margin:6px 0}}
.path-stats .bar-lane{{background:#1565C0}}
.path-stats .bar-road{{background:#E65100}}
.path-stats table{{width:100%;border-collapse:collapse;font-size:.9em}}
.path-stats td{{padding:3px 6px}}
.path-stats td:last-child{{text-align:right;font-weight:700}}
</style>
</head>
<body>
<div class="header">
  <h1>Jerusalem Bike Lane Analysis</h1>
  <span id="totalImp">Estimated total improvement: 0%</span>
</div>
<div class="formula">
  <span>Accessibility model:</span>
  <span class="math">
    N = &Sigma;<sub>i</sub> &Sigma;<sub>j</sub> P<sub>i</sub> &middot; E<sub>j</sub> &middot; &tau;<sub>ij</sub><sup class="var">&theta;</sup>
    &nbsp;&nbsp;where&nbsp;
    &tau;<sub>ij</sub> = shortest path with weight
    <span class="var">K</span>&middot;d for roads,&nbsp; 1&middot;d for bike lanes
  </span>
</div>
<div class="controls">
  <div class="cg">
    <label>K (no-lane penalty):</label>
    <select id="kSel">
      <option value="10">10</option>
      <option value="50">50</option>
      <option value="100" selected>100</option>
      <option value="200">200</option>
      <option value="500">500</option>
    </select>
  </div>
  <div class="cg">
    <label>&theta; (distance decay):</label>
    <select id="tSel">
      <option value="-0.5">-0.5</option>
      <option value="-1.0" selected>-1.0</option>
      <option value="-1.5">-1.5</option>
      <option value="-2.0">-2.0</option>
      <option value="-3.0">-3.0</option>
    </select>
  </div>
  <div class="cg">
    <label>Color areas by:</label>
    <div class="radio-group">
      <label><input type="radio" name="accMode" value="origin" checked onchange="refresh()"> Origin (jobs reachable)</label>
      <label><input type="radio" name="accMode" value="dest" onchange="refresh()"> Destination (people reaching)</label>
    </div>
  </div>
  <div class="cg">
    <button onclick="clearSel()">Clear selection</button>
  </div>
</div>
<div class="main">
  <div class="map-wrap">
    <div id="map"></div>
    <div class="legend">
      <strong>Legend - Lanes</strong>
      <div class="legend-item"><div class="legend-line" style="background:#1B5E20"></div>Existing lanes</div>
      <div class="legend-item"><div class="legend-line" style="background:#81C784"></div>Under construction</div>
      <div class="legend-item"><div class="legend-line" style="background:#FF9800"></div>Wishing list</div>
      <div class="legend-item"><div class="legend-line" style="background:#9b59b6;height:6px"></div>Selected lane</div>
      <div class="legend-item"><div class="legend-line" style="background:#1565C0;height:6px"></div>Path on bike lane</div>
      <div class="legend-item"><div class="legend-line" style="background:#E65100;height:6px"></div>Path on road</div>
      <hr style="margin:8px 0;border:none;border-top:1px solid #ccc">
      <strong>Area Accessibility</strong>
      <div class="legend-item" style="flex-direction:column;align-items:flex-start;gap:2px">
        <div style="display:flex;align-items:center;gap:4px">
          <div style="width:80px;height:12px;background:linear-gradient(to right,rgb(255,0,100),rgb(128,128,100),rgb(0,255,100));border-radius:2px"></div>
        </div>
        <div style="display:flex;justify-content:space-between;width:80px;font-size:0.75em">
          <span>Low</span><span>High</span>
        </div>
      </div>
      <div id="deltaLegend" style="display:none;margin-top:4px">
        <strong>Change (<span id="deltaSelCount">1</span> lane selected)</strong>
        <div class="legend-item" style="flex-direction:column;align-items:flex-start;gap:2px">
          <div style="display:flex;align-items:center;gap:4px">
            <div style="width:80px;height:12px;background:linear-gradient(to right,rgb(240,240,240),rgb(100,100,255));border-radius:2px"></div>
          </div>
          <div style="display:flex;justify-content:space-between;width:80px;font-size:0.75em">
            <span>0</span><span id="deltaMax">+</span>
          </div>
        </div>
      </div>
    </div>
    <div class="info" id="info">
      <strong id="infoTitle"></strong>
      <div id="infoBody"></div>
    </div>
  </div>
  <div class="sidebar">
    <div class="tabs">
      <button class="act" onclick="showTab('lanes',this)">Select Lanes</button>
      <button onclick="showTab('paths',this)">Find Path</button>
      <button onclick="showTab('compute',this)">Accessibility</button>
    </div>
    <div id="lanes" class="tc act">
      <h3>Select Wishing Lanes</h3>
      <div class="note">Click lanes to select them for the network. Selected lanes affect path finding and accessibility calculations.</div>
      <input type="text" id="laneSearch" placeholder="Search lanes..." style="width:100%;padding:8px;margin:8px 0;border:1px solid #ddd;border-radius:4px;box-sizing:border-box" oninput="filterLanes()">
      <div id="laneList"></div>
    </div>
    <div id="compute" class="tc">
      <h3>Compute Accessibility</h3>
      <div class="note">Calculate accessibility for current network with selected lanes. This computes all area-to-area distances (30-60 sec).</div>
      <div class="path-ctl">
        <p><b>Selected lanes:</b> <span id="selCount">0</span></p>
        <p><b>Parameters:</b> K=<span id="compK">100</span>, &theta;=<span id="compT">-1.0</span></p>
        <button onclick="computeAccessibility()" id="computeBtn">Compute Accessibility</button>
        <div id="computeProgress" style="margin-top:10px"></div>
      </div>
      <div id="computeResults" style="margin-top:10px"></div>
    </div>
    <div id="paths" class="tc">
      <h3>Find Shortest Path</h3>
      <div class="note">Compute shortest path between areas using current network (with selected lanes).</div>
      <div class="path-ctl">
        <label>Origin area:</label>
        <select id="origSel"><option value="">-- select --</option></select>
        <label>Destination area:</label>
        <select id="destSel"><option value="">-- select --</option></select>
        <button onclick="showPath()">Show path</button>
      </div>
      <div id="pathInfo"></div>
    </div>
  </div>
</div>

<script>
// === EMBEDDED DATA ===
const AREAS={js_json(areas_geojson)};
const COMPLETED={js_json(completed_geojson)};
const CONSTRUCTION={js_json(construction_geojson)};
const WISHING={js_json(wishing_geojson)};
const LANE_NAMES={js_json(lane_names)};
const AREA_NAMES={js_json(area_names)};
const SENS={js_json(sens_data)};
const ACC_ORIG={js_json(acc_orig_data)};
const ACC_DEST={js_json(acc_dest_data)};
const AREA_POP={js_json(area_pop)};
const AREA_EMP={js_json(area_emp)};
const AREA_NODES={js_json(area_center_nodes)};
const NODES={js_json(nodes_wgs)};
const EDGES={js_json(edges_list)};
const EDGE_GEOMS={js_json(edge_geoms_wgs)};
const WISHING_EDGES={js_json(wishing_edges)};
const WISHING_GEOMS={js_json(wishing_geoms)};
const CENTROIDS={js_json(centroids_wgs)};

// === STATE ===
const sel=new Set();
let wishLyr,areasLyr,pathLyrGroup;

// === MAP INIT ===
const map=L.map("map").setView([31.78,35.22],12);
L.tileLayer("https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png",{{
  attribution:'&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  subdomains:'abcd',
  maxZoom:20
}}).addTo(map);

// Areas
areasLyr=L.geoJSON(AREAS,{{
  style:()=>({{fillColor:"#3498db",weight:1,opacity:.5,color:"#2c3e50",fillOpacity:.1}}),
  onEachFeature:(f,layer)=>{{
    const p=f.properties;
    const name=AREA_NAMES[p.area_id]||"Area "+p.area_id;
    layer.bindPopup("<b>"+name+"</b><br>Pop: "+Math.round(p.pop).toLocaleString()+"<br>Emp: "+Math.round(p.emp).toLocaleString());
  }}
}}).addTo(map);

// Completed (dark green)
if(COMPLETED.features.length)
  L.geoJSON(COMPLETED,{{style:{{color:"#1B5E20",weight:3,opacity:.8}},
    onEachFeature:(f,l)=>l.bindPopup("<b>Existing:</b> "+(f.properties.Name||""))
  }}).addTo(map);

// Construction (light green)
if(CONSTRUCTION.features.length)
  L.geoJSON(CONSTRUCTION,{{style:{{color:"#81C784",weight:3,opacity:.8}},
    onEachFeature:(f,l)=>l.bindPopup("<b>Under construction:</b> "+(f.properties.Name||""))
  }}).addTo(map);

// Wishing list (orange, purple when selected)
wishLyr=L.geoJSON(WISHING,{{
  style:f=>{{
    const s=sel.has(f.properties.lane_id);
    return {{color:s?"#9b59b6":"#FF9800",weight:s?5:3,opacity:.8}};
  }},
  onEachFeature:(f,layer)=>{{
    const lid=f.properties.lane_id;
    layer.on("click",()=>{{sel.has(lid)?sel.delete(lid):sel.add(lid);refresh();}});
    layer.on("mouseover",()=>showInfo(lid));
    layer.on("mouseout",hideInfo);
  }}
}}).addTo(map);

// Populate area selects (sorted alphabetically)
(function(){{
  const items=AREA_NAMES.map((n,i)=>({{id:i,name:n}}));
  items.sort((a,b)=>a.name.localeCompare(b.name,'he'));
  const opts=items.map(it=>'<option value="'+it.id+'">'+it.name+'</option>').join("");
  document.getElementById("origSel").innerHTML='<option value="">-- select origin --</option>'+opts;
  document.getElementById("destSel").innerHTML='<option value="">-- select dest --</option>'+opts;
}})();

// === KEY HELPER ===
function paramKey(){{
  return document.getElementById("kSel").value+"_"+document.getElementById("tSel").value;
}}

// === REFRESH ===
function refresh(){{
  // Update wishing layer style
  wishLyr.setStyle(f=>{{
    const s=sel.has(f.properties.lane_id);
    return {{color:s?"#9b59b6":"#FF9800",weight:s?5:3,opacity:.8}};
  }});
  // Update lane list
  buildLaneList();
  // Update total improvement
  const imps=SENS[paramKey()]||[];
  let tot=0;
  sel.forEach(id=>{{tot+=imps[id]||0;}});
  document.getElementById("totalImp").textContent="Estimated total improvement: +"+tot.toFixed(2)+"%";
  // Update area colors
  updateAreaColors();
  // Update compute panel
  updateComputePanel();
}}

function buildLaneList(){{
  const imps=SENS[paramKey()]||[];
  const search=(document.getElementById("laneSearch").value||"").toLowerCase();
  // Build alphabetically sorted list
  const items=LANE_NAMES.map((name,i)=>({{id:i,name:name,imp:imps[i]||0}}));
  items.sort((a,b)=>a.name.localeCompare(b.name,'he'));
  // Filter by search
  const filtered=search?items.filter(it=>it.name.toLowerCase().includes(search)):items;
  const container=document.getElementById("laneList");
  container.innerHTML=filtered.map(it=>{{
    const cls=sel.has(it.id)?"lane sel":"lane";
    return '<div class="'+cls+'" onclick="toggleLane('+it.id+')"><span>'+it.name+'</span><span class="pct">+'+it.imp.toFixed(2)+'%</span></div>';
  }}).join("");
}}

function filterLanes(){{
  buildLaneList();
}}

function toggleLane(id){{
  sel.has(id)?sel.delete(id):sel.add(id);
  refresh();
}}

function clearSel(){{sel.clear();refresh();updateComputePanel();}}

function getAccMode(){{
  const radio=document.querySelector('input[name="accMode"]:checked');
  return radio?radio.value:"origin";
}}

// Store computed accessibility results
let computedAcc = null;
let computedKey = null;
let computedSel = null;

function updateAreaColors(){{
  const mode=getAccMode();
  const key=paramKey();
  const deltaLegend=document.getElementById("deltaLegend");

  // If we have computed results for current selection, show delta
  if(computedAcc && computedKey===key && computedSel===JSON.stringify([...sel].sort())){{
    const base=(mode==="dest")?ACC_DEST[key]:ACC_ORIG[key];
    const comp=(mode==="dest")?computedAcc.dest:computedAcc.orig;
    if(base&&comp){{
      // Calculate percentage change for each area
      const deltaPct=comp.map((v,i)=>base[i]>0?100*(v-base[i])/base[i]:0);
      const maxPct=Math.max(...deltaPct,0.01);
      // Update legend with percentage
      const deltaMaxEl=document.getElementById("deltaMax");
      const deltaSelEl=document.getElementById("deltaSelCount");
      if(deltaMaxEl)deltaMaxEl.textContent="+"+maxPct.toFixed(1)+"%";
      if(deltaSelEl)deltaSelEl.textContent=sel.size;
      areasLyr.eachLayer(layer=>{{
        const aid=layer.feature.properties.area_id;
        const v=Math.max(0,deltaPct[aid]||0);
        const n=v/maxPct;
        // Color from gray (0) to blue (max positive)
        const r=Math.round(240*(1-n)+100*n);
        const g=Math.round(240*(1-n)+100*n);
        const b=Math.round(240*(1-n)+255*n);
        layer.setStyle({{fillColor:"rgb("+r+","+g+","+b+")",fillOpacity:.5,weight:1,opacity:.5,color:"#2c3e50"}});
      }});
      if(deltaLegend)deltaLegend.style.display="block";
      return;
    }}
  }}

  // Show baseline accessibility coloring
  if(deltaLegend)deltaLegend.style.display="none";
  const acc=(mode==="dest")?ACC_DEST[key]:ACC_ORIG[key];
  if(!acc||!acc.length)return;
  const pos=acc.filter(a=>a>0);
  if(!pos.length)return;
  const mn=Math.min(...pos),mx=Math.max(...pos);
  areasLyr.eachLayer(layer=>{{
    const aid=layer.feature.properties.area_id;
    const v=acc[aid]||0;
    const n=mx>mn?(v-mn)/(mx-mn):0;
    const r=Math.round(255*(1-n)),g=Math.round(255*n);
    layer.setStyle({{fillColor:"rgb("+r+","+g+",100)",fillOpacity:.4,weight:1,opacity:.5,color:"#2c3e50"}});
  }});
}}

function showInfo(lid){{
  const imps=SENS[paramKey()]||[];
  const panel=document.getElementById("info");
  document.getElementById("infoTitle").textContent=LANE_NAMES[lid];
  document.getElementById("infoBody").textContent="Improvement: +"+((imps[lid]||0).toFixed(3))+"%";
  panel.style.display="block";
}}
function hideInfo(){{document.getElementById("info").style.display="none";}}

// === TABS ===
function showTab(id,btn){{
  document.querySelectorAll(".tc").forEach(e=>e.classList.remove("act"));
  document.querySelectorAll(".tabs button").forEach(b=>b.classList.remove("act"));
  document.getElementById(id).classList.add("act");
  btn.classList.add("act");
}}

// === PATH FINDING ===
function showPath(){{
  const oi=document.getElementById("origSel").value;
  const di=document.getElementById("destSel").value;
  if(oi===""||di===""){{alert("Please select origin and destination areas.");return;}}
  if(pathLyrGroup)map.removeLayer(pathLyrGroup);

  const k=parseInt(document.getElementById("kSel").value);
  document.getElementById("pathInfo").innerHTML="<p>Computing path...</p>";

  setTimeout(()=>{{
    try{{
    const result=dijkstra(parseInt(oi),parseInt(di),k);
    if(result&&result.segments.length>0){{
      // Draw segments with different colors
      pathLyrGroup=L.featureGroup();
      let totalLaneDist=0,totalRoadDist=0;
      for(const seg of result.segments){{
        let coords;
        if(seg.geom){{
          // Wishing lane with full geometry
          coords=seg.geom.map(c=>[c[1],c[0]]);
        }}else{{
          // Regular segment (straight line between nodes)
          coords=[[seg.from[1],seg.from[0]],[seg.to[1],seg.to[0]]];
        }}
        const color=seg.bike?"#1565C0":"#E65100";
        const line=L.polyline(coords,{{color:color,weight:6,opacity:.85}});
        pathLyrGroup.addLayer(line);
        if(seg.bike)totalLaneDist+=seg.len;
        else totalRoadDist+=seg.len;
      }}
      pathLyrGroup.addTo(map);
      if(pathLyrGroup.getLayers().length>0){{
        map.fitBounds(pathLyrGroup.getBounds(),{{padding:[30,30]}});
      }}

      // Report stats
      const totalDist=totalLaneDist+totalRoadDist;
      const lanePct=totalDist>0?(100*totalLaneDist/totalDist):0;
      const roadPct=totalDist>0?(100*totalRoadDist/totalDist):0;
      document.getElementById("pathInfo").innerHTML=
        '<div class="path-stats">'+
        '<p><b>Path found</b></p>'+
        '<table>'+
        '<tr><td>Total distance:</td><td>'+(totalDist/1000).toFixed(2)+' km</td></tr>'+
        '<tr><td style="color:#1565C0">On bike lane:</td><td style="color:#1565C0">'+(totalLaneDist/1000).toFixed(2)+' km ('+lanePct.toFixed(1)+'%)</td></tr>'+
        '<tr><td style="color:#E65100">On road:</td><td style="color:#E65100">'+(totalRoadDist/1000).toFixed(2)+' km ('+roadPct.toFixed(1)+'%)</td></tr>'+
        '<tr><td>Segments:</td><td>'+result.segments.length+'</td></tr>'+
        '</table>'+
        '<div class="bar">'+
        '<div class="bar-lane" style="width:'+lanePct+'%"></div>'+
        '<div class="bar-road" style="width:'+roadPct+'%"></div>'+
        '</div>'+
        '</div>';
    }}else{{
      document.getElementById("pathInfo").innerHTML='<p style="color:#e74c3c">No path found between these areas.</p>';
    }}
    }}catch(err){{
      document.getElementById("pathInfo").innerHTML='<p style="color:#e74c3c">Error computing path: '+err.message+'</p>';
      console.error(err);
    }}
  }},50);
}}

// Build edge lookup: "nodeA_nodeB" -> {{len, bike}}
let edgeLookup=null;
function getEdgeLookup(){{
  if(edgeLookup)return edgeLookup;
  edgeLookup={{}};
  for(const e of EDGES){{
    const a=String(e[0]),b=String(e[1]);
    const info={{len:e[2],bike:!!e[3]}};
    edgeLookup[a+"_"+b]=info;
    edgeLookup[b+"_"+a]=info;
  }}
  return edgeLookup;
}}

function dijkstra(origIdx,destIdx,k){{
  const oc=CENTROIDS[origIdx],dc=CENTROIDS[destIdx];
  let oNode=null,dNode=null,oD=Infinity,dD=Infinity;
  for(const[nid,c]of Object.entries(NODES)){{
    const d1=(c[0]-oc[0])**2+(c[1]-oc[1])**2;
    const d2=(c[0]-dc[0])**2+(c[1]-dc[1])**2;
    if(d1<oD){{oD=d1;oNode=nid;}}
    if(d2<dD){{dD=d2;dNode=nid;}}
  }}
  if(!oNode||!dNode)return null;

  // Build set of edges covered by selected wishing lanes
  const wishingEdgeSet=new Set();
  for(const lid of sel){{
    const edges=WISHING_EDGES[lid]||[];
    for(const e of edges){{
      const a=Math.min(e[0],e[1]),b=Math.max(e[0],e[1]);
      wishingEdgeSet.add(a+'_'+b);
    }}
  }}

  // Build adjacency with edge info - mark edges covered by wishing lanes as bike lanes
  const adj={{}};
  for(const e of EDGES){{
    const len=e[2];
    let bike=!!e[3];
    const a=String(e[0]),b=String(e[1]);
    const edgeKey=Math.min(e[0],e[1])+'_'+Math.max(e[0],e[1]);
    // If this edge is covered by a selected wishing lane, treat as bike lane
    if(wishingEdgeSet.has(edgeKey))bike=true;
    const w=bike?len:len*k;
    if(!adj[a])adj[a]=[];
    if(!adj[b])adj[b]=[];
    adj[a].push({{n:b,w:w,len:len,bike:bike,eKey:edgeKey}});
    adj[b].push({{n:a,w:w,len:len,bike:bike,eKey:edgeKey}});
  }}

  const dist={{}},prev={{}},prevEdge={{}},visited=new Set();
  dist[oNode]=0;
  let pq=[[0,oNode]];

  while(pq.length){{
    const[cd,cur]=pq.shift();
    if(visited.has(cur))continue;
    visited.add(cur);
    if(cur===dNode)break;
    for(const{{n,w,len,bike,eKey}}of(adj[cur]||[])){{
      if(visited.has(n))continue;
      const nd=cd+w;
      if(dist[n]===undefined||nd<dist[n]){{
        dist[n]=nd;prev[n]=cur;prevEdge[n]={{len:len,bike:bike,eKey:eKey}};
        let ins=pq.findIndex(x=>x[0]>nd);
        if(ins<0)ins=pq.length;
        pq.splice(ins,0,[nd,n]);
      }}
    }}
  }}

  if(dist[dNode]===undefined)return null;

  // Reconstruct path as segments with edge info
  const segments=[];
  let c=dNode;
  while(prev[c]!==undefined){{
    const p=prev[c];
    const e=prevEdge[c];
    // Use edge geometry if available, otherwise fall back to node coords
    const geom=EDGE_GEOMS[e.eKey];
    if(geom&&geom.length>=2){{
      // Check if we need to reverse the geometry (based on direction of travel)
      const fromNode=NODES[p],toNode=NODES[c];
      const g0=geom[0],gN=geom[geom.length-1];
      const d0=Math.abs(g0[0]-fromNode[0])+Math.abs(g0[1]-fromNode[1]);
      const dN=Math.abs(gN[0]-fromNode[0])+Math.abs(gN[1]-fromNode[1]);
      const coords=(d0<=dN)?geom:geom.slice().reverse();
      segments.unshift({{geom:coords,len:e.len,bike:e.bike}});
    }}else{{
      segments.unshift({{from:NODES[p],to:NODES[c],len:e.len,bike:e.bike}});
    }}
    c=p;
  }}
  return {{segments:segments}};
}}

// === ONLINE COMPUTATION ===
function updateComputePanel(){{
  document.getElementById("selCount").textContent=sel.size;
  document.getElementById("compK").textContent=document.getElementById("kSel").value;
  document.getElementById("compT").textContent=document.getElementById("tSel").value;
}}

function computeAccessibility(){{
  const k=parseInt(document.getElementById("kSel").value);
  const theta=parseFloat(document.getElementById("tSel").value);
  const key=paramKey();
  const selArr=[...sel].sort();
  const selKey=JSON.stringify(selArr);

  const btn=document.getElementById("computeBtn");
  const prog=document.getElementById("computeProgress");
  const results=document.getElementById("computeResults");

  btn.disabled=true;
  btn.textContent="Computing...";
  prog.innerHTML="<p>Building network with "+sel.size+" selected lanes...</p>";

  setTimeout(()=>{{
    // Build adjacency list with selected lanes
    const adj={{}};
    for(const e of EDGES){{
      const len=e[2],bike=!!e[3];
      const w=bike?len:len*k;
      const a=String(e[0]),b=String(e[1]);
      if(!adj[a])adj[a]=[];
      if(!adj[b])adj[b]=[];
      adj[a].push({{n:b,w:w,len:len}});
      adj[b].push({{n:a,w:w,len:len}});
    }}
    // Build edge length lookup
    const edgeLenLookup={{}};
    for(const e of EDGES){{
      const key=Math.min(e[0],e[1])+'_'+Math.max(e[0],e[1]);
      edgeLenLookup[key]=e[2];
    }}
    // Mark edges covered by selected wishing lanes as bike lanes
    for(const lid of sel){{
      const edges=WISHING_EDGES[lid]||[];
      for(const e of edges){{
        const a=String(e[0]),b=String(e[1]);
        const key=Math.min(e[0],e[1])+'_'+Math.max(e[0],e[1]);
        const len=edgeLenLookup[key]||0;
        // Update adjacency - these edges should have no K penalty
        if(adj[a])adj[a]=adj[a].map(x=>x.n===b?{{...x,w:len}}:x);
        if(adj[b])adj[b]=adj[b].map(x=>x.n===a?{{...x,w:len}}:x);
      }}
    }}

    const n=AREA_NODES.length;
    const acc_orig=new Array(n).fill(0);
    const acc_dest=new Array(n).fill(0);
    let totalN=0;
    let processed=0;

    function processArea(i){{
      if(i>=n){{
        // Done - show results
        computedAcc={{orig:acc_orig,dest:acc_dest,totalN:totalN}};
        computedKey=key;
        computedSel=selKey;

        // Compute baseline for comparison
        const baseOrig=ACC_ORIG[key]||[];
        const baseDest=ACC_DEST[key]||[];
        let baseN=0;
        for(let ii=0;ii<n;ii++){{
          for(let jj=0;jj<n;jj++){{
            if(ii!==jj)baseN+=AREA_POP[ii]*AREA_EMP[jj];
          }}
        }}
        // Use actual baseline N from pre-computed (approximate)
        const baseOrigSum=baseOrig.reduce((a,b)=>a+b,0);
        const newOrigSum=acc_orig.reduce((a,b)=>a+b,0);
        const improvement=baseOrigSum>0?100*(newOrigSum-baseOrigSum)/baseOrigSum:0;

        btn.disabled=false;
        btn.textContent="Compute Accessibility";
        prog.innerHTML="<p style='color:#27ae60'>Computation complete!</p>";
        results.innerHTML=
          '<div class="path-stats">'+
          '<p><b>Results for selected network:</b></p>'+
          '<table>'+
          '<tr><td>Total N (gravity sum):</td><td>'+totalN.toExponential(3)+'</td></tr>'+
          '<tr><td>Improvement vs baseline:</td><td style="color:'+(improvement>=0?"#27ae60":"#e74c3c")+'">'+
            (improvement>=0?"+":"")+improvement.toFixed(3)+'%</td></tr>'+
          '</table>'+
          '<p style="font-size:.85em;margin-top:8px">Area colors now show change from baseline.</p>'+
          '</div>';
        updateAreaColors();
        return;
      }}

      const src=String(AREA_NODES[i]);
      // Run Dijkstra from area i
      const dist={{}},visited=new Set();
      dist[src]=0;
      let pq=[[0,src]];
      while(pq.length){{
        const[cd,cur]=pq.shift();
        if(visited.has(cur))continue;
        visited.add(cur);
        for(const{{n:nb,w}}of(adj[cur]||[])){{
          if(visited.has(nb))continue;
          const nd=cd+w;
          if(dist[nb]===undefined||nd<dist[nb]){{
            dist[nb]=nd;
            let ins=pq.findIndex(x=>x[0]>nd);
            if(ins<0)ins=pq.length;
            pq.splice(ins,0,[nd,nb]);
          }}
        }}
      }}

      // Accumulate accessibility
      for(let j=0;j<n;j++){{
        if(i===j)continue;
        const dstNode=String(AREA_NODES[j]);
        if(dist[dstNode]!==undefined){{
          const tau=Math.max(dist[dstNode]/1000,0.1);
          const decay=Math.pow(tau,theta);
          acc_orig[i]+=AREA_EMP[j]*decay;
          acc_dest[j]+=AREA_POP[i]*decay;
          totalN+=AREA_POP[i]*AREA_EMP[j]*decay;
        }}
      }}

      processed++;
      if(processed%10===0){{
        prog.innerHTML="<p>Processing areas: "+processed+"/"+n+" ("+Math.round(100*processed/n)+"%)</p>";
      }}
      setTimeout(()=>processArea(i+1),0);
    }}

    processArea(0);
  }},50);
}}

// === EVENT LISTENERS ===
document.getElementById("kSel").onchange=()=>{{refresh();updateComputePanel();}};
document.getElementById("tSel").onchange=()=>{{refresh();updateComputePanel();}};

// Initial render
buildLaneList();
updateAreaColors();
updateComputePanel();
</script>
</body>
</html>'''


if __name__ == '__main__':
    main()
