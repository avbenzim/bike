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
        return nid

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
                s = get_or_create_node(coords[0][0], coords[0][1])
                e = get_or_create_node(coords[-1][0], coords[-1][1])
                if s != e:
                    if G.has_edge(s, e):
                        G[s][e]['has_bike_lane'] = True
                    else:
                        G.add_edge(s, e, length=geom.length, has_bike_lane=True)

    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_array) if len(coords_array) > 0 else None
    return G, node_coords, node_tree, node_ids


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
    G_base, nc, nt, ni = build_network(roads_proj, [completed, construction])
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

    # Pre-compute edges for each wishing list lane (for online path calculation)
    print("Computing wishing lane edges for online path finding...")
    wishing_proj = wishing.to_crs(TARGET_CRS)
    wishing_edges = {}  # lane_id -> [[nodeA, nodeB, length], ...]

    # Build a coord_to_node lookup from the base network
    coord_to_node = {}
    for nid, (x, y) in nc.items():
        key = (round(x / NODE_TOLERANCE) * NODE_TOLERANCE, round(y / NODE_TOLERANCE) * NODE_TOLERANCE)
        coord_to_node[key] = nid

    for lid in range(len(wishing_proj)):
        geom = wishing_proj.iloc[lid].geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            wishing_edges[lid] = []
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            wishing_edges[lid] = []
            continue

        # Get or find nearest node for start and end
        sx, sy = coords[0][0], coords[0][1]
        ex, ey = coords[-1][0], coords[-1][1]

        skey = (round(sx / NODE_TOLERANCE) * NODE_TOLERANCE, round(sy / NODE_TOLERANCE) * NODE_TOLERANCE)
        ekey = (round(ex / NODE_TOLERANCE) * NODE_TOLERANCE, round(ey / NODE_TOLERANCE) * NODE_TOLERANCE)

        # Find nearest existing node if not exact match
        if skey in coord_to_node:
            snode = coord_to_node[skey]
        else:
            # Find nearest node
            _, idx = nt.query([sx, sy])
            snode = ni[idx]

        if ekey in coord_to_node:
            enode = coord_to_node[ekey]
        else:
            _, idx = nt.query([ex, ey])
            enode = ni[idx]

        if snode != enode:
            wishing_edges[lid] = [[snode, enode, round(geom.length, 1)]]
        else:
            wishing_edges[lid] = []

    # Area centroids in WGS84
    areas_wgs = areas.to_crs(WGS84)
    centroids_wgs = [[round(c.x, 6), round(c.y, 6)] for c in areas_wgs.geometry.centroid]

    # Generate HTML
    print("Generating HTML...")
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
        nodes_wgs=nodes_wgs,
        edges_list=edges_list,
        wishing_edges=wishing_edges,
        centroids_wgs=centroids_wgs,
    )

    out_path = script_dir / 'bike_analysis.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Done! Open {out_path}")


def generate_html(*, areas_geojson, completed_geojson, construction_geojson,
                  wishing_geojson, lane_names, area_names, sens_data,
                  acc_orig_data, acc_dest_data, nodes_wgs, edges_list,
                  wishing_edges, centroids_wgs):

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
      <strong>Legend</strong>
      <div class="legend-item"><div class="legend-line" style="background:#27ae60"></div>Existing lanes</div>
      <div class="legend-item"><div class="legend-line" style="background:#f39c12"></div>Under construction</div>
      <div class="legend-item"><div class="legend-line" style="background:#e74c3c"></div>Wishing list</div>
      <div class="legend-item"><div class="legend-line" style="background:#9b59b6;height:6px"></div>Selected lane</div>
      <div class="legend-item"><div class="legend-line" style="background:#1565C0;height:6px"></div>Path on bike lane</div>
      <div class="legend-item"><div class="legend-line" style="background:#E65100;height:6px"></div>Path on road</div>
    </div>
    <div class="info" id="info">
      <strong id="infoTitle"></strong>
      <div id="infoBody"></div>
    </div>
  </div>
  <div class="sidebar">
    <div class="tabs">
      <button class="act" onclick="showTab('lanes',this)">Lanes</button>
      <button onclick="showTab('paths',this)">Paths</button>
    </div>
    <div id="lanes" class="tc act">
      <h3>Wishing List Lanes</h3>
      <div class="note">Click a lane to select it. Selected lanes are included in path calculations. Areas colored by accessibility (red=low, green=high).</div>
      <div id="laneList"></div>
    </div>
    <div id="paths" class="tc">
      <h3>Shortest Path</h3>
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
const NODES={js_json(nodes_wgs)};
const EDGES={js_json(edges_list)};
const WISHING_EDGES={js_json(wishing_edges)};
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

// Completed
if(COMPLETED.features.length)
  L.geoJSON(COMPLETED,{{style:{{color:"#27ae60",weight:3,opacity:.8}},
    onEachFeature:(f,l)=>l.bindPopup("<b>Existing:</b> "+(f.properties.Name||""))
  }}).addTo(map);

// Construction
if(CONSTRUCTION.features.length)
  L.geoJSON(CONSTRUCTION,{{style:{{color:"#f39c12",weight:3,opacity:.8}},
    onEachFeature:(f,l)=>l.bindPopup("<b>Under construction:</b> "+(f.properties.Name||""))
  }}).addTo(map);

// Wishing list
wishLyr=L.geoJSON(WISHING,{{
  style:f=>{{
    const s=sel.has(f.properties.lane_id);
    return {{color:s?"#9b59b6":"#e74c3c",weight:s?5:3,opacity:.8}};
  }},
  onEachFeature:(f,layer)=>{{
    const lid=f.properties.lane_id;
    layer.on("click",()=>{{sel.has(lid)?sel.delete(lid):sel.add(lid);refresh();}});
    layer.on("mouseover",()=>showInfo(lid));
    layer.on("mouseout",hideInfo);
  }}
}}).addTo(map);

// Populate area selects
(function(){{
  const opts=AREA_NAMES.map((n,i)=>'<option value="'+i+'">'+n+'</option>').join("");
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
    return {{color:s?"#9b59b6":"#e74c3c",weight:s?5:3,opacity:.8}};
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
}}

function buildLaneList(){{
  const imps=SENS[paramKey()]||[];
  // Build sorted list
  const items=LANE_NAMES.map((name,i)=>({{id:i,name:name,imp:imps[i]||0}}));
  items.sort((a,b)=>b.imp-a.imp);
  const container=document.getElementById("laneList");
  container.innerHTML=items.map(it=>{{
    const cls=sel.has(it.id)?"lane sel":"lane";
    return '<div class="'+cls+'" onclick="toggleLane('+it.id+')"><span>'+it.name+'</span><span class="pct">+'+it.imp.toFixed(2)+'%</span></div>';
  }}).join("");
}}

function toggleLane(id){{
  sel.has(id)?sel.delete(id):sel.add(id);
  refresh();
}}

function clearSel(){{sel.clear();refresh();}}

function getAccMode(){{
  const radio=document.querySelector('input[name="accMode"]:checked');
  return radio?radio.value:"origin";
}}

function updateAreaColors(){{
  const mode=getAccMode();
  const acc=(mode==="dest")?ACC_DEST[paramKey()]:ACC_ORIG[paramKey()];
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
    const result=dijkstra(parseInt(oi),parseInt(di),k);
    if(result&&result.segments.length>0){{
      // Draw segments with different colors
      pathLyrGroup=L.layerGroup();
      let totalLaneDist=0,totalRoadDist=0;
      for(const seg of result.segments){{
        const coords=[[seg.from[1],seg.from[0]],[seg.to[1],seg.to[0]]];
        const color=seg.bike?"#1565C0":"#E65100";
        const line=L.polyline(coords,{{color:color,weight:6,opacity:.85}});
        pathLyrGroup.addLayer(line);
        if(seg.bike)totalLaneDist+=seg.len;
        else totalRoadDist+=seg.len;
      }}
      pathLyrGroup.addTo(map);
      map.fitBounds(pathLyrGroup.getBounds(),{{padding:[30,30]}});

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

  // Build adjacency with edge info
  const adj={{}};
  for(const e of EDGES){{
    const len=e[2],bike=!!e[3];
    const w=bike?len:len*k;
    const a=String(e[0]),b=String(e[1]);
    if(!adj[a])adj[a]=[];
    if(!adj[b])adj[b]=[];
    adj[a].push({{n:b,w:w,len:len,bike:bike}});
    adj[b].push({{n:a,w:w,len:len,bike:bike}});
  }}

  // Add edges from selected wishing lanes (they are bike lanes)
  for(const lid of sel){{
    const edges=WISHING_EDGES[lid]||[];
    for(const e of edges){{
      const len=e[2];
      const w=len; // bike lane, no penalty
      const a=String(e[0]),b=String(e[1]);
      if(!adj[a])adj[a]=[];
      if(!adj[b])adj[b]=[];
      adj[a].push({{n:b,w:w,len:len,bike:true}});
      adj[b].push({{n:a,w:w,len:len,bike:true}});
    }}
  }}

  const dist={{}},prev={{}},prevEdge={{}},visited=new Set();
  dist[oNode]=0;
  let pq=[[0,oNode]];

  while(pq.length){{
    const[cd,cur]=pq.shift();
    if(visited.has(cur))continue;
    visited.add(cur);
    if(cur===dNode)break;
    for(const{{n,w,len,bike}}of(adj[cur]||[])){{
      if(visited.has(n))continue;
      const nd=cd+w;
      if(dist[n]===undefined||nd<dist[n]){{
        dist[n]=nd;prev[n]=cur;prevEdge[n]={{len:len,bike:bike}};
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
    segments.unshift({{from:NODES[p],to:NODES[c],len:e.len,bike:e.bike}});
    c=p;
  }}
  return {{segments:segments}};
}}

// === EVENT LISTENERS ===
document.getElementById("kSel").onchange=refresh;
document.getElementById("tSel").onchange=refresh;

// Initial render
buildLaneList();
updateAreaColors();
</script>
</body>
</html>'''


if __name__ == '__main__':
    main()
