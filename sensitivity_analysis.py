"""Rank wishing list lanes for different K and theta values."""
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial import cKDTree
import fiona
import warnings
warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039

# Parameter grid
K_VALUES = [10, 50, 100, 200, 500]
THETA_VALUES = [-0.5, -1, -1.5, -2, -3]

def build_network(roads_gdf, bike_lanes_list, tolerance=15, snap_tolerance=50):
    G = nx.Graph()
    coord_to_node, node_coords, node_counter = {}, {}, [0]

    def get_node_grid(x, y):
        key = (round(x/tolerance)*tolerance, round(y/tolerance)*tolerance)
        if key in coord_to_node: return coord_to_node[key]
        nid = node_counter[0]; node_counter[0] += 1
        coord_to_node[key] = nid; node_coords[nid] = (x, y)
        return nid

    for _, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString': continue
        coords = list(geom.coords)
        if len(coords) >= 2:
            s, e = get_node_grid(coords[0][0], coords[0][1]), get_node_grid(coords[-1][0], coords[-1][1])
            if s != e: G.add_edge(s, e, length=geom.length, has_bike_lane=False)

    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    tree = cKDTree(coords_array)

    def get_nearest(x, y):
        if len(coords_array) > 0:
            d, i = tree.query([x, y])
            if d <= snap_tolerance: return node_ids[i]
        return get_node_grid(x, y)

    for bl_gdf in bike_lanes_list:
        if bl_gdf is None: continue
        for _, row in bl_gdf.to_crs(TARGET_CRS).iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or geom.geom_type != 'LineString': continue
            coords = list(geom.coords)
            if len(coords) >= 2:
                s, e = get_nearest(coords[0][0], coords[0][1]), get_nearest(coords[-1][0], coords[-1][1])
                if s != e:
                    if G.has_edge(s, e): G[s][e]['has_bike_lane'] = True
                    else: G.add_edge(s, e, length=geom.length, has_bike_lane=True)

    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    return G, node_coords, cKDTree(coords_array), node_ids

def calc_total_N(G, node_coords, node_tree, node_ids, areas, theta, k):
    G = G.copy()
    for u, v in G.edges():
        G[u][v]['weight'] = G[u][v]['length'] if G[u][v].get('has_bike_lane') else G[u][v]['length'] * k

    centroids = areas.geometry.centroid
    center_nodes = [node_ids[node_tree.query([c.x, c.y])[1]] for c in centroids]
    pop, emp = areas['pop'].values, areas['emp'].values
    largest_cc = max(nx.connected_components(G), key=len)

    total_N = 0
    for i in range(len(areas)):
        if center_nodes[i] not in largest_cc: continue
        try: dists = nx.single_source_dijkstra_path_length(G, center_nodes[i], weight='weight')
        except: continue
        for j in range(len(areas)):
            if i != j and center_nodes[j] in dists:
                tau = max(dists[center_nodes[j]] / 1000, 0.1)
                total_N += pop[i] * emp[j] * (tau ** theta)
    return total_N

def main():
    print("Loading data...")
    areas = gpd.read_file(script_dir / "jer_areas.shp")
    areas = areas[areas['in_jeru'] == 1].copy()
    areas['pop'] = areas['pop_2025'].fillna(0)
    areas['emp'] = areas['emp_2025'].fillna(0)
    areas = areas.to_crs(TARGET_CRS)

    roads = gpd.read_file(script_dir / "jerusalem_roads.kml", driver='KML').to_crs(TARGET_CRS)
    completed = gpd.read_file(script_dir / "bike_lanes_completed.kml", driver='KML')
    construction = gpd.read_file(script_dir / "bike_lanes_construction.kml", driver='KML')
    wishing = gpd.read_file(script_dir / "bike_lanes_wishing_list.kml", driver='KML')

    results = []

    for K in K_VALUES:
        for THETA in THETA_VALUES:
            print(f"\n=== K={K}, theta={THETA} ===")

            # Baseline
            G_base, nc, nt, ni = build_network(roads, [completed, construction])
            base_N = calc_total_N(G_base, nc, nt, ni, areas, THETA, K)
            print(f"Baseline N: {base_N:.2e}")

            # Each lane
            for i in range(len(wishing)):
                lane = wishing.iloc[[i]]
                name = lane['Name'].iloc[0] if 'Name' in lane.columns else f"Lane_{i}"
                G, nc, nt, ni = build_network(roads, [completed, construction, lane])
                N = calc_total_N(G, nc, nt, ni, areas, THETA, K)
                imp = N - base_N
                pct = 100 * imp / base_N if base_N > 0 else 0
                results.append({'K': K, 'theta': THETA, 'lane': name, 'improvement_pct': pct})
                print(f"  {name[:30]:30s} {pct:+.3f}%")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(script_dir / 'sensitivity_analysis.csv', index=False)

    # Create pivot table for top lane per K/theta
    print("\n=== TOP LANE PER K/THETA ===")
    for K in K_VALUES:
        for THETA in THETA_VALUES:
            subset = df[(df['K'] == K) & (df['theta'] == THETA)]
            top = subset.loc[subset['improvement_pct'].idxmax()]
            print(f"K={K:3d} θ={THETA:4.1f}: {top['lane'][:25]:25s} ({top['improvement_pct']:+.2f}%)")

    print(f"\nSaved: sensitivity_analysis.csv")

if __name__ == '__main__':
    main()
