"""Sensitivity analysis: rank lanes for different K and theta values."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import from rank_wishing_lanes
sys.path.insert(0, str(Path(__file__).parent))
from rank_wishing_lanes import (
    load_data, build_network_with_intersections, calculate_total_N,
    NODE_TOLERANCE, TARGET_CRS
)

script_dir = Path(__file__).parent

# Parameter grid
K_VALUES = [10, 50, 100, 200, 500]
THETA_VALUES = [-0.5, -1, -1.5, -2, -3]


def main():
    print("Loading data...")
    areas, roads, completed, construction, wishing = load_data()

    results = []

    for K in K_VALUES:
        for THETA in THETA_VALUES:
            print(f"\n=== K={K}, theta={THETA} ===")

            # Baseline
            G_base, nc, nt, ni = build_network_with_intersections(
                roads, [completed, construction], tolerance=NODE_TOLERANCE
            )
            base_N = calculate_total_N(G_base, nc, nt, ni, areas, THETA, K)
            print(f"Baseline N: {base_N:.2e}")

            # Each lane
            for i in range(len(wishing)):
                lane = wishing.iloc[[i]]
                name = lane['Name'].iloc[0] if 'Name' in lane.columns else f"Lane_{i}"
                G, nc, nt, ni = build_network_with_intersections(
                    roads, [completed, construction, lane], tolerance=NODE_TOLERANCE
                )
                N = calculate_total_N(G, nc, nt, ni, areas, THETA, K)
                imp = N - base_N
                pct = 100 * imp / base_N if base_N > 0 else 0
                results.append({'K': K, 'theta': THETA, 'lane': name, 'improvement_pct': pct})
                print(f"  {name[:30]:30s} {pct:+.3f}%")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(script_dir / 'sensitivity_analysis.csv', index=False)

    # Print top lane per K/theta
    print("\n=== TOP LANE PER K/THETA ===")
    for K in K_VALUES:
        for THETA in THETA_VALUES:
            subset = df[(df['K'] == K) & (df['theta'] == THETA)]
            top = subset.loc[subset['improvement_pct'].idxmax()]
            print(f"K={K:3d} θ={THETA:4.1f}: {top['lane'][:25]:25s} ({top['improvement_pct']:+.2f}%)")

    print(f"\nSaved: sensitivity_analysis.csv")


if __name__ == '__main__':
    main()
