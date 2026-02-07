# Jerusalem Bike Lane Analysis - Methodology

## Overview

This tool ranks proposed ("wishing list") bike lanes by their potential contribution to city-wide accessibility. It uses a gravity-based accessibility model to measure how well people can reach jobs across the city, with bike lanes significantly reducing the effective travel cost.

## The Accessibility Model

### Core Formula

The total accessibility metric N is computed as:

```
N = Σᵢ Σⱼ Pᵢ × Eⱼ × τᵢⱼ^θ
```

Where:
- **Pᵢ** = Population of area i (potential trip origins)
- **Eⱼ** = Employment in area j (potential trip destinations)
- **τᵢⱼ** = Travel cost (shortest path distance in km) from area i to area j
- **θ** = Distance decay parameter (negative, typically -1 to -2)

### Parameters

#### K - No-Lane Penalty
Roads without bike lanes are penalized by multiplying their length by K:
- `weight = length` for roads WITH bike lanes
- `weight = length × K` for roads WITHOUT bike lanes

Higher K values mean cyclists strongly prefer bike lanes, even if it means longer routes:
- K=10: Mild preference for bike lanes
- K=100: Strong preference (default)
- K=500: Very strong preference

#### θ (Theta) - Distance Decay
Controls how quickly accessibility decreases with distance:
- θ = -0.5: Slow decay (long trips acceptable)
- θ = -1.0: Moderate decay (default)
- θ = -2.0: Fast decay (only nearby destinations matter)
- θ = -3.0: Very fast decay (very local trips only)

## How Lanes Are Ranked

### Process

1. **Baseline Calculation**: Compute total N using existing bike lanes (completed + under construction)

2. **Per-Lane Evaluation**: For each wishing list lane:
   - Add the lane to the network
   - Recompute total N
   - Calculate improvement: `ΔN = N_with_lane - N_baseline`
   - Calculate percentage improvement: `%Δ = 100 × ΔN / N_baseline`

3. **Ranking**: Sort lanes by improvement (highest first)

### Interpretation

A lane's improvement reflects how much it:
- Connects previously disconnected bike infrastructure
- Provides shortcuts between population centers and employment hubs
- Reduces effective travel cost for many origin-destination pairs

## Network Construction

### Data Sources

| Data | Source | Year |
|------|--------|------|
| **Statistical Areas** | Jerusalem Transportation Master Plan Team | 2025 projections |
| **Population (pop_2025)** | Jerusalem Transportation Master Plan Team | 2025 projections |
| **Employment (emp_2025)** | Jerusalem Transportation Master Plan Team | 2025 projections |
| **Completed Bike Lanes** | Jerusalem Transportation Master Plan Team | Current |
| **Under Construction Bike Lanes** | Jerusalem Transportation Master Plan Team | Current |
| **Wishing List Bike Lanes** | The author | Proposed |
| **Road Network** | OpenStreetMap via ISR.parquet | Current |

- **Areas**: Statistical areas with population and employment projections for 2025 (Shapefile: `jer_areas.shp`)
- **Bike Lanes**: Three KML files for completed, under construction, and wishing list lanes (provided by Transportation Master Plan Team)
- **Roads**: Jerusalem road network extracted from OpenStreetMap (`jerusalem_roads.kml`), filtered to Jerusalem area with 1km buffer

### Graph Building
1. Roads are converted to a graph with nodes at endpoints
2. Bike lanes overlay the road network
3. Each edge stores: length (meters), has_bike_lane (boolean)
4. Node tolerance of 15m is used to merge nearby endpoints

### Coordinate Systems
- Internal calculations use Israel TM (EPSG:2039) for accurate distances
- Display uses WGS84 (EPSG:4326) for web mapping

### Network Connectivity

The tool ensures the network is fully connected through several mechanisms:

#### Node Merging
- **Tolerance**: Nodes within 15 meters of each other are merged into a single node
- **Purpose**: Handles imprecise GPS coordinates and ensures lane endpoints connect properly to roads

#### Intersection Detection
- Bike lanes are overlaid on the road network
- Intersection points between bike lanes and roads are detected automatically
- New nodes are created at every intersection point
- Edges are split at intersection points to enable routing through the network

#### Gap Connection Algorithm
For bike lanes with gaps between segments:

1. **Endpoint Extraction**: Extract start and end points from all lane geometries
2. **KD-Tree Indexing**: Build spatial index for efficient nearest-neighbor queries
3. **Dangling Endpoint Detection**: Identify endpoints not touching other lanes (within 1m tolerance)
4. **Gap Bridging**: Connect dangling endpoints to nearest neighbor within 50m tolerance
   - Preference given to connecting two dangling endpoints (both isolated)
   - Falls back to connecting to any nearby endpoint

#### Component Connection
For disconnected network components:

1. **Component Detection**: Use NetworkX to find all connected components
2. **Minimum Spanning Tree Approach**: Connect isolated components by adding edges between closest nodes
3. **Result**: Ensures every area can reach every other area through some path

This connectivity fixing is essential because raw GIS data often has small gaps, coordinate mismatches, or isolated segments that would otherwise break shortest path calculations.

## Area Accessibility Metrics

### Origin Accessibility (Jobs Reachable)
For each area i:
```
acc_origin[i] = Σⱼ Eⱼ × τᵢⱼ^θ
```
This measures how many jobs are accessible FROM area i.

### Destination Accessibility (People Reaching)
For each area j:
```
acc_dest[j] = Σᵢ Pᵢ × τᵢⱼ^θ
```
This measures how many people can reach area j (e.g., how accessible is a workplace).

## Interactive Map Features

### Lane Selection
- Click lanes on map or sidebar to select/deselect
- Selected lanes are highlighted in purple
- Total estimated improvement shown in header

### Area Coloring
- Red = Low accessibility
- Green = High accessibility
- Toggle between origin and destination views

### Shortest Path
- Select origin and destination areas
- Path computed using Dijkstra's algorithm with K penalty
- Blue segments = bike lanes, Orange segments = roads
- Statistics show distance breakdown

### Parameters
- Adjust K and θ to see how rankings change
- Different parameter combinations favor different lane types:
  - High K: Lanes that connect existing infrastructure
  - Low θ (more negative): Lanes serving local trips
  - High θ (less negative): Lanes enabling longer commutes

## Files

| File | Description |
|------|-------------|
| `rank_wishing_lanes.py` | Core ranking algorithm |
| `sensitivity_analysis.py` | Runs ranking for multiple K/θ combinations |
| `generate_interactive_map.py` | Generates the interactive HTML |
| `bike_analysis.html` | Self-contained interactive visualization |
| `sensitivity_analysis.csv` | Results for all K/θ combinations |
| `wishing_list_ranking.csv` | Detailed ranking for default parameters |

## Technical Notes

### Performance
- Network has ~8,000 nodes and ~11,000 edges
- Each accessibility calculation requires ~100 Dijkstra runs (one per area)
- Full sensitivity analysis (25 parameter combinations × 24 lanes) takes ~30 minutes

### Assumptions
- Straight-line connections between lane endpoints and nearest road nodes
- All roads are bidirectional
- No turn penalties or traffic signals modeled
- Population and employment from 2025 projections

## References

- Donaldson, D., & Hornbeck, R. (2016). Railroads and American economic growth: A "market access" approach. *The Quarterly Journal of Economics*, 131(2), 799-858.
- Tsivanidis, N. (2024). Evaluating the Impact of Urban Transit Infrastructure: Evidence from Bogotá's TransMilenio. *American Economic Review*, 116(2), 418-463.

## Close

This methodology provides a systematic, data-driven approach to prioritizing bike lane investments. By combining:

- **Gravity model physics**: Captures the fundamental relationship between accessibility, distance, and demand
- **Network analysis**: Ensures realistic routing through the actual road/bike lane network
- **Sensitivity analysis**: Tests robustness across different cyclist behavior assumptions (K) and trip distance preferences (θ)
- **Interactive visualization**: Enables planners to explore scenarios and understand trade-offs

The rankings should be considered alongside other factors not modeled here, including:
- Construction costs and feasibility
- Safety considerations and accident data
- Equity and access for underserved neighborhoods
- Integration with public transit
- Political and community priorities
