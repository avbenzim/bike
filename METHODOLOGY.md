# Jerusalem Bike Lane Analysis - Methodology

## Overview

This tool ranks proposed ("wishing list") bike lanes by their potential contribution to city-wide accessibility. It uses a gravity-based accessibility model to measure how well people can reach destinations across the city, with bike lanes significantly reducing the effective travel cost.

## The Accessibility Model

### Core Formula

The total accessibility metric N is computed as:

```
N = Σᵢ Σⱼ Pᵢ × Dⱼ × τᵢⱼ^θ
```

Where:
- **Pᵢ** = Population of area i (potential trip origins)
- **Dⱼ** = Weighted destination attractiveness of area j (see Multi-Destination Model below)
- **τᵢⱼ** = Travel cost (shortest path distance in km) from area i to area j
- **θ** = Distance decay parameter (negative, typically -1 to -2)

### Multi-Destination Model

The destination weight Dⱼ combines multiple destination types:

```
Dⱼ = w_emp × Eⱼ + w_edu × Sⱼ + w_transit × Tⱼ
```

Where:
- **Eⱼ** = Employment in area j
- **Sⱼ** = Number of students in academic institutions in area j
- **Tⱼ** = Daily transit passengers in area j (train stations)
- **w_emp, w_edu, w_transit** = User-adjustable weights (default: equal, summing to 1)

This allows the model to evaluate bike lanes not just for commuting to work, but also for reaching universities, colleges, and transit hubs.

#### Education Data
Academic institutions are assigned to their containing statistical area, weighted by student count. Institutions include Hebrew University (3 campuses), Bezalel, Hadassah College, Machon Lev, David Yellin, Azrieli Engineering, the Music Academy, Sam Spiegel, Musrara, and Al-Quds University (~53,600 students total).

#### Transit Data
Train stations are assigned to their containing statistical area, weighted by daily passenger count. Currently includes Yitzhak Navon station (~15,000 daily passengers).

### Parameters

#### K - No-Lane Penalty
Roads without bike lanes are penalized by multiplying their length by K and a road discomfort factor:
- `weight = length` for roads WITH bike lanes
- `weight = length × K × discomfort` for roads WITHOUT bike lanes

Higher K values mean cyclists strongly prefer bike lanes, even if it means longer routes:
- K=10: Mild preference for bike lanes
- K=100: Strong preference (default)
- K=500: Very strong preference

#### Road Discomfort Factor
Not all roads are equally unpleasant to cycle on. Roads are classified by their OSM highway type, and each type receives a discomfort multiplier:

| Road Type | Discomfort Factor | Examples |
|-----------|------------------|----------|
| residential / living_street | 1.0 | Neighborhood streets |
| tertiary | 1.5 | Minor urban roads |
| secondary | 2.0 | Urban arterials |
| primary | 3.0 | Major arterials (Golda Meir, Herzl) |
| trunk | 4.0 | Major roads (Bazak, Derech Hebron) |
| motorway | 5.0 | Expressways (Begin) |

This means a bike lane on Derech Begin (motorway, discomfort=5) removes 5× more penalty than a bike lane on a quiet residential street, reflecting its much greater impact on cyclist comfort and safety.

Road classification data is sourced from OpenStreetMap via the Overpass API (`jerusalem_roads_major.csv`). Roads not found in the classification default to discomfort=1.0.

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
- Provides shortcuts between population centers and destinations (jobs, universities, transit)
- Reduces effective travel cost for many origin-destination pairs
- Removes high discomfort penalties on major roads

## Network Construction

### Data Sources

| Data | Source |
|------|--------|
| **Statistical Areas** | Jerusalem Transportation Master Plan Team |
| **Population** | Jerusalem Transportation Master Plan Team |
| **Employment** | Jerusalem Transportation Master Plan Team |
| **Completed Bike Lanes** | Jerusalem Transportation Master Plan Team |
| **Under Construction Bike Lanes** | Jerusalem Transportation Master Plan Team |
| **Wishing List Bike Lanes** | The author |
| **Road Network** | OpenStreetMap |
| **Road Classification** | OpenStreetMap (Overpass API) |
| **Academic Institutions** | OpenStreetMap + student count estimates |
| **Train Station Passengers** | Israel Railways estimates |

### Source Files

| File | Description |
|------|-------------|
| `jer_areas.shp` | Statistical areas with population and employment data |
| `bike_lanes_completed.kml` | Existing bike lanes |
| `bike_lanes_construction.kml` | Lanes under construction |
| `bike_lanes_wishing_list.kml` | Proposed future lanes |
| `jerusalem_roads.kml` | Road network (filtered to Jerusalem with 1km buffer) |
| `jerusalem_roads_major.csv` | OSM highway classification for road discomfort factors |

### Graph Building
1. Roads are converted to a graph with nodes at endpoints
2. Bike lanes overlay the road network
3. Each edge stores: length (meters), has_bike_lane (boolean), discomfort factor
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

### Origin Accessibility (Destinations Reachable)
For each area i:
```
acc_origin[i] = Σⱼ Dⱼ × τᵢⱼ^θ
```
This measures how many weighted destinations are accessible FROM area i (where Dⱼ combines employment, education, and transit).

### Destination Accessibility (People Reaching)
For each area j:
```
acc_dest[j] = Σᵢ Pᵢ × τᵢⱼ^θ
```
This measures how many people can reach area j (e.g., how accessible is a workplace or campus).

## Interactive Map Features

### Wishing Lane Selection
- Click lanes on map or sidebar to select/deselect
- Selected lanes are highlighted in purple
- Total estimated improvement shown in header

### Area Coloring
- Red = Low accessibility
- Green = High accessibility
- Toggle between origin and destination views

### Shortest Path Finder
- Select origin and destination areas from dropdowns
- Or click directly on the map to choose custom points
- Path computed using Dijkstra's algorithm with K penalty
- Blue segments = bike lanes, Orange segments = roads
- Statistics show distance breakdown

### Custom Lane Drawing
Users can draw their own proposed bike lanes:
- Click "Draw New Lane" to start drawing mode
- Click on map to add points (automatically snaps to nearby road nodes within 15m)
- Double-click or press "Finish" to complete the lane
- Custom lanes integrate with the network for path finding
- Export lanes as GeoJSON for sharing or import previously saved lanes

### Lane Snapping
When drawing custom lanes:
- Points automatically snap to nearby road network nodes (15m threshold)
- Points also snap to existing custom lane endpoints
- Visual feedback shows when snapping occurs (green markers)
- Ensures drawn lanes properly connect to the road network

### Parameters
- Adjust K and θ to see how rankings change
- Adjust destination weights to prioritize employment, education, or transit accessibility
- Different parameter combinations favor different lane types:
  - High K: Lanes that connect existing infrastructure
  - Low θ (more negative): Lanes serving local trips
  - High θ (less negative): Lanes enabling longer commutes
  - High education weight: Lanes connecting to university campuses
  - High transit weight: Lanes connecting to train stations

## Output Files

| File | Description |
|------|-------------|
| `generate_interactive_map.py` | Script to generate the interactive HTML |
| `bike_analysis.html` | Self-contained interactive visualization |

## Technical Notes

### Performance
- Network has ~9,600 nodes and ~13,700 edges
- Each accessibility calculation requires ~100 Dijkstra runs (one per area)
- All calculations run client-side in the browser

### Assumptions
- Straight-line connections between lane endpoints and nearest road nodes
- All roads are bidirectional
- No turn penalties or traffic signals modeled
- Population and employment from projections

## References

- Donaldson, D., & Hornbeck, R. (2016). Railroads and American economic growth: A "market access" approach. *The Quarterly Journal of Economics*, 131(2), 799-858.
- Tsivanidis, N. (2024). Evaluating the Impact of Urban Transit Infrastructure: Evidence from Bogotá's TransMilenio. *American Economic Review*, 116(2), 418-463.
