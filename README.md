# Bike Path Shortest Route Calculator - Jerusalem

Calculate the shortest bike path between the centers of areas in Jerusalem using bike lane network data.

## Features

- Load bike lanes from Shapefile or GeoJSON
- Load areas (neighborhoods/zones) from Shapefile
- Calculate centroids of each area
- Build a graph network from bike lanes
- Calculate shortest paths between all pairs of area centers
- Export distance matrix and path geometries

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python bike_path_calculator.py <bike_lanes_file> <areas_file>
```

### Full Options

```bash
python bike_path_calculator.py bike_lanes.shp areas.shp \
    -o ./output \
    -n name_column \
    -t 1.0 \
    --crs EPSG:2039
```

### Arguments

| Argument | Description |
|----------|-------------|
| `bike_lanes` | Path to bike lanes file (Shapefile or GeoJSON) |
| `areas` | Path to areas Shapefile |
| `-o, --output` | Output directory (default: ./output) |
| `-n, --name-column` | Column name for area names in areas shapefile |
| `-t, --tolerance` | Node snapping tolerance in CRS units (default: 1.0) |
| `--crs` | Target CRS for calculations (default: EPSG:2039 - Israel TM Grid) |

## Input Data Requirements

### Bike Lanes File
- Format: Shapefile (.shp) or GeoJSON (.geojson)
- Geometry: LineString or MultiLineString
- Should contain the bike lane network as connected line segments

### Areas File
- Format: Shapefile (.shp)
- Geometry: Polygon or MultiPolygon
- Optionally include a name column to identify areas

## Output Files

The script generates the following files in the output directory:

| File | Description |
|------|-------------|
| `distance_matrix.csv` | Distance matrix between all area centers (meters) |
| `shortest_paths.geojson` | GeoJSON with path geometries between all pairs |
| `area_centers.geojson` | GeoJSON with area centroid locations |
| `summary.json` | Summary statistics |

## Example

```bash
# Calculate paths between Jerusalem neighborhoods
python bike_path_calculator.py \
    data/jerusalem_bike_lanes.shp \
    data/jerusalem_neighborhoods.shp \
    -n NEIGHBORHOOD_NAME \
    -o results/
```

## Output Example

### Distance Matrix (CSV)
```
,Area_0,Area_1,Area_2
Area_0,0.0,1234.5,2345.6
Area_1,1234.5,0.0,1567.8
Area_2,2345.6,1567.8,0.0
```

### Summary (JSON)
```json
{
  "total_areas": 10,
  "total_path_pairs": 45,
  "average_path_length": 3456.78,
  "max_path_length": 8901.23,
  "min_path_length": 567.89
}
```

## Notes

- The script uses EPSG:2039 (Israel TM Grid) by default for accurate distance calculations in meters
- If areas are not connected via the bike network, those paths will show as infinite distance
- The tolerance parameter helps connect nearby network segments that should be joined

## Dependencies

- geopandas - Geographic data handling
- networkx - Graph operations and shortest path algorithms
- numpy - Numerical operations
- pandas - Data manipulation
- shapely - Geometric operations
- scipy - Spatial indexing (cKDTree)
- pyproj - Coordinate reference system transformations
- fiona - Reading geospatial files
