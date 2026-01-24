# Bike Path Shortest Route Calculator - Jerusalem

Calculate the shortest bike path between the centers of areas in Jerusalem using bike lane network data.

## Features

- Load bike lanes from KML file
- Load areas (neighborhoods/zones) from Shapefile
- Calculate centroids of each area
- Build a network graph from bike lanes using `sfnetworks`
- Calculate shortest paths between all pairs of area centers
- Export distance matrix, path geometries, and visualizations

## Data Files

The repository includes Jerusalem data:
- `bike_lanes_completed.kml` - Completed bike lanes in Jerusalem
- `bike_lanes_construction.kml` - Bike lanes under construction
- `jer_areas.shp` - Jerusalem statistical areas shapefile

## Installation

Install required R packages:

```r
install.packages(c("sf", "sfnetworks", "tidygraph", "dplyr", "igraph", "tidyr", "ggplot2", "jsonlite"))
```

## Usage

### Running the Script

```r
# From R/RStudio
source("bike_path_calculator.R")

# From command line
Rscript bike_path_calculator.R
```

### Configuration

Edit the configuration section at the top of `bike_path_calculator.R`:

```r
# Input files
BIKE_LANES_FILE <- "bike_lanes_completed.kml"
AREAS_FILE <- "jer_areas.shp"

# Output directory
OUTPUT_DIR <- "./output"

# CRS for distance calculations (Israel TM Grid - meters)
TARGET_CRS <- 2039
```

### Customizing for Your Data

The script filters Jerusalem areas using `in_jeru == 1`. Modify the `main()` function if your data has different filtering requirements:

```r
# Load areas with custom filter
areas <- load_areas(AREAS_FILE, filter_column = "your_column", filter_value = "your_value")

# Or load all areas without filtering
areas <- load_areas(AREAS_FILE)
```

## Input Data Requirements

### Bike Lanes File
- Format: KML (.kml), Shapefile (.shp), or GeoJSON (.geojson)
- Geometry: LineString or MultiLineString
- Should contain the bike lane network as connected line segments

### Areas File
- Format: Shapefile (.shp)
- Geometry: Polygon or MultiPolygon
- Optionally include a name column to identify areas (default: `STAT11_HEB`)

## Output Files

The script generates the following files in the output directory:

| File | Description |
|------|-------------|
| `distance_matrix.csv` | Distance matrix between all area centers (meters) |
| `shortest_paths.geojson` | GeoJSON with path geometries between all pairs |
| `area_centers.geojson` | GeoJSON with area centroid locations |
| `summary.json` | Summary statistics |
| `network_map.png` | Visualization of the bike network and area centers |

## Output Example

### Distance Matrix (CSV)
```
area,Area_0,Area_1,Area_2
Area_0,0,1234.5,2345.6
Area_1,1234.5,0,1567.8
Area_2,2345.6,1567.8,0
```

### Summary (JSON)
```json
{
  "total_areas": 10,
  "total_path_pairs": 45,
  "connected_pairs": 42,
  "average_path_length_m": 3456.78,
  "min_path_length_m": 567.89,
  "max_path_length_m": 8901.23,
  "median_path_length_m": 3200.5
}
```

## Notes

- The script uses EPSG:2039 (Israel TM Grid) by default for accurate distance calculations in meters
- If areas are not connected via the bike network, those paths will show as infinite distance in the matrix
- The `sfnetworks` package automatically handles network topology and node connectivity

## Dependencies

| Package | Purpose |
|---------|---------|
| `sf` | Spatial data handling |
| `sfnetworks` | Spatial network analysis |
| `tidygraph` | Tidy network manipulation |
| `igraph` | Graph algorithms (shortest paths) |
| `dplyr` | Data manipulation |
| `tidyr` | Data tidying |
| `ggplot2` | Visualization |
| `jsonlite` | JSON export |

## Related Files

- `jer_areas.r` - Original analysis script for population density visualization
