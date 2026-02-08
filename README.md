# Jerusalem Bike Lane Analysis

Interactive map for analyzing Jerusalem's bike lane network and planning new routes.

## Output

**`bike_analysis.html`** - Self-contained interactive map with:
- Existing bike lanes (completed and under construction)
- Proposed "wishing list" lanes with impact analysis
- Custom lane drawing with network snapping
- Shortest path calculations between areas
- Accessibility scoring for neighborhoods

Open the HTML file in any browser - no server required.

## Regenerating the HTML

### Requirements

```bash
pip install geopandas shapely
```

### Generate

```bash
python generate_interactive_map.py
```

## Source Data Files

| File | Description |
|------|-------------|
| `jer_areas.shp` | Jerusalem statistical areas (shapefile) |
| `jerusalem_roads.kml` | Road network |
| `bike_lanes_completed.kml` | Existing bike lanes |
| `bike_lanes_construction.kml` | Lanes under construction |
| `bike_lanes_wishing_list.kml` | Proposed future lanes |

## Documentation

See `METHODOLOGY.md` for details on the accessibility model and calculations.
