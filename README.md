# Jerusalem Bike Lane Analysis

Interactive map for analyzing Jerusalem's bike lane network and planning new routes.

## Project Structure

```
bike/
├── generate_interactive_map.py  # Main script to generate the HTML map
├── bike_analysis.html           # Generated interactive map
├── METHODOLOGY.md               # Detailed methodology documentation
├── README.md                    # This file
│
├── Input Data Files:
│   ├── jer_areas.shp/dbf/shx/prj  # Jerusalem statistical areas
│   ├── jerusalem_roads.kml         # Road network
│   ├── bike_lanes_completed.kml    # Existing bike lanes
│   ├── bike_lanes_construction.kml # Lanes under construction
│   └── bike_lanes_wishing_list.kml # Proposed future lanes
│
├── paper/                       # Academic paper and figure generation
│   ├── ACADEMIC_PAPER.tex/pdf   # LaTeX paper source and output
│   ├── generate_paper_figures_*.py  # Figure generation scripts
│   ├── figures/                 # PDF figures for the paper
│   ├── tables/                  # LaTeX tables for the paper
│   └── html_figures/            # HTML figure outputs
│
└── scripts/                     # Analysis and test scripts
    ├── analyze_bike_gaps.py
    ├── test_network_connectivity.py
    └── ...
```

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

## Documentation

See `METHODOLOGY.md` for details on the accessibility model and calculations.
