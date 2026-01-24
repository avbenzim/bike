"""
Create Map of Jerusalem Areas and Bike Lanes
"""

import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import fiona

# Configuration
script_dir = Path(__file__).parent
BIKE_LANES_COMPLETED = script_dir / "bike_lanes_completed.kml"
BIKE_LANES_CONSTRUCTION = script_dir / "bike_lanes_construction.kml"
AREAS_FILE = script_dir / "jer_areas.shp"
OUTPUT_FILE = script_dir / "jerusalem_bike_lanes_map.png"

# Enable fiona KML driver
fiona.drvsupport.supported_drivers['KML'] = 'rw'

print("Loading data...")

# Load bike lanes
bike_completed = gpd.read_file(BIKE_LANES_COMPLETED, driver='KML')
bike_construction = gpd.read_file(BIKE_LANES_CONSTRUCTION, driver='KML')

# Load areas
areas = gpd.read_file(AREAS_FILE)

# Filter to Jerusalem areas
areas = areas[areas['in_jeru'] == 1]

print(f"Loaded {len(areas)} areas")
print(f"Loaded {len(bike_completed)} completed bike lane segments")
print(f"Loaded {len(bike_construction)} bike lanes under construction")

# Transform to WGS84
bike_completed = bike_completed.to_crs(4326)
bike_construction = bike_construction.to_crs(4326)
areas = areas.to_crs(4326)

# Create the map
print("Creating map...")

fig, ax = plt.subplots(1, 1, figsize=(14, 12))

# Plot areas as base layer
areas.plot(ax=ax, facecolor='lightyellow', edgecolor='gray', linewidth=0.5, alpha=0.7)

# Plot bike lanes under construction (orange dashed)
bike_construction.plot(ax=ax, color='orange', linewidth=2, linestyle='--', label='Under Construction')

# Plot completed bike lanes (green solid)
bike_completed.plot(ax=ax, color='darkgreen', linewidth=1.5, label='Completed')

# Styling
ax.set_title('Jerusalem Bike Lanes Network', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Longitude', fontsize=10)
ax.set_ylabel('Latitude', fontsize=10)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='darkgreen', linewidth=2, label='Completed'),
    Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='Under Construction'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Grid
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()

# Save
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Map saved to: {OUTPUT_FILE}")

plt.show()
