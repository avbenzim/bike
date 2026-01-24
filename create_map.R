# =============================================================================
# Create Map of Jerusalem Areas and Bike Lanes
# =============================================================================

library(sf)
library(ggplot2)
library(dplyr)

# Clear workspace
rm(list = ls())

# Get script directory
script_dir <- tryCatch({
  dirname(rstudioapi::getSourceEditorContext()$path)
}, error = function(e) {
  getwd()
})

# Load data
message("Loading data...")

# Load bike lanes
bike_lanes_completed <- st_read(file.path(script_dir, "bike_lanes_completed.kml"), quiet = TRUE)
bike_lanes_construction <- st_read(file.path(script_dir, "bike_lanes_construction.kml"), quiet = TRUE)

# Load areas
areas <- st_read(file.path(script_dir, "jer_areas.shp"), quiet = TRUE)

# Filter to Jerusalem areas
areas <- areas %>% filter(in_jeru == 1)

message("Loaded ", nrow(areas), " areas")
message("Loaded ", nrow(bike_lanes_completed), " completed bike lane segments")
message("Loaded ", nrow(bike_lanes_construction), " bike lanes under construction")

# Transform to WGS84 for consistency
bike_lanes_completed <- st_transform(bike_lanes_completed, 4326)
bike_lanes_construction <- st_transform(bike_lanes_construction, 4326)
areas <- st_transform(areas, 4326)

# Create the map
message("Creating map...")

p <- ggplot() +
  # Areas as base layer
  geom_sf(data = areas, fill = "lightyellow", color = "gray50", linewidth = 0.3, alpha = 0.7) +

  # Bike lanes under construction (dashed orange)
  geom_sf(data = bike_lanes_construction, color = "orange", linewidth = 1.2, linetype = "dashed") +

  # Completed bike lanes (solid green)
  geom_sf(data = bike_lanes_completed, color = "darkgreen", linewidth = 1) +

  # Styling
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray40"),
    legend.position = "bottom",
    panel.grid = element_line(color = "gray90"),
    axis.text = element_text(size = 8)
  ) +
  labs(
    title = "Jerusalem Bike Lanes Network",
    subtitle = "Green: Completed | Orange (dashed): Under Construction",
    x = "Longitude",
    y = "Latitude"
  )

# Save the map
output_file <- file.path(script_dir, "jerusalem_bike_lanes_map.png")
ggsave(output_file, plot = p, width = 12, height = 10, dpi = 200)
message("Map saved to: ", output_file)

# Display the plot
print(p)
