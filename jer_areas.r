library(sf)
library(ggplot2)
library(dplyr)

# remove data from memory
rm(list = ls())

# Load KML (for geometry)
kml_data <- st_read("/Users/shmuelsan/Dropbox/HUJI/misc/twitter/jer_areas/oren.kml")

# Load CSV (for attributes)
csv_data <- fread("/Users/shmuelsan/Dropbox/HUJI/misc/twitter/jer_areas/oren.csv", fill = TRUE)

# Combine
dt_combined <- cbind(kml_data, csv_data)

# Filter
bike_lanes_completed <- dt_combined[dt_combined$status_tex == "בוצע" | dt_combined$openyear_n == "בוצע", ]
bike_lanes_construction <- dt_combined[dt_combined$status_tex == "בביצוע"| dt_combined$openyear_n == "בביצוע", ]

# Transform to WGS84
bike_lanes_completed <- st_transform(bike_lanes_completed, crs = 4326)
bike_lanes_construction <- st_transform(bike_lanes_construction, crs = 4326)

# Read the shapefile
map_data <- st_read("/Users/shmuelsan/Dropbox/HUJI/misc/twitter/jer_areas/jer_areas.shp")

# Filter to keep only Jerusalem areas
map_data <- map_data |> 
  filter(in_jeru == 1 )


# Calculate area and population density
map_data <- map_data |> 
  mutate(
    area_km2 = as.numeric(st_area(geometry)) / 1e6,
    pop_density_2020_km2 = (pop_2020) / area_km2,
    pop_density_2040_km2 = (pop_2040) / area_km2,
    pop_change_per = (pop_2040-pop_2020) / pop_2020 * 100 ,
    pop_change_km2 = (pop_2040-pop_2020) / area_km2
  )

map_data <- map_data |>
  mutate(
    pop_density_2040_km2 = ifelse(pop_density_2040_km2 > 40000, 40000, pop_density_2040_km2)
  )

# Plot with different color options
p1 = ggplot(map_data) +
  geom_sf(aes(fill = pop_density_2020_km2)) +
  scale_fill_gradientn(colors = terrain.colors(10), name = "Pop 2020 per km²")

# Plot with different color options
p2 = ggplot(map_data) +
  geom_sf(aes(fill = (pop_change_per))) +
  scale_fill_gradientn(colors = terrain.colors(10), name = "Pop percentage change")


# Plot with different color options
p3 = ggplot(map_data) +
  geom_sf(aes(fill = log(pop_change_km2))) +
  scale_fill_gradientn(colors = heat.colors(10, rev = TRUE), name = "Pop change 2020-40 per km² (in logs)")

# Plot with different color options
p4 = ggplot(map_data) +
  geom_sf(aes(fill = log(pop_density_2040_km2))) +
  scale_fill_gradientn(colors = terrain.colors(10), name = "Pop 2040 per km² (in logs)")


# Create tmap (tmap v4 syntax)
g <- tm_shape(map_data) +
  tm_polygons(
    fill = "pop_density",
    fill.scale = tm_scale_continuous(trans = "log"),
    fill_alpha = 0.6,
    fill.legend = tm_legend(title = "Pop per m²")
  ) +
  tm_shape(bike_lanes_completed) +
  tm_lines(col = "blue", lwd = 3) +
  tm_shape(bike_lanes_construction) +
  tm_lines(col = "orange", lwd = 2, lty = "dashed") +
  tm_basemap("OpenStreetMap")

g <- tm_shape(map_data) +
  tm_fill(col = "pop_density_2040_km2", alpha = 0.6, palette = "Reds", title = "Pop per m²") +
  tm_borders() +
  tm_shape(bike_lanes_completed) +
  tm_lines(col = "red", lwd = 2) +
  tm_shape(bike_lanes_construction) +
  tm_lines(col = "blue", lwd = 2) 

# save
tmap_mode("plot")
tmap_save(g, "/Users/shmuelsan/Dropbox/HUJI/misc/twitter/jer_areas/jerusalem_bike_lanes.png", width = 10, height = 8, dpi = 300)

ggsave("/Users/shmuelsan/Dropbox/HUJI/misc/twitter/jer_areas/jerusalem_pop_2020.png", plot = p1, width = 12, height = 10, dpi = 600)
ggsave("/Users/shmuelsan/Dropbox/HUJI/misc/twitter/jer_areas/jerusalem_pop_change_percentage.png", plot = p2, width = 12, height = 10, dpi = 600)
ggsave("/Users/shmuelsan/Dropbox/HUJI/misc/twitter/jer_areas/jerusalem_pop_change_2020_40.png", plot = p3, width = 12, height = 10, dpi = 600)
ggsave("/Users/shmuelsan/Dropbox/HUJI/misc/twitter/jer_areas/jerusalem_pop_change_2040.png", plot = p4, width = 12, height = 10, dpi = 600)



g <- tm_shape(map_data) +
  tm_fill(col = "pop_density_2040_km2", alpha = 0.6, palette = "Reds", title = "Pop per m²") +
  tm_borders() +
  tm_shape(bike_lanes_completed) +
  tm_lines(col = "red", lwd = 2) +
  tm_shape(bike_lanes_construction) +
  tm_lines(col = "blue", lwd = 2) 

# Save
tmap_save(g, "/Users/shmuelsan/Dropbox/HUJI/misc/twitter/jer_areas/density.html")


