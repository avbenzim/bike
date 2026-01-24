# =============================================================================
# Bike Path Shortest Route Calculator for Jerusalem
# =============================================================================
# This script calculates the shortest bike path between the centers of areas
# using bike lane network data.
#
# Input:
#   - Bike lanes KML file (line geometries representing bike paths)
#   - Areas shapefile (polygon geometries representing areas/neighborhoods)
#
# Output:
#   - Shortest paths between all pairs of area centers
#   - Distance matrix
#   - Path geometries as GeoJSON
# =============================================================================

# Load required libraries
library(sf)
library(sfnetworks)
library(tidygraph)
library(dplyr)
library(igraph)
library(tidyr)
library(ggplot2)

# Clear workspace
rm(list = ls())

# =============================================================================
# Configuration - Edit these paths as needed
# =============================================================================

# Get the script's directory (works when running interactively or via Rscript)
script_dir <- tryCatch({
  dirname(rstudioapi::getSourceEditorContext()$path)
}, error = function(e) {
  getwd()
})

# Input files (relative to script directory)
BIKE_LANES_FILE <- file.path(script_dir, "bike_lanes_completed.kml")
AREAS_FILE <- file.path(script_dir, "jer_areas.shp")

# Output directory
OUTPUT_DIR <- file.path(script_dir, "output")

# CRS for distance calculations (Israel TM Grid - meters)
TARGET_CRS <- 2039

# =============================================================================
# Functions
# =============================================================================

#' Load bike lanes from KML file
#' @param file_path Path to KML file
#' @return sf object with bike lane geometries
load_bike_lanes <- function(file_path) {
  message("Loading bike lanes from: ", file_path)

  bike_lanes <- st_read(file_path, quiet = TRUE)

  # Keep only line geometries
  geom_types <- st_geometry_type(bike_lanes)
  bike_lanes <- bike_lanes[geom_types %in% c("LINESTRING", "MULTILINESTRING"), ]

  if (nrow(bike_lanes) == 0) {
    stop("No valid line geometries found in bike lanes file")
  }

  # Convert MULTILINESTRING to LINESTRING if needed
  bike_lanes <- st_cast(bike_lanes, "LINESTRING", warn = FALSE)

  message("Loaded ", nrow(bike_lanes), " bike lane segments")
  return(bike_lanes)
}


#' Load areas from shapefile
#' @param file_path Path to shapefile
#' @param filter_column Column name for filtering (optional)
#' @param filter_value Value to filter by (optional)
#' @return sf object with area polygons
load_areas <- function(file_path, filter_column = NULL, filter_value = NULL) {
  message("Loading areas from: ", file_path)

  areas <- st_read(file_path, quiet = TRUE)

  # Keep only polygon geometries
  geom_types <- st_geometry_type(areas)
  areas <- areas[geom_types %in% c("POLYGON", "MULTIPOLYGON"), ]

  # Apply filter if specified
  if (!is.null(filter_column) && !is.null(filter_value)) {
    if (filter_column %in% names(areas)) {
      areas <- areas[areas[[filter_column]] == filter_value, ]
      message("Filtered to ", nrow(areas), " areas where ", filter_column, " = ", filter_value)
    }
  }

  if (nrow(areas) == 0) {
    stop("No valid polygon geometries found in areas file")
  }

  message("Loaded ", nrow(areas), " areas")
  return(areas)
}


#' Calculate centroids of areas
#' @param areas sf object with area polygons
#' @param name_column Column containing area names (optional)
#' @return sf object with area centroids
get_area_centers <- function(areas, name_column = NULL) {
  centers <- st_centroid(areas)

  # Create area IDs
  if (!is.null(name_column) && name_column %in% names(centers)) {
    centers$area_name <- centers[[name_column]]
  } else {
    centers$area_name <- paste0("Area_", seq_len(nrow(centers)))
  }

  # Add unique ID
  centers$area_id <- seq_len(nrow(centers))

  message("Calculated ", nrow(centers), " area centers")
  return(centers)
}


#' Build network graph from bike lanes
#' @param bike_lanes sf object with bike lane line geometries
#' @param tolerance Snapping tolerance in CRS units (meters)
#' @return sfnetwork object
build_network <- function(bike_lanes, tolerance = 1.0) {
  message("Building network graph...")

  # Create sfnetwork from bike lanes
  # The sfnetwork will automatically create nodes at line endpoints
  net <- as_sfnetwork(bike_lanes, directed = FALSE)

  # Clean the network:
  # 1. Subdivide edges at intersections
  # 2. Remove pseudo-nodes (nodes with degree 2)
  # 3. Simplify edges
  net <- net %>%
    activate("edges") %>%
    mutate(weight = edge_length()) %>%
    activate("nodes")

  # Try to subdivide edges at internal intersections
  tryCatch({
    net <- convert(net, to_spatial_subdivision, .clean = TRUE)
  }, error = function(e) {
    message("Note: Could not subdivide network - ", e$message)
  })

  # Get network stats
  n_nodes <- net %>% activate("nodes") %>% as_tibble() %>% nrow()
  n_edges <- net %>% activate("edges") %>% as_tibble() %>% nrow()

  message("Built network with ", n_nodes, " nodes and ", n_edges, " edges")

  return(net)
}


#' Find nearest network node to a point
#' @param point sf point geometry
#' @param net sfnetwork object
#' @return Node index
find_nearest_node <- function(point, net) {
  nodes <- net %>%
    activate("nodes") %>%
    st_as_sf()

  nearest_idx <- st_nearest_feature(point, nodes)
  return(nearest_idx)
}


#' Calculate shortest paths between all pairs of area centers
#' @param net sfnetwork object
#' @param centers sf object with area centroids
#' @return List with distance matrix and paths
calculate_shortest_paths <- function(net, centers) {
  message("Calculating shortest paths...")

  n_areas <- nrow(centers)
  area_names <- centers$area_name

  # Find nearest network node for each center
  center_nodes <- sapply(seq_len(n_areas), function(i) {
    find_nearest_node(centers[i, ], net)
  })

  # Initialize distance matrix
  distance_matrix <- matrix(Inf, nrow = n_areas, ncol = n_areas,
                           dimnames = list(area_names, area_names))
  diag(distance_matrix) <- 0

  # Store paths
  paths <- list()

  # Get edge weights
  net <- net %>%
    activate("edges") %>%
    mutate(weight = as.numeric(edge_length()))

  # Convert to igraph for shortest path calculation
  g <- as.igraph(net)

  # Calculate all pairs shortest paths
  total_pairs <- n_areas * (n_areas - 1) / 2
  pair_count <- 0

  for (i in seq_len(n_areas - 1)) {
    for (j in (i + 1):n_areas) {
      pair_count <- pair_count + 1

      node_i <- center_nodes[i]
      node_j <- center_nodes[j]

      # Calculate shortest path
      sp <- tryCatch({
        shortest_paths(g, from = node_i, to = node_j,
                      weights = E(g)$weight, output = "both")
      }, error = function(e) NULL)

      if (!is.null(sp) && length(sp$vpath[[1]]) > 0) {
        path_nodes <- as.integer(sp$vpath[[1]])

        # Calculate path length
        path_length <- 0
        if (length(path_nodes) > 1) {
          for (k in seq_len(length(path_nodes) - 1)) {
            edge_id <- get.edge.ids(g, c(path_nodes[k], path_nodes[k + 1]))
            if (edge_id > 0) {
              path_length <- path_length + E(g)$weight[edge_id]
            }
          }
        }

        distance_matrix[i, j] <- path_length
        distance_matrix[j, i] <- path_length

        # Store path
        path_key <- paste(area_names[i], area_names[j], sep = " -> ")
        paths[[path_key]] <- list(
          origin = area_names[i],
          destination = area_names[j],
          nodes = path_nodes,
          length = path_length
        )
      }

      # Progress update
      if (pair_count %% 100 == 0) {
        message("  Processed ", pair_count, "/", total_pairs, " pairs")
      }
    }
  }

  message("Calculated ", length(paths), " valid paths out of ", total_pairs, " pairs")

  return(list(
    distance_matrix = distance_matrix,
    paths = paths,
    center_nodes = center_nodes
  ))
}


#' Get path geometry from node sequence
#' @param net sfnetwork object
#' @param path_nodes Vector of node indices
#' @return sf linestring geometry
get_path_geometry <- function(net, path_nodes) {
  if (length(path_nodes) < 2) return(NULL)

  nodes_sf <- net %>%
    activate("nodes") %>%
    st_as_sf()

  # Get coordinates of path nodes
  path_coords <- st_coordinates(nodes_sf[path_nodes, ])

  # Create linestring
  if (nrow(path_coords) >= 2) {
    line <- st_linestring(path_coords[, 1:2])
    return(line)
  }

  return(NULL)
}


#' Export results to files
#' @param results List with distance matrix and paths
#' @param net sfnetwork object
#' @param centers sf object with area centroids
#' @param output_dir Output directory path
export_results <- function(results, net, centers, output_dir) {
  message("Exporting results to: ", output_dir)

  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # 1. Export distance matrix as CSV
  dist_df <- as.data.frame(results$distance_matrix)
  dist_df <- cbind(area = rownames(dist_df), dist_df)
  write.csv(dist_df, file.path(output_dir, "distance_matrix.csv"), row.names = FALSE)
  message("  Saved distance_matrix.csv")

  # 2. Export paths as GeoJSON
  path_geometries <- list()

  for (path_key in names(results$paths)) {
    path_info <- results$paths[[path_key]]
    geom <- get_path_geometry(net, path_info$nodes)

    if (!is.null(geom)) {
      path_geometries[[length(path_geometries) + 1]] <- list(
        geometry = geom,
        origin = path_info$origin,
        destination = path_info$destination,
        length_meters = path_info$length
      )
    }
  }

  if (length(path_geometries) > 0) {
    # Create sf object for paths
    paths_sf <- st_sf(
      origin = sapply(path_geometries, function(x) x$origin),
      destination = sapply(path_geometries, function(x) x$destination),
      length_meters = sapply(path_geometries, function(x) x$length),
      geometry = st_sfc(lapply(path_geometries, function(x) x$geometry),
                       crs = st_crs(centers))
    )

    st_write(paths_sf, file.path(output_dir, "shortest_paths.geojson"),
             driver = "GeoJSON", delete_dsn = TRUE, quiet = TRUE)
    message("  Saved shortest_paths.geojson")
  }

  # 3. Export area centers as GeoJSON
  centers_export <- centers[, c("area_name", "area_id")]
  st_write(centers_export, file.path(output_dir, "area_centers.geojson"),
           driver = "GeoJSON", delete_dsn = TRUE, quiet = TRUE)
  message("  Saved area_centers.geojson")

  # 4. Export summary statistics
  valid_distances <- results$distance_matrix[results$distance_matrix > 0 &
                                              results$distance_matrix < Inf]

  summary_stats <- list(
    total_areas = nrow(centers),
    total_path_pairs = length(results$paths),
    connected_pairs = length(valid_distances),
    average_path_length_m = round(mean(valid_distances), 2),
    min_path_length_m = round(min(valid_distances), 2),
    max_path_length_m = round(max(valid_distances), 2),
    median_path_length_m = round(median(valid_distances), 2)
  )

  writeLines(
    jsonlite::toJSON(summary_stats, pretty = TRUE, auto_unbox = TRUE),
    file.path(output_dir, "summary.json")
  )
  message("  Saved summary.json")

  return(summary_stats)
}


#' Create visualization of results
#' @param net sfnetwork object
#' @param centers sf object with area centroids
#' @param areas sf object with area polygons
#' @param output_dir Output directory
create_visualization <- function(net, centers, areas, output_dir) {
  message("Creating visualization...")

  # Get network edges as sf
  edges_sf <- net %>%
    activate("edges") %>%
    st_as_sf()

  # Create plot
  p <- ggplot() +
    geom_sf(data = areas, fill = "lightgray", color = "white", alpha = 0.5) +
    geom_sf(data = edges_sf, color = "blue", linewidth = 0.5) +
    geom_sf(data = centers, color = "red", size = 2) +
    theme_minimal() +
    labs(title = "Jerusalem Bike Network and Area Centers",
         subtitle = "Blue: Bike lanes | Red: Area centers")

  ggsave(file.path(output_dir, "network_map.png"), plot = p,
         width = 12, height = 10, dpi = 150)
  message("  Saved network_map.png")
}


# =============================================================================
# Main Execution
# =============================================================================

main <- function() {
  message("\n", paste(rep("=", 60), collapse = ""))
  message("BIKE PATH SHORTEST ROUTE CALCULATOR - JERUSALEM")
  message(paste(rep("=", 60), collapse = ""), "\n")

  # Load data
  bike_lanes <- load_bike_lanes(BIKE_LANES_FILE)
  areas <- load_areas(AREAS_FILE, filter_column = "in_jeru", filter_value = 1)

  # Transform to projected CRS for accurate distance calculations
  message("\nTransforming to CRS ", TARGET_CRS, " for distance calculations...")
  bike_lanes <- st_transform(bike_lanes, TARGET_CRS)
  areas <- st_transform(areas, TARGET_CRS)

  # Calculate area centers
  centers <- get_area_centers(areas, name_column = "STAT11_HEB")

  # Build network
  net <- build_network(bike_lanes)

  # Calculate shortest paths
  results <- calculate_shortest_paths(net, centers)

  # Export results
  summary_stats <- export_results(results, net, centers, OUTPUT_DIR)

  # Create visualization
  tryCatch({
    create_visualization(net, centers, areas, OUTPUT_DIR)
  }, error = function(e) {
    message("Could not create visualization: ", e$message)
  })

  # Print summary
  message("\n", paste(rep("=", 60), collapse = ""))
  message("SUMMARY")
  message(paste(rep("=", 60), collapse = ""))
  message("Areas processed: ", summary_stats$total_areas)
  message("Path pairs calculated: ", summary_stats$total_path_pairs)
  message("Connected pairs: ", summary_stats$connected_pairs)
  message("Average path length: ", round(summary_stats$average_path_length_m), " meters")
  message("Shortest path: ", round(summary_stats$min_path_length_m), " meters")
  message("Longest path: ", round(summary_stats$max_path_length_m), " meters")

  message("\nDistance Matrix (first 5x5):")
  print(results$distance_matrix[1:min(5, nrow(results$distance_matrix)),
                                1:min(5, ncol(results$distance_matrix))])

  message("\nResults saved to: ", OUTPUT_DIR)

  # Return results for further analysis if needed
  invisible(list(
    distance_matrix = results$distance_matrix,
    paths = results$paths,
    centers = centers,
    network = net
  ))
}

# Run if executed directly
if (!interactive() || TRUE) {
  results <- main()
}
