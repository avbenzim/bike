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

# Network connection parameters (in meters)
# Gaps smaller than this will be connected
CONNECTION_TOLERANCE <- 50
# Lanes further than this from any other lane are considered truly isolated
ISOLATION_THRESHOLD <- 200

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


#' Connect nearby bike lane segments to create a more connected network
#'
#' This function finds "dangling" endpoints (endpoints not already touching
#' another lane) and connects them to their nearest neighbor within tolerance.
#'
#' @param bike_lanes sf object with bike lane line geometries
#' @param connection_tolerance Maximum distance (in CRS units) to connect gaps
#' @param isolation_threshold Distance beyond which a lane is considered truly isolated
#' @return sf object with additional connecting segments
connect_network_gaps <- function(bike_lanes,
                                  connection_tolerance = 50,
                                  isolation_threshold = 200) {
  message("Connecting network gaps (tolerance: ", connection_tolerance, "m)...")

  # Extract all endpoints from the bike lanes
  get_endpoints <- function(line) {
    coords <- st_coordinates(line)
    if (nrow(coords) < 2) return(NULL)

    start_pt <- st_point(coords[1, 1:2])
    end_pt <- st_point(coords[nrow(coords), 1:2])

    return(list(start = start_pt, end = end_pt))
  }

  # Collect all endpoints
  all_endpoints <- list()
  endpoint_info <- data.frame(
    lane_id = integer(),
    endpoint_type = character(),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(nrow(bike_lanes))) {
    eps <- get_endpoints(bike_lanes$geometry[i])
    if (!is.null(eps)) {
      all_endpoints[[length(all_endpoints) + 1]] <- eps$start
      endpoint_info <- rbind(endpoint_info, data.frame(
        lane_id = i, endpoint_type = "start", stringsAsFactors = FALSE
      ))

      all_endpoints[[length(all_endpoints) + 1]] <- eps$end
      endpoint_info <- rbind(endpoint_info, data.frame(
        lane_id = i, endpoint_type = "end", stringsAsFactors = FALSE
      ))
    }
  }

  n_endpoints <- length(all_endpoints)
  message("  Found ", n_endpoints, " endpoints from ", nrow(bike_lanes), " lane segments")

  # Create sf object of endpoints
  endpoints_sf <- st_sf(
    endpoint_info,
    geometry = st_sfc(all_endpoints, crs = st_crs(bike_lanes))
  )

  # Calculate distance matrix
  dist_matrix <- st_distance(endpoints_sf)

  # Identify "dangling" endpoints (not touching another lane within 1m)
  TOUCH_TOLERANCE <- 1.0
  is_connected <- rep(FALSE, n_endpoints)

  for (i in seq_len(n_endpoints)) {
    for (j in seq_len(n_endpoints)) {
      if (i == j) next
      if (endpoint_info$lane_id[i] == endpoint_info$lane_id[j]) next
      if (as.numeric(dist_matrix[i, j]) < TOUCH_TOLERANCE) {
        is_connected[i] <- TRUE
        break
      }
    }
  }

  n_dangling <- sum(!is_connected)
  message("  Found ", n_dangling, " dangling endpoints (not touching other lanes)")

  # For each dangling endpoint, find nearest neighbor from different lane
  connections_to_add <- list()
  already_connected <- rep(FALSE, n_endpoints)

  for (i in seq_len(n_endpoints)) {
    if (is_connected[i]) next  # Skip already connected
    if (already_connected[i]) next  # Skip if we made a connection for this

    # Find nearest endpoint from a different lane within tolerance
    best_j <- NA
    best_dist <- Inf

    for (j in seq_len(n_endpoints)) {
      if (i == j) next
      if (endpoint_info$lane_id[i] == endpoint_info$lane_id[j]) next

      dist <- as.numeric(dist_matrix[i, j])
      if (dist < 0.1 || dist > connection_tolerance) next

      # Prefer connecting to other dangling endpoints
      if (!is_connected[j] && !already_connected[j]) {
        if (dist < best_dist) {
          best_dist <- dist
          best_j <- j
        }
      }
    }

    # If no dangling neighbor, try any endpoint
    if (is.na(best_j)) {
      for (j in seq_len(n_endpoints)) {
        if (i == j) next
        if (endpoint_info$lane_id[i] == endpoint_info$lane_id[j]) next

        dist <- as.numeric(dist_matrix[i, j])
        if (dist < 0.1 || dist > connection_tolerance) next

        if (dist < best_dist) {
          best_dist <- dist
          best_j <- j
          break  # Take first valid one
        }
      }
    }

    if (!is.na(best_j)) {
      # Create connecting line
      pt1 <- all_endpoints[[i]]
      pt2 <- all_endpoints[[best_j]]

      connecting_line <- st_linestring(rbind(
        st_coordinates(pt1),
        st_coordinates(pt2)
      ))

      connections_to_add[[length(connections_to_add) + 1]] <- list(
        geometry = connecting_line,
        distance = best_dist,
        from_lane = endpoint_info$lane_id[i],
        to_lane = endpoint_info$lane_id[best_j]
      )

      already_connected[i] <- TRUE
      already_connected[best_j] <- TRUE
    }
  }

  n_connections <- length(connections_to_add)
  message("  Adding ", n_connections, " connecting segments")

  if (n_connections > 0) {
    # Create sf object for new connections
    new_connections <- st_sf(
      Name = paste0("Connection_", seq_len(n_connections)),
      Description = sapply(connections_to_add, function(x)
        paste0("Gap: ", round(x$distance, 1), "m")),
      geometry = st_sfc(
        lapply(connections_to_add, function(x) x$geometry),
        crs = st_crs(bike_lanes)
      )
    )

    # Add any missing columns to match bike_lanes structure
    for (col in setdiff(names(bike_lanes), names(new_connections))) {
      new_connections[[col]] <- NA
    }

    # Select only columns that exist in bike_lanes
    new_connections <- new_connections[, names(bike_lanes)]

    # Combine original lanes with new connections
    bike_lanes_connected <- rbind(bike_lanes, new_connections)

    message("  Network now has ", nrow(bike_lanes_connected), " segments (",
            nrow(bike_lanes), " original + ", n_connections, " connections)")
  } else {
    bike_lanes_connected <- bike_lanes
    message("  No gaps found within tolerance - network unchanged")
  }

  return(bike_lanes_connected)
}


#' Analyze network connectivity and identify isolated components
#' @param bike_lanes sf object with bike lane geometries
#' @param isolation_threshold Distance to consider a component isolated
#' @return List with connectivity statistics
analyze_connectivity <- function(bike_lanes, isolation_threshold = 200) {
  message("Analyzing network connectivity...")

  # Create a simple network
  net <- as_sfnetwork(bike_lanes, directed = FALSE)
  g <- as.igraph(net)

  # Find connected components
  components <- components(g)
  n_components <- components$no
  component_sizes <- table(components$membership)

  message("  Found ", n_components, " connected components")
  message("  Largest component: ", max(component_sizes), " nodes")
  message("  Components with >10 nodes: ", sum(component_sizes > 10))
  message("  Isolated segments (1-2 nodes): ", sum(component_sizes <= 2))

  return(list(
    n_components = n_components,
    component_sizes = component_sizes,
    membership = components$membership
  ))
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
#' @param bike_lanes_original Original bike lanes (before connections)
#' @param bike_lanes_connected Connected bike lanes (after filling gaps)
#' @param output_dir Output directory
create_visualization <- function(net, centers, areas,
                                  bike_lanes_original, bike_lanes_connected,
                                  output_dir) {
  message("Creating visualization...")

  # Identify which segments are connections (added segments)
  n_original <- nrow(bike_lanes_original)
  n_connected <- nrow(bike_lanes_connected)

  if (n_connected > n_original) {
    original_lanes <- bike_lanes_connected[1:n_original, ]
    connecting_segments <- bike_lanes_connected[(n_original + 1):n_connected, ]

    # Create plot showing connections
    p <- ggplot() +
      geom_sf(data = areas, fill = "lightgray", color = "white", alpha = 0.5) +
      geom_sf(data = original_lanes, color = "darkgreen", linewidth = 0.7) +
      geom_sf(data = connecting_segments, color = "red", linewidth = 1.2, linetype = "dashed") +
      geom_sf(data = centers, color = "blue", size = 1.5, alpha = 0.7) +
      theme_minimal() +
      labs(title = "Jerusalem Bike Network (Connected)",
           subtitle = "Green: Bike lanes | Red dashed: Gap connections | Blue: Area centers")
  } else {
    # No connections added
    p <- ggplot() +
      geom_sf(data = areas, fill = "lightgray", color = "white", alpha = 0.5) +
      geom_sf(data = bike_lanes_connected, color = "darkgreen", linewidth = 0.7) +
      geom_sf(data = centers, color = "blue", size = 1.5, alpha = 0.7) +
      theme_minimal() +
      labs(title = "Jerusalem Bike Network and Area Centers",
           subtitle = "Green: Bike lanes | Blue: Area centers")
  }

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

  # Analyze connectivity before connecting gaps
  message("\n--- Before connecting gaps ---")
  connectivity_before <- analyze_connectivity(bike_lanes, ISOLATION_THRESHOLD)

  # Connect nearby lane segments to improve network connectivity
  message("\n--- Connecting network gaps ---")
  bike_lanes_connected <- connect_network_gaps(
    bike_lanes,
    connection_tolerance = CONNECTION_TOLERANCE,
    isolation_threshold = ISOLATION_THRESHOLD
  )

  # Analyze connectivity after connecting gaps
  message("\n--- After connecting gaps ---")
  connectivity_after <- analyze_connectivity(bike_lanes_connected, ISOLATION_THRESHOLD)

  # Calculate area centers
  centers <- get_area_centers(areas, name_column = "STAT11_HEB")

  # Build network from connected lanes
  net <- build_network(bike_lanes_connected)

  # Calculate shortest paths
  results <- calculate_shortest_paths(net, centers)

  # Export results
  summary_stats <- export_results(results, net, centers, OUTPUT_DIR)

  # Create visualization
  tryCatch({
    create_visualization(net, centers, areas,
                        bike_lanes, bike_lanes_connected, OUTPUT_DIR)
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

  message("\nNetwork Connectivity Improvement:")
  message("  Components before: ", connectivity_before$n_components)
  message("  Components after:  ", connectivity_after$n_components)
  message("  Improvement: ", connectivity_before$n_components - connectivity_after$n_components,
          " components merged")

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
