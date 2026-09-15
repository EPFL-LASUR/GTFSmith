import pandas as pd
import geopandas as gpd
import gc

from .gtfs_class import GTFS
from . import gtfs_utils

def create_intercity_transfers(
        self: "GTFS",
        blob_buffer_km: float = 5.0,
        min_stations_per_blob: int = 3,
        max_dist_km: float = 10.0,
        mean_tp_speed: float = 20.0,
        interchange_penalty_min: float = 10.0
):
    """
    Some cities like Paris, Berlin or Milano have mulitple stations. These stations are usually
    connected by local public transportation. In order to not need to find the GTFS feeds of each
    local PT provider, this function creates transfers between stations close enough.
    
    A buffer of the specified radius (blob_buffer_km) is build around each station. The buffers are
    then merged (thus creating "blobs"), and transfers are created between the stations in the same blobs,
    given that there is at least the mininimum number of stations in the blob as specified, and that
    the distance between the two stations is less than the specified value.
    
    The transfer time (TT) is approximated by:
        TT = distance / mean_tp_speed + interchange_penalty
    """
    
    transfer_columns = [
        "from_stop_id",
        "to_stop_id",
        "transfer_type",
        "min_transfer_time",
    ]

    stops = self.files.get("stops.txt")

    # Get station parents
    stations = stops[stops["location_type"] == 1]

    # Get GeoDataFrame
    stations = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations["stop_lon"], stations["stop_lat"]), crs="EPSG:4326",)
    stations = stations.to_crs("EPSG:3035")

    # Build urban blobs
    buffers = gpd.GeoDataFrame(geometry=stations.geometry.buffer(blob_buffer_km * 1000), crs=stations.crs,)

    union = buffers.geometry.union_all()

    if union.geom_type == "Polygon":
        blob_geometries = [union]
    else:
        blob_geometries = list(union.geoms)

    blobs = gpd.GeoDataFrame(
        {"blob_id": range(len(blob_geometries))},
        geometry=blob_geometries,
        crs=stations.crs,
    )

    # Assign stations to blobs
    stations = gpd.sjoin(
        stations,
        blobs,
        how="inner",
        predicate="within",
    ).drop(columns="index_right")

    # Add a min of stations per blob to try and capture only urban areas
    blob_counts = stations["blob_id"].value_counts()
    urban_blobs = blob_counts[blob_counts >= min_stations_per_blob].index
    stations = stations[stations["blob_id"].isin(urban_blobs)]

    # Fin all stations pairs within max. distance
    max_distance_m = max_dist_km * 1000

    left_indices, right_indices = stations.sindex.query(
        stations.geometry,
        predicate="dwithin",
        distance=max_distance_m,
    )

    pairs = pd.DataFrame({
        "left_idx": left_indices,
        "right_idx": right_indices,
    })

    # Remove self-pairs
    pairs = pairs[pairs["left_idx"] != pairs["right_idx"]]

    # Keep only stations belonging to the same urban blob
    pairs = pairs[
        stations.iloc[pairs["left_idx"]]["blob_id"].to_numpy()
        ==
        stations.iloc[pairs["right_idx"]]["blob_id"].to_numpy()
    ]

    if pairs.empty:
        return pd.DataFrame(columns=transfer_columns)

    # Resolve station pairs
    left = stations.iloc[pairs["left_idx"]].reset_index(drop=True)
    right = stations.iloc[pairs["right_idx"]].reset_index(drop=True)

    # Calculate transfer time at parent station level
    distances_m = left.geometry.distance(right.geometry)

    transfer_times = (
        distances_m
        / 1000.0
        / mean_tp_speed
        * 3600.0
        + interchange_penalty_min * 60.0
    ).round().astype("int64")

    # Keep only transfer times up to 1 hour
    valid = transfer_times <= 3600

    left = left[valid].reset_index(drop=True)
    right = right[valid].reset_index(drop=True)
    transfer_times = transfer_times[valid].reset_index(drop=True)

    if left.empty:
        return pd.DataFrame(columns=transfer_columns)

    # Build parent -> children mapping
    #
    # _build_child_to_parent_mapping() gives:
    #           child_stop_id -> parent_station_id
    #
    # We invert it to obtain:
    #           parent_station_id -> [child_stop_id, ...]


    child_to_parent = gtfs_utils._build_child_to_parent_mapping(stops)

    children_by_parent = {}

    for stop_id, parent_id in child_to_parent.items():
        # Do not include the parent station itself as a child.
        if stop_id == parent_id:
            continue

        children_by_parent.setdefault(parent_id, []).append(stop_id)

    # Expand each station pair into child -> child transfers
    transfer_rows = []

    for i in range(len(left)):

        from_parent = left.loc[i, "stop_id"]
        to_parent = right.loc[i, "stop_id"]

        from_children = children_by_parent.get(from_parent, [])
        to_children = children_by_parent.get(to_parent, [])

        # No children for one of the stations: there is nothing routable to connect.
        if not from_children or not to_children:
            continue

        transfer_time = int(transfer_times.iloc[i])

        # Create transfer between every child of orig parent stop and every child of dest parent stop
        for from_child in from_children:
            for to_child in to_children:

                if from_child == to_child:
                    continue

                transfer_rows.append({
                    "from_stop_id": from_child,
                    "to_stop_id": to_child,
                    "transfer_type": 2,
                    "min_transfer_time": transfer_time,
                })

    # Create final dataframe
    if not transfer_rows:
        return pd.DataFrame(columns=transfer_columns)

    transfers = pd.DataFrame(transfer_rows, columns=transfer_columns,)
    transfers = transfers.drop_duplicates(subset=["from_stop_id", "to_stop_id",], keep="first",)

    return transfers


def merge_duplicated_physical_stops(
    gtfs,
    radius_m: float = 300.0,
):
    """
    Some feed (i.e. french feed) have different stop_ids and stop_names for physical same stations.
    Example:
        "Paris Montparnasse Hall 1 - 2" -- "Paris Montparnasse Vaugirard"
        "Mulhouse" -- "Mulhouse Gare Centrale"

    As the merging stop process supposes that each physical stop is represented only once
    per feed, it can be useful to merge these stops into one stop.

    For every group of nearby parent stations:
    
            - the parent appearing first in stops.txt wins;
            - the losing parent station is removed;
            - all children of the losing parent are re-parented to the
              winning parent;
            - child stop_ids are preserved;
            - references to the losing parent are remapped.
    """

    if "stops.txt" not in gtfs.files:
        raise ValueError("GTFS object does not contain stops.txt")

    stops = gtfs.files["stops.txt"]

    required_columns = {
        "stop_id",
        "location_type",
        "parent_station",
        "stop_lat",
        "stop_lon",
    }

    missing_columns = required_columns - set(stops.columns)

    if missing_columns:
        raise ValueError(f"stops.txt is missing required columns: {sorted(missing_columns)}")

    if radius_m <= 0:
        raise ValueError("radius_m must be greater than zero")

    # Find nearby parent stations.
    nearby_parents = gtfs_utils._find_nearby_parent_stations(stops, radius_m=radius_m)

    if nearby_parents.empty:
        return gtfs

    # Build:
    # losing parent -> winning parent
    parent_mapping = gtfs_utils._build_parent_merge_mapping(stops, nearby_parents)

    if not parent_mapping:
        return gtfs

    # Re-parent children and other references.
    gtfs = gtfs_utils._remap_parent_references(gtfs, parent_mapping)

    # Remove losing parent stations.
    location_type = pd.to_numeric(stops["location_type"], errors="coerce")

    losing_parent_ids = set(parent_mapping.keys())

    remove_mask = ((location_type == 1) & stops["stop_id"].isin(losing_parent_ids))

    stops = stops.loc[~remove_mask]

    
    # Safety check.
    duplicate_stop_ids = stops["stop_id"].duplicated(keep=False)

    if duplicate_stop_ids.any():
        duplicates = sorted(
            stops.loc[duplicate_stop_ids, "stop_id"]
            .astype(str)
            .unique()
        )

        raise ValueError(f"Duplicate stop_ids remain after merging duplicated physical stops: {duplicates[:20]}")

    gtfs.files["stops.txt"] = stops.reset_index(drop=True)

    return gtfs


def filter_gtfs_per_mode(gtfs: GTFS, route_type: int, output_path: str = None) -> GTFS:
    """
    Not all the route_types are needed. This function allows to filter a GTFS feed
    according to the desired route feed. All unnecessary trips, stops, stop_times,
    frequencies, transfers and dates are removed from the feed.
    
    This operation can make a big GTFS object (like the swiss GTFS) substantially lighter
    """

    files = gtfs.files

    routes = files.pop("routes.txt")
    filtered_routes = routes[routes["route_type"] == route_type]
    filtered_routes_ids = filtered_routes["route_id"].unique()
    del routes

    trips = files.pop("trips.txt")
    filtered_trips = trips[trips["route_id"].isin(filtered_routes_ids)]
    filtered_trips_ids = filtered_trips["trip_id"].unique()
    filtered_service_ids = filtered_trips["service_id"].unique()
    del trips

    stop_times = files.pop("stop_times.txt")
    filtered_stop_times = stop_times[stop_times["trip_id"].isin(filtered_trips_ids)]
    filtered_child_stop_ids = filtered_stop_times["stop_id"].unique()
    del stop_times
    gc.collect()  # release big unfiltered stop_times before continuing


    stops = files.pop("stops.txt")
    filtered_parent_stop_ids = (
        stops.loc[stops["stop_id"].isin(filtered_child_stop_ids), "parent_station"].unique()
    )
    filtered_stop_ids = pd.Index(filtered_child_stop_ids).union(filtered_parent_stop_ids)
    filtered_stops = stops[stops["stop_id"].isin(filtered_stop_ids)]
    del stops

    
    frequencies = files.pop("frequencies.txt", None)
    if frequencies is not None:
        filtered_frequencies = frequencies[frequencies["trip_id"].isin(filtered_trips_ids)]
        del frequencies
    else:
        filtered_frequencies = None

    
    transfers = files.pop("transfers.txt")
    filtered_transfers = transfers[
        gtfs_utils.valid_id(transfers["from_stop_id"], filtered_stop_ids)
        & gtfs_utils.valid_id(transfers["to_stop_id"], filtered_stop_ids)
        & gtfs_utils.valid_id(transfers["from_trip_id"], filtered_trips_ids)
        & gtfs_utils.valid_id(transfers["to_trip_id"], filtered_trips_ids)
        & gtfs_utils.valid_id(transfers["from_route_id"], filtered_routes_ids)
        & gtfs_utils.valid_id(transfers["to_route_id"], filtered_routes_ids)
    ]
    del transfers


    calendar = files.pop("calendar.txt", None)
    if calendar is not None:
        filtered_calendar = calendar[calendar["service_id"].isin(filtered_service_ids)]
        del calendar
    else:
        filtered_calendar = None

    calendar_dates = files.pop("calendar_dates.txt")
    filtered_calendar_dates = calendar_dates[calendar_dates["service_id"].isin(filtered_service_ids)]
    del calendar_dates

    gc.collect()

    filtered_gtfs = GTFS({
        "agency.txt": files.pop("agency.txt"),
        "routes.txt": filtered_routes,
        "stops.txt": filtered_stops,
        "stop_times.txt": filtered_stop_times,
        "trips.txt": filtered_trips,
        "calendar_dates.txt": filtered_calendar_dates,
        "transfers.txt": filtered_transfers,
    })

    if filtered_frequencies is not None:
        filtered_gtfs.files["frequencies.txt"] = filtered_frequencies

    if filtered_calendar is not None:
        filtered_gtfs.files["calendar.txt"] = filtered_calendar

    if output_path is not None:
        filtered_gtfs.export_gtfs(output_path)

    return filtered_gtfs

def filter_gtfs_per_bbox(gtfs: GTFS, bbox: tuple[float, float, float, float], output_path: str = None):
    """
    Allows to filter all the services present in a GTFS feed in a specific geographical area.

    All the stops in the bounding box are kept. stop_times.txt entries with the stops inside the bounding box are kept,
    as well as the corresonping trips, routes and services.
    
    Args:
        - gtfs (GTFS): gtfs feed
        - bbox (tuple[float, float, float, float]): limits of the bounding box (min_lon, min_lat, max_lon, max_lat).
        - output_path (str): Optional. Path to export the filtered GTFS feed.

    Returns:
        - filtered_gtfs (GTFS): GTFS feed filtered to a specific bounding box. Export it if output_path is specified.

    """

    files = gtfs.files
    min_lon, min_lat, max_lon, max_lat = bbox

    stops = files.pop("stops.txt")
    mask = (
        stops["stop_lon"].between(min_lon, max_lon)
        & stops["stop_lat"].between(min_lat, max_lat)
    )
    filtered_stops = stops.loc[mask]

    filtered_stop_ids = filtered_stops["stop_id"]
    if filtered_stop_ids.empty:
            raise ValueError("The geographical zone does not contain any GTFS stop.")

    del stops

    stop_times = files.pop("stop_times.txt")
    filtered_stop_times = stop_times[stop_times["stop_id"].isin(filtered_stop_ids)]
    filtered_trips_ids = filtered_stop_times["trip_id"].unique()
    del stop_times
    gc.collect()

    trips = files.pop("trips.txt")
    filtered_trips = trips[trips["trip_id"].isin(filtered_trips_ids)]
    filtered_route_ids = filtered_trips["route_id"].unique()
    filtered_service_ids = filtered_trips["service_id"].unique()
    del trips

    routes = files.pop("routes.txt")
    filtered_routes = routes[routes["route_id"].isin(filtered_route_ids)]
    filtered_routes_ids = filtered_routes["route_id"].unique()
    del routes
    gc.collect()

    frequencies = files.pop("frequencies.txt", None)
    if frequencies is not None:
        filtered_frequencies = frequencies[frequencies["trip_id"].isin(filtered_trips_ids)]
        del frequencies
    else:
        filtered_frequencies = None

    transfers = files.pop("transfers.txt")
    filtered_transfers = transfers[
        gtfs_utils.valid_id(transfers["from_stop_id"], filtered_stop_ids)
        & gtfs_utils.valid_id(transfers["to_stop_id"], filtered_stop_ids)
        & gtfs_utils.valid_id(transfers["from_trip_id"], filtered_trips_ids)
        & gtfs_utils.valid_id(transfers["to_trip_id"], filtered_trips_ids)
        & gtfs_utils.valid_id(transfers["from_route_id"], filtered_routes_ids)
        & gtfs_utils.valid_id(transfers["to_route_id"], filtered_routes_ids)
    ]
    del transfers


    calendar = files.pop("calendar.txt", None)
    if calendar is not None:
        filtered_calendar = calendar[calendar["service_id"].isin(filtered_service_ids)]
        del calendar
    else:
        filtered_calendar = None

    calendar_dates = files.pop("calendar_dates.txt")
    filtered_calendar_dates = calendar_dates[calendar_dates["service_id"].isin(filtered_service_ids)]
    del calendar_dates

    gc.collect()

    filtered_gtfs = GTFS({
        "agency.txt": files.pop("agency.txt"),
        "routes.txt": filtered_routes,
        "stops.txt": filtered_stops,
        "stop_times.txt": filtered_stop_times,
        "trips.txt": filtered_trips,
        "calendar_dates.txt": filtered_calendar_dates,
        "transfers.txt": filtered_transfers,
    })

    if filtered_frequencies is not None:
        filtered_gtfs.files["frequencies.txt"] = filtered_frequencies

    if filtered_calendar is not None:
        filtered_gtfs.files["calendar.txt"] = filtered_calendar

    if output_path is not None:
        filtered_gtfs.export_gtfs(output_path)

    return filtered_gtfs


def prefix_stop_ids(
    gtfs: "GTFS",
    prefix: str,
) -> "GTFS":
    """
    Some GTFS feed have the same structure for stop_ids. This function renames the stop_ids
    of a feed with a specified prefix to avoid collisions while merging feeds.
    
    All the references in the GTFS feed (stops.txt, stop_times.txt and transfers.txt) are updated.

    Example:
        stop_id "123" -> "it:123"

    Returns:
        A new GTFS object.
    """

    prefix = str(prefix).rstrip(":") + ":"

    stops = gtfs.files["stops.txt"].copy()
    stop_times = gtfs.files["stop_times.txt"].copy()

    # Original -> prefixed ID
    stop_ids = stops["stop_id"].astype(str)

    stop_id_mapping = {
        stop_id: f"{prefix}{stop_id}"
        for stop_id in stop_ids.unique()
    }

    # stops.txt
    stops["stop_id"] = stops["stop_id"].astype(str).replace(stop_id_mapping)

    if "parent_station" in stops.columns:
        stops["parent_station"] = stops["parent_station"].replace(stop_id_mapping)

    # stop_times.txt
    stop_times["stop_id"] = stop_times["stop_id"].astype(str).replace(stop_id_mapping)

    # transfers.txt
    files = {
        name: df.copy()
        for name, df in gtfs.files.items()
    }

    files["stops.txt"] = stops
    files["stop_times.txt"] = stop_times

    if "transfers.txt" in files:
        transfers = files["transfers.txt"]

        if "from_stop_id" in transfers.columns:
            transfers["from_stop_id"] = (
                transfers["from_stop_id"]
                .astype("string")
                .replace(stop_id_mapping)
            )

        if "to_stop_id" in transfers.columns:
            transfers["to_stop_id"] = (
                transfers["to_stop_id"]
                .astype("string")
                .replace(stop_id_mapping)
            )

    return GTFS(files)

def create_parent_stations(gtfs: "GTFS") -> "GTFS":
    """
    Some feeds (e.g. spanish, italian) do not work with the parent and child hierarchy,
    and have only one level. As many functions in this package rely on this hierarchy,
    this functions creates a parent station for each station in the feed.

    For every stop in a GTFS stops DataFrame:
      - Create a parent station with stop_id = 'Par:' + child stop_id
      - Set the child's parent_station to the new parent
      - Set the child's location_type to 0
      - Set the parent's location_type to 1

    Returns a new DataFrame containing the parent stations and
    the original child stops.
    """

    stops = gtfs.files.get("stops.txt")

    # Create parent stations from the existing stops
    parents = stops.copy()
    parents["stop_id"] = "Par" + parents["stop_id"].astype(str)
    parents["parent_station"] = ""
    parents["location_type"] = 1

    # Update the original stops to become children
    stops["parent_station"] = "Par" + stops["stop_id"].astype(str)
    stops["location_type"] = 0

    # Combine parents and children
    new_stops = pd.concat([parents, stops], ignore_index=True)

    gtfs.files["stops.txt"] = new_stops

    return gtfs

def add_missing_parent_station(gtfs: "GTFS") -> "GTFS":
    """
    Some feed (e.g. austrian )are missing parent stations for stops and only have the platforms.
    As many function in this package work on the parent level, parent stations are created
    for stops where they are missing, i.e. where parent_station = NA and location_type = NA

    For each group:
      - A new parent station is created.
      - Its stop_id is derived from the child stop_ids.
      - Its stop_name is taken from the child stops.
      - Its latitude/longitude are the mean of the child coordinates.
      - location_type is set to 1.
      - Each child stop gets parent_station set to the new parent ID.
      - Child location_type is set to 0.

    Parameters
    ----------
    stops : pd.DataFrame
        GTFS stops.txt DataFrame.

    Returns
    -------
    pd.DataFrame
        Updated stops DataFrame.
    """

    stops = gtfs.files["stops.txt"]

    # Rows that are candidates for a newly created parent station
    mask = stops["parent_station"].isna() & stops["location_type"].isna()

    children = stops.loc[mask]

    if children.empty:
        return stops

    # For Austrian IDs such as:
    # at:41:3351:0:1
    # at:41:3351:0:4
    #
    # derive:
    # at:41:3351
    children["parent_id"] = children["stop_id"].str.rsplit(":", n=2).str[0]

    # Create one parent for each group
    parents = (
        children
        .groupby("parent_id")
        .agg(
            stop_name=("stop_name", "first"),
            stop_lat=("stop_lat", "mean"),
            stop_lon=("stop_lon", "mean"),
        )
        .reset_index()
        .rename(columns={"parent_id": "stop_id"})
    )

    # Set parent-specific values
    parents["location_type"] = 1
    parents["parent_station"] = pd.NA

    # Children are regular stops
    stops.loc[children.index, "parent_station"] = children["parent_id"]
    stops.loc[children.index, "location_type"] = 0

    # Make sure the new parent rows have the same columns as stops
    for column in stops.columns:
        if column not in parents.columns:
            parents[column] = pd.NA

    # Keep exactly the same column order
    parents = parents[stops.columns]

    # Add parents to the feed
    stops = pd.concat([stops, parents], ignore_index=True)

    gtfs.files["stops.txt"] = stops

    return gtfs