import pandas as pd
import geopandas as gpd
import gc
from pathlib import Path

from .gtfs_class import GTFS
from . import gtfs_utils

def create_intercity_transfers(
        gtfs: "GTFS | str | Path",
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

    if not isinstance(gtfs, GTFS):
        gtfs = GTFS.from_zip(gtfs)
    
    transfer_columns = [
        "from_stop_id",
        "to_stop_id",
        "transfer_type",
        "min_transfer_time",
    ]

    stops = gtfs.files.get("stops.txt")

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

######## ------------------ filters ------------------ ########


def filter_gtfs_per_mode(gtfs: "GTFS | str | Path", route_type: int, output_path: str = None) -> GTFS:
    """
    Not all the route_types are needed. This function allows to filter a GTFS feed
    according to the desired route type. All unnecessary trips, stops, stop_times,
    frequencies, transfers and dates are removed from the feed.

    This operation can make a big GTFS object (like the swiss GTFS) substantially lighter.

    The input feed is never modified. If a path is given, the feed is loaded here and
    its tables are released as soon as they have been filtered, to save memory.
    """
    consume = not isinstance(gtfs, GTFS)
    if consume:
        gtfs = GTFS.from_zip(gtfs)

    files = gtfs.files

    routes = gtfs_utils._read_table(files, "routes.txt", consume)
    filtered_routes = routes[routes["route_type"] == route_type]
    route_ids = pd.Index(filtered_routes["route_id"].unique())
    del routes

    trips = gtfs_utils._read_table(files, "trips.txt", consume)
    filtered_trips = trips[trips["route_id"].isin(route_ids)]
    trip_ids = pd.Index(filtered_trips["trip_id"].unique())
    service_ids = pd.Index(filtered_trips["service_id"].unique())
    del trips

    stop_times = gtfs_utils._read_table(files, "stop_times.txt", consume)
    filtered_stop_times = stop_times[stop_times["trip_id"].isin(trip_ids)]
    served_stop_ids = filtered_stop_times["stop_id"].unique()
    del stop_times
    if consume:
        gc.collect()  # release the big unfiltered stop_times before continuing

    stops = gtfs_utils._read_table(files, "stops.txt", consume)
    stop_ids = pd.Index(list(gtfs_utils._with_ancestors(stops, served_stop_ids)))
    filtered_stops = stops[stops["stop_id"].isin(stop_ids)]
    del stops

    frequencies = gtfs_utils._read_table(files, "frequencies.txt", consume)
    if frequencies is not None:
        frequencies = frequencies[frequencies["trip_id"].isin(trip_ids)]

    transfers = gtfs_utils._filter_transfers(
        gtfs_utils._read_table(files, "transfers.txt", consume), stop_ids, trip_ids, route_ids
    )
    calendar = gtfs_utils._filter_services(gtfs_utils._read_table(files, "calendar.txt", consume), service_ids)
    calendar_dates = gtfs_utils._filter_services(gtfs_utils._read_table(files, "calendar_dates.txt", consume), service_ids)

    agency = gtfs_utils._read_table(files, "agency.txt", consume)
    if agency is not None:
        agency = agency.copy()

    if consume:
        gc.collect()

    return gtfs_utils._build_filtered_gtfs({
        "agency.txt": agency,
        "routes.txt": filtered_routes,
        "stops.txt": filtered_stops,
        "stop_times.txt": filtered_stop_times,
        "trips.txt": filtered_trips,
        "calendar.txt": calendar,
        "calendar_dates.txt": calendar_dates,
        "transfers.txt": transfers,
        "frequencies.txt": frequencies,
    }, output_path)


def filter_gtfs_per_bbox(gtfs: "GTFS | str | Path", bbox: tuple[float, float, float, float], output_path: str = None):
    """
    Allows to filter all the services present in a GTFS feed in a specific geographical area.

    All the stops in the bounding box are kept, together with their parent stations
    (even if a parent lies outside the box, so that no parent_station reference is left
    dangling). stop_times.txt entries with the stops inside the bounding box are kept,
    as well as the corresponding trips, routes and services.

    Args:
        - gtfs (GTFS | str | Path): gtfs feed, or path to a GTFS zip
        - bbox (tuple[float, float, float, float]): limits of the bounding box
          (min_lon, min_lat, max_lon, max_lat). CRS is EPSG:4326.
        - output_path (str): Optional. Path to export the filtered GTFS feed.

    Returns:
        - filtered_gtfs (GTFS): GTFS feed filtered to a specific bounding box.
          Exported if output_path is specified.

    The input feed is never modified.
    """
    consume = not isinstance(gtfs, GTFS)
    if consume:
        gtfs = GTFS.from_zip(gtfs)

    files = gtfs.files
    min_lon, min_lat, max_lon, max_lat = bbox

    stops = gtfs_utils._read_table(files, "stops.txt", consume)
    in_box = (
        stops["stop_lon"].between(min_lon, max_lon)
        & stops["stop_lat"].between(min_lat, max_lat)
    )
    box_stop_ids = stops.loc[in_box, "stop_id"]
    if box_stop_ids.empty:
        raise ValueError("The geographical zone does not contain any GTFS stop.")

    # Stops in the box + their ancestors (which may lie outside the box)
    stop_ids = pd.Index(list(gtfs_utils._with_ancestors(stops, box_stop_ids)))
    filtered_stops = stops[stops["stop_id"].isin(stop_ids)]
    del stops

    stop_times = gtfs_utils._read_table(files, "stop_times.txt", consume)
    filtered_stop_times = stop_times[stop_times["stop_id"].isin(box_stop_ids)]
    trip_ids = pd.Index(filtered_stop_times["trip_id"].unique())
    del stop_times
    if consume:
        gc.collect()

    trips = gtfs_utils._read_table(files, "trips.txt", consume)
    filtered_trips = trips[trips["trip_id"].isin(trip_ids)]
    route_ids = pd.Index(filtered_trips["route_id"].unique())
    service_ids = pd.Index(filtered_trips["service_id"].unique())
    del trips

    routes = gtfs_utils._read_table(files, "routes.txt", consume)
    filtered_routes = routes[routes["route_id"].isin(route_ids)]
    del routes

    frequencies = gtfs_utils._read_table(files, "frequencies.txt", consume)
    if frequencies is not None:
        frequencies = frequencies[frequencies["trip_id"].isin(trip_ids)]

    transfers = gtfs_utils._filter_transfers(
        gtfs_utils._read_table(files, "transfers.txt", consume), stop_ids, trip_ids, route_ids
    )
    calendar = gtfs_utils._filter_services(gtfs_utils._read_table(files, "calendar.txt", consume), service_ids)
    calendar_dates = gtfs_utils._filter_services(gtfs_utils._read_table(files, "calendar_dates.txt", consume), service_ids)

    agency = gtfs_utils._read_table(files, "agency.txt", consume)
    if agency is not None:
        agency = agency.copy()

    if consume:
        gc.collect()

    return gtfs_utils._build_filtered_gtfs({
        "agency.txt": agency,
        "routes.txt": filtered_routes,
        "stops.txt": filtered_stops,
        "stop_times.txt": filtered_stop_times,
        "trips.txt": filtered_trips,
        "calendar.txt": calendar,
        "calendar_dates.txt": calendar_dates,
        "transfers.txt": transfers,
        "frequencies.txt": frequencies,
    }, output_path)