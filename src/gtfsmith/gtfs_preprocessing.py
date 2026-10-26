import pandas as pd
from pathlib import Path

from .gtfs_class import GTFS
from . import gtfs_utils



def _remap_stop_references(gtfs: GTFS, mapping: dict) -> None:
    """Replace stop_ids according to `mapping` in every table that references stops.

    Works in place. Each table is popped, modified and put back, so no second
    copy of the feed is kept around. Transfers that become self-transfers
    (from == to) because of the remap are dropped, and exact duplicate
    transfers are removed.
    """
    # Columns that reference stop_ids, per GTFS table.
    _STOP_REFERENCES = {
        "stops.txt": ["parent_station"],
        "stop_times.txt": ["stop_id"],
        "transfers.txt": ["from_stop_id", "to_stop_id"],
        "pathways.txt": ["from_stop_id", "to_stop_id"],
    }


    for file_name, columns in _STOP_REFERENCES.items():
        table = gtfs.files.pop(file_name, None)
        if table is None:
            continue

        changed = pd.Series(False, index=table.index)

        for column in columns:
            if column not in table.columns:
                continue
            original = table[column]
            remapped = original.replace(mapping)
            changed |= remapped.ne(original).fillna(False)
            table[column] = remapped

        if file_name == "transfers.txt" and {"from_stop_id", "to_stop_id"} <= set(table.columns):
            self_loop = changed & (table["from_stop_id"] == table["to_stop_id"])
            table = table.loc[~self_loop].drop_duplicates().reset_index(drop=True)

        gtfs.files[file_name] = table


def merge_duplicated_physical_stops(
    gtfs: "GTFS | str | Path",
    radius_m: float = 300.0,
    output_path: str | Path | None = None,
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
            - references to the losing parent (stops, stop_times, transfers,
              pathways) are remapped.

    The GTFS object is modified in place and returned.
    """
    if not isinstance(gtfs, GTFS):
        gtfs = GTFS.from_zip(gtfs)

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
        if output_path is not None:
            gtfs.export_gtfs(output_path)

        return gtfs

    # losing parent -> winning parent
    parent_mapping = gtfs_utils._build_parent_merge_mapping(stops, nearby_parents)

    if not parent_mapping:
        if output_path is not None:
            gtfs.export_gtfs(output_path)
        
        return gtfs

    # Re-parent children and remap every reference to a losing parent.
    _remap_stop_references(gtfs, parent_mapping)

    # Re-read: the remap put a modified table back.
    stops = gtfs.files["stops.txt"]

    # Remove losing parent stations.
    location_type = pd.to_numeric(stops["location_type"], errors="coerce")
    remove_mask = (location_type == 1) & stops["stop_id"].isin(set(parent_mapping))
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

    if output_path is not None:
        gtfs.export_gtfs(output_path)

    return gtfs


def prefix_stop_ids(
    gtfs: "GTFS | str | Path",
    prefix: str,
    output_path: str | Path | None = None,
) -> "GTFS":
    """
    Prefix all stop IDs in a GTFS feed.

    All references to stop IDs in stops.txt, stop_times.txt,
    and transfers.txt are updated.

    Example:
        stop_id "123" -> "it:123"

    The GTFS object is modified in place and returned.
    """
    if not isinstance(gtfs, GTFS):
        gtfs = GTFS.from_zip(gtfs)

    files = gtfs.files

    prefix = str(prefix).rstrip(":") + ":"

    stops = files.pop("stops.txt")

    stop_id_mapping = {
        stop_id: f"{prefix}{stop_id}"
        for stop_id in stops["stop_id"].astype(str).unique()
    }

    stops["stop_id"] = stops["stop_id"].astype(str).replace(stop_id_mapping)

    if "parent_station" in stops.columns:
        stops["parent_station"] = (
            stops["parent_station"]
            .astype("string")
            .replace(stop_id_mapping)
        )

    # stop_times.txt
    stop_times = files.pop("stop_times.txt")

    stop_times["stop_id"] = stop_times["stop_id"].astype(str).replace(stop_id_mapping)

    # transfers.txt
    transfers = files.pop("transfers.txt", None)

    if transfers is not None:
        for column in ("from_stop_id", "to_stop_id"):
            if column in transfers.columns:
                transfers[column] = (
                    transfers[column]
                    .astype("string")
                    .replace(stop_id_mapping)
                )

        files["transfers.txt"] = transfers

    files["stops.txt"] = stops
    files["stop_times.txt"] = stop_times

    if output_path is not None:
        gtfs.export_gtfs(output_path)

    return gtfs


def create_parent_stations(gtfs: "GTFS | str | Path", output_file: str = None) -> "GTFS":
    """
    Some feeds (e.g. spanish, italian) do not work with the parent and child hierarchy,
    and have only one level. As many functions in this package rely on this hierarchy,
    this functions creates a parent station for each station in the feed.

    For every stop in a GTFS stops DataFrame:
      - Create a parent station with stop_id = 'Par' + child stop_id
      - Set the child's parent_station to the new parent
      - Set the child's location_type to 0
      - Set the parent's location_type to 1

    The GTFS object is modified in place and returned.
    """
    if not isinstance(gtfs, GTFS):
        gtfs = GTFS.from_zip(gtfs)

    stops = gtfs.files.pop("stops.txt")

    # Create parent stations from the existing stops
    parents = stops.copy()
    parents["stop_id"] = "Par" + parents["stop_id"].astype(str)
    parents["parent_station"] = ""
    parents["location_type"] = 1

    # Update the original stops to become children
    stops["parent_station"] = "Par" + stops["stop_id"].astype(str)
    stops["location_type"] = 0

    # Combine parents and children
    gtfs.files["stops.txt"] = pd.concat([parents, stops], ignore_index=True)

    if output_file is not None:
        gtfs.export_gtfs(output_file)

    return gtfs


def add_missing_parent_station(
    gtfs: "GTFS | str | Path",
    output_file: str = None,
) -> "GTFS":
    """
    Some feed (e.g. austrian) are missing parent stations for stops and only have the platforms.
    As many function in this package work on the parent level, parent stations are created
    for stops where they are missing, i.e. where parent_station = NA and location_type = NA

    The parent id is derived from the child id by dropping its last two ':'-separated
    segments. If the id has no such structure (derived id == child id, or it clashes
    with an existing non-station stop) the parent id is 'Par' + child id.

    For each group:
        - If a station with the derived id already exists, it is reused.
        - Otherwise a new parent station is created:
            - its stop_name is taken from the first child;
            - its latitude/longitude are the mean of the child coordinates;
            - location_type is set to 1.
        - Each child stop gets parent_station set to the parent ID.
        - Child location_type is set to 0.

    The GTFS object is modified in place and returned.
    """
    if not isinstance(gtfs, GTFS):
        gtfs = GTFS.from_zip(gtfs)

    if "stops.txt" not in gtfs.files:
        raise ValueError("Missing required file: stops.txt")

    stops = gtfs.files.pop("stops.txt")

    # Check whether the hierarchy columns exist.
    if "parent_station" not in stops.columns:
        stops["parent_station"] = pd.NA

    if "location_type" not in stops.columns:
        stops["location_type"] = pd.NA

    # Rows that are candidates for a newly created parent station.
    mask = stops["parent_station"].isna() & stops["location_type"].isna()

    children = stops.loc[mask].copy()

    if children.empty:
        gtfs.files["stops.txt"] = stops

        if output_file is not None:
            gtfs.export_gtfs(output_file)
        return gtfs

    child_ids = children["stop_id"].astype(str)
    existing_ids = set(stops["stop_id"].astype(str))
    station_ids = set(
        stops.loc[pd.to_numeric(stops["location_type"], errors="coerce") == 1, "stop_id"].astype(str)
    )

    # Derive parent IDs from child IDs.
    parent_ids = child_ids.str.rsplit(":", n=2).str[0]

    # No usable structure in the id (or it clashes with a non-station stop):
    # fall back to a prefixed id so we never duplicate a stop_id.
    clash = (parent_ids == child_ids) | parent_ids.isin(existing_ids - station_ids)
    parent_ids = parent_ids.where(~clash, "Par" + child_ids)

    children["parent_id"] = parent_ids

    # Only create parents that do not exist yet.
    to_create = children[~children["parent_id"].isin(station_ids)]

    parents = None
    if not to_create.empty:
        parents = (
            to_create
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

        # Make sure the new parent rows have the same columns as stops
        for column in stops.columns:
            if column not in parents.columns:
                parents[column] = pd.NA

        # Keep exactly the same column order
        parents = parents[stops.columns]

    # Children are regular stops
    stops.loc[children.index, "parent_station"] = children["parent_id"]
    stops.loc[children.index, "location_type"] = 0

    # Add parents to the feed
    if parents is not None:
        stops = pd.concat([stops, parents], ignore_index=True)

    gtfs.files["stops.txt"] = stops

    if output_file is not None:
        gtfs.export_gtfs(output_file)

    return gtfs


def normalize_swiss_routes(
    gtfs: "GTFS | str | Path",
    output_file: str = None,
) -> "GTFS":
    """
    Swiss GTFS feeds have a different classification of the routes types. To match the route types
    of the other GTFS feeds, these have to be normalized, using normalize_route_type.
    
    The GTFS object is modified in place and returned.
    """
    
    if not isinstance(gtfs, GTFS):
            gtfs = GTFS.from_zip(gtfs)

    routes = gtfs.files.pop("routes.txt")
    routes["route_type"] = routes["route_type"].apply(gtfs_utils.normalize_route_type)
    gtfs.files["routes.txt"] = routes

    if output_file is not None:
        gtfs.export_gtfs(output_file)

    return gtfs