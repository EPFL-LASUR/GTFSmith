import pandas as pd
from pathlib import Path

from .gtfs_class import GTFS
from . import gtfs_utils


def merge_duplicated_physical_stops(
    gtfs: "GTFS | str | Path",
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

    Returns:
        The mutated GTFS object with new prefixes.
    """
    if not isinstance(gtfs, GTFS):
        gtfs = GTFS.from_zip(gtfs)

    files = gtfs.files

    prefix = str(prefix).rstrip(":") + ":"

    stops = files.pop("stops.txt")

    stop_ids = stops["stop_id"].astype(str)

    stop_id_mapping = {
        stop_id: f"{prefix}{stop_id}"
        for stop_id in stop_ids.unique()
    }

    stops["stop_id"] = (
        stops["stop_id"]
        .astype(str)
        .replace(stop_id_mapping)
    )

    if "parent_station" in stops.columns:
        stops["parent_station"] = (
            stops["parent_station"]
            .astype("string")
            .replace(stop_id_mapping)
        )

    # stop_times.txt
    stop_times = files.pop("stop_times.txt")

    stop_times["stop_id"] = (
        stop_times["stop_id"]
        .astype(str)
        .replace(stop_id_mapping)
    )

    # transfers.txt
    transfers = files.pop("transfers.txt", None)

    if transfers is not None:
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

        files["transfers.txt"] = transfers

    files["stops.txt"] = stops
    files["stop_times.txt"] = stop_times

    prefixed_gtfs = GTFS(files)

    if output_path is not None:
        prefixed_gtfs.export_gtfs(output_path)

    return prefixed_gtfs

def create_parent_stations(gtfs: "GTFS | str | Path", output_file: str = None) -> "GTFS":
    """
    Some feeds (e.g. spanish, italian) do not work with the parent and child hierarchy,
    and have only one level. As many functions in this package rely on this hierarchy,
    this functions creates a parent station for each station in the feed.

    For every stop in a GTFS stops DataFrame:
      - Create a parent station with stop_id = 'Par:' + child stop_id
      - Set the child's parent_station to the new parent
      - Set the child's location_type to 0
      - Set the parent's location_type to 1

    Returns the GTFS feed expanded with the new parent stations and
    the original child stops.
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
    new_stops = pd.concat([parents, stops], ignore_index=True)

    gtfs.files["stops.txt"] = new_stops

    if output_file is not None:
        gtfs.export_gtfs(output_file)

    return gtfs

def add_missing_parent_station(
    gtfs: "GTFS | str | Path",
    output_file: str = None,
) -> "GTFS":
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
    "GTFS" object
        Updated GTFS with the missing parent stations
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

    # Derive parent IDs from child IDs.
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

    if output_file is not None:
        gtfs.export_gtfs(output_file)

    return gtfs