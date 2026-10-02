import pandas as pd
import numpy as np
from pathlib import Path
import gc
from rapidfuzz.fuzz import ratio
from scipy.spatial import cKDTree
import warnings

from .gtfs_class import GTFS
from . import gtfs_utils
from .gtfs_schema import GTFS_Schema
from .gtfs_operations import create_intercity_transfers

######## --------------------- STOPS ----------------------- ############


def _stop_name_similarity(first: "GTFS", second: "GTFS") -> float:
    """
    Returns the ratio of similarity between two stops
    """

    first_normalized = gtfs_utils._normalize_stop_name(first)
    second_normalized = gtfs_utils._normalize_stop_name(second)

    if not first_normalized or not second_normalized:
        return 0.0

    return (
        ratio(first_normalized, second_normalized) / 100.0
    )


def _match_parent_stations(
        self: "GTFS",
        other: "GTFS",
        distance_threshold_m: float = 250.0
) -> dict[str, str]:
    """
    Match parent stations (location_type = 1) between two feeds.

    Returns:
        dict mapping second_stop_id -> first_stop_id for matched parent
        stations (empty dict if either feed has no parent stations, or if
        no matches were found).
    """

    # Get the parent stations
    first_stops = gtfs_utils._get_parent_stations(self.files["stops.txt"])
    second_stops = gtfs_utils._get_parent_stations(other.files["stops.txt"])

    # Prepare the coordinates
    first_stops = gtfs_utils._prepare_coordinates(first_stops)
    second_stops = gtfs_utils._prepare_coordinates(second_stops)

    if first_stops.empty or second_stops.empty:
        return {}

    # Convert coordinates to meters
    reference_lat = pd.concat([first_stops["stop_lat"], second_stops["stop_lat"]]).mean()

    first_xy = gtfs_utils._latlon_to_xy(first_stops, reference_lat)
    second_xy = gtfs_utils._latlon_to_xy(second_stops, reference_lat)

    # Spatial index over the first feed
    tree = cKDTree(first_xy)
    matches = []

    for second_position, second_xy_point in enumerate(second_xy):
        second_stop = second_stops.iloc[second_position]

        candidate_positions = tree.query_ball_point(second_xy_point, distance_threshold_m)

        if not candidate_positions:
            continue

        best_candidate = None
        best_score = -1.0

        for first_position in candidate_positions:
            first_stop = first_stops.iloc[first_position]

            # Euclidean distance in our local metric projection
            distance_m = float( ( (first_xy[first_position] - second_xy_point)**2 ).sum()**0.5 )
            name_similarity = _stop_name_similarity(first_stop["stop_name"], second_stop["stop_name"])


            # Distance score: 1.0 at zero meters; 0.0 at threshold distance
            distance_score = max(0.0, 1.0 - (distance_m / distance_threshold_m))

            # Name is deliberately more important
            score = 0.25 * name_similarity + 0.75 * distance_score

            if score > best_score:
                best_score = score
                best_candidate = {
                    "first_stop_id": first_stop["stop_id"],
                    "second_stop_id": second_stop["stop_id"],
                    "first_stop_name": first_stop["stop_name"],
                    "second_stop_name": second_stop["stop_name"],
                    "distance_m": distance_m,
                    "name_similarity": name_similarity,
                    "score": score
                }

        # Append the candidate to the other matches
        if best_candidate is not None:
            matches.append(best_candidate)

    result = pd.DataFrame(matches)

    if result.empty:
        return {}

    # Keep only the strongest match for each first station
    result = result.sort_values("score", ascending=False).drop_duplicates(subset=["first_stop_id"], keep="first")

    # Prevent one station from mapping to multiple first stations
    result = result.drop_duplicates(subset=["second_stop_id"], keep="first").sort_values("first_stop_id").reset_index(drop=True)

    if result.empty:
        return {}

    return dict(zip(result["second_stop_id"], result["first_stop_id"]))



def _match_child_stops(
    self: "GTFS",
    other: "GTFS",
    mapping_matched_parent: dict[str, str],
) -> dict[str, str]:
    """
    Match child stops between two feeds based on platform_code,
    but only for children belonging to already-matched parent stations.

    Args:
        self:
            First GTFS feed. Its stop IDs are authoritative.
        other:
            Second GTFS feed.
        mapping_matched_parent:
            Mapping of second parent station IDs to first parent station IDs.

    Returns:
        Mapping of second child stop IDs to first child stop IDs.
    """

    if not mapping_matched_parent:
        return {}

    first_stops = self.files["stops.txt"]
    second_stops = other.files["stops.txt"]

    # Both feeds need platform_code to match child stops.
    if (
        "platform_code" not in first_stops.columns
        or "platform_code" not in second_stops.columns
    ):
        return {}

    # Build child -> parent mappings for both feeds.
    first_child_to_parent = gtfs_utils._build_child_to_parent_mapping(
        first_stops
    )
    second_child_to_parent = gtfs_utils._build_child_to_parent_mapping(
        second_stops
    )

    # Get location types.
    first_location_type = pd.to_numeric(
        first_stops["location_type"],
        errors="coerce",
    )
    second_location_type = pd.to_numeric(
        second_stops["location_type"],
        errors="coerce",
    )

    # Keep only child stops.
    first_children = first_stops.loc[
        first_location_type != 1
    ].copy()

    second_children = second_stops.loc[
        second_location_type != 1
    ].copy()

    if first_children.empty or second_children.empty:
        return {}

    # Find the parent station of every child.
    first_children["_parent_station"] = (
        first_children["stop_id"].map(first_child_to_parent)
    )

    second_children["_parent_station"] = (
        second_children["stop_id"].map(second_child_to_parent)
    )

    # Normalize platform codes before matching.
    first_children["_platform_code"] = (
        first_children["platform_code"]
        .apply(gtfs_utils._normalize_platform_code)
    )

    second_children["_platform_code"] = (
        second_children["platform_code"]
        .apply(gtfs_utils._normalize_platform_code)
    )

    # Remove children for which no parent station could be resolved.
    first_children = first_children.dropna(
        subset=["_parent_station"]
    )

    second_children = second_children.dropna(
        subset=["_parent_station"]
    )

    first_children = first_children.dropna(subset=["_platform_code"])
    second_children = second_children.dropna(subset=["_platform_code"])

    if first_children.empty or second_children.empty:
        return {}

    # For every child in the second feed, determine which first-feed
    # parent station it corresponds to.
    #
    # mapping_matched_parent:
    #     second_parent_id -> first_parent_id
    second_children["_matched_first_parent"] = (
        second_children["_parent_station"].map(
            mapping_matched_parent
        )
    )

    # Keep only children whose parent station was matched.
    second_children = second_children.dropna(
        subset=["_matched_first_parent"]
    )

    if second_children.empty:
        return {}

    # Match children when:
    #   1. they belong to the same matched parent station
    #   2. they have the same normalized platform code
    candidates = first_children.merge(
        second_children,
        left_on=["_parent_station", "_platform_code"],
        right_on=[
            "_matched_first_parent",
            "_platform_code",
        ],
        suffixes=("_first", "_second"),
    )

    if candidates.empty:
        return {}

    # Build the match result.
    result = pd.DataFrame({
        "first_stop_id": candidates["stop_id_first"],
        "second_stop_id": candidates["stop_id_second"],
        "first_stop_name": candidates["stop_name_first"],
        "second_stop_name": candidates["stop_name_second"],
        "platform_code": candidates["_platform_code"],
    })

    # A first stop must not match multiple second stops.
    first_duplicated = result["first_stop_id"].duplicated(
        keep=False
    )

    # A second stop must not match multiple first stops.
    second_duplicated = result["second_stop_id"].duplicated(
        keep=False
    )

    if first_duplicated.any() or second_duplicated.any():
        ambiguous = sorted(
            set(
                result.loc[
                    first_duplicated,
                    "first_stop_id",
                ].astype(str)
            )
            |
            set(
                result.loc[
                    second_duplicated,
                    "second_stop_id",
                ].astype(str)
            )
        )

        raise ValueError(
            "Ambiguous platform_code match: the following "
            f"stop_ids matched more than one candidate: "
            f"{ambiguous[:20]}"
        )

    result = (
        result
        .sort_values(
            ["first_stop_id", "second_stop_id"]
        )
        .reset_index(drop=True)
    )

    # Return a flat mapping:
    #
    # second child ID -> first child ID
    return dict(
        zip(
            result["second_stop_id"],
            result["first_stop_id"],
        )
    )



def _merge_stop_hierarchy(
        self: "GTFS",
        other: "GTFS",
        parent_mapping: dict[str, str],
        child_mapping: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """
    Merge the stop hierarchies of two feeds.
    The first feed is authoritative for parent station and child stop IDs.

    Returns:
        merged_stops
        child_to_parent
        stop_id_mapping   -- second_stop_id -> first_stop_id, for every
                            matched station AND every matched child.
                            Callers use this to remap stop_times.txt
                            and transfers.txt in `other`.
    """

    child_mapping = child_mapping or {}

    first = self.files["stops.txt"].copy()
    second = other.files["stops.txt"].copy()

    # second_stop_id -> first_stop_id, covering both stations and children.
    stop_id_mapping = {**parent_mapping, **child_mapping}

    # Drop matched parent stations and matched children from the second feed
    matched_second_ids = set(stop_id_mapping.keys())
    second = second.loc[~second["stop_id"].isin(matched_second_ids)].copy()

    # Re-parent all surviving (unmatched) children of matched stations.
    if "parent_station" in second.columns:
        second["parent_station"] = second["parent_station"].replace(parent_mapping)

    merged = pd.concat([first, second], ignore_index=True, copy=False)

    duplicate_stop_ids = merged["stop_id"].duplicated(keep=False)
    if duplicate_stop_ids.any():
        duplicates = sorted(
            merged.loc[duplicate_stop_ids, "stop_id"].astype(str).unique()
        )
        raise ValueError(f"Duplicate stop_ids remain after stop merge: {duplicates[:20]}")

    child_to_parent = gtfs_utils._build_child_to_parent_mapping(merged)

    return merged, child_to_parent, stop_id_mapping


def merge_stops(
        self: "GTFS",
        other: "GTFS",
        distance_threshold_m: float = 250.0,
        return_matches: bool = False
):
    """
    Merge stops from another GTFS feed. self is authoritative.

    Parent stations (location_type == 1) are matched by geographic distance
    and fuzzy name similarity. Children of already-matched
    parent stations are then matched by exact platform_code.

    Args:
        - return_matches:
            If True, return:
                merged_stops, matched_parent, matched_children,
                child_to_parent, stop_id_mapping
            Otherwise, returns only:
                merged_stops
    """

    parent_mapping = _match_parent_stations(
        self,
        other,
        distance_threshold_m=distance_threshold_m
    )

    child_mapping = _match_child_stops(self, other, parent_mapping)

    merged_stops, child_to_parent, stop_id_mapping = _merge_stop_hierarchy(
        self, other, parent_mapping, child_mapping
    )

    if return_matches:
        return merged_stops, parent_mapping, child_mapping, child_to_parent, stop_id_mapping

    return merged_stops


######## --------------------- TRIPS ----------------------- ############


def frequencies_to_trips(gtfs: "GTFS") -> "GTFS":
    """
    Expand frequencies.txt entries with exact_times == 0 into explicit trips.
    
    The representative trip from trips.txt is removed and replaced by the generated trips.
    
    Args:
        - GTFS: feed with a frequencies.txt file
        
    Returns:
        - GTFS: feed with updated trips and stop_times dataframes
    
    """
    frequencies = gtfs.files.get("frequencies.txt")
    trips = gtfs.files["trips.txt"]
    stop_times = gtfs.files["stop_times.txt"]

    if frequencies is None:
        return gtfs

    if frequencies.empty:
        gtfs.files.pop("frequencies.txt", None)
        return gtfs


    if "exact_times" not in frequencies.columns:
        raise ValueError("frequencies.txt does not contain the required 'exact_times' column.")

    frequencies = frequencies[frequencies["exact_times"] == 0]

    if frequencies.empty:
        return gtfs

    # Give every frequency row a unique identifier
    frequencies = frequencies.reset_index(drop=True)
    frequencies["frequency_id"] = frequencies.index

    # Convert time to seconds
    frequencies["start_time_secs"] = frequencies["start_time"].apply(gtfs_utils.convert_to_seconds)
    frequencies["end_time_secs"] = frequencies["end_time"].apply(gtfs_utils.convert_to_seconds)

    # Validate headways and time ranges
    if (frequencies["headway_secs"] <= 0).any():
        raise ValueError("frequencies.txt contains a non positive headway_secs.")
    
    if (frequencies["end_time_secs"] <= frequencies["start_time_secs"]).any():
        raise ValueError("frequencies.txt contains an end_time that is not after start_time")

    # Calculate number of trips
    # Caution: the code has been implemented so it does not calculate a trip at the end_date of the frequency.
    # This might need to be modified depending on how the GTFS is build.
    duration = frequencies["end_time_secs"] - frequencies["start_time_secs"]

    frequencies["total_trips"] = ((duration - 1) // frequencies["headway_secs"]) + 1
    frequencies["total_trips"] = frequencies["total_trips"].astype(int)

    # Find representative trips
    representative_trips_ids = set(frequencies["trip_id"])
    representative_trips = trips[trips["trip_id"].isin(representative_trips_ids)].copy()

    missing_trips = representative_trips_ids - set(representative_trips["trip_id"])
    if missing_trips:
        raise ValueError(f"The following trip_ids appear in frequencies.txt but do not exist in trips.txt: {sorted(missing_trips)}")

    # Generate explicit trips
    generated_trip_rows = []
    for _, frequency in frequencies.iterrows():
        original_trip_id = frequency["trip_id"]
        frequency_id = frequency["frequency_id"]
        total_trips = int(frequency["total_trips"])

        representative = representative_trips[representative_trips["trip_id"] == original_trip_id]

        if len(representative) != 1:
            raise ValueError(f"Expected exactly one representative trip for trip_id {original_trip_id}, found {len(representative)}")

        representative = representative.iloc[0]
        for instance in range(total_trips):
            new_trip_id = f"{original_trip_id}_freq_{frequency_id}_{instance + 1}"

            row = representative.copy()
            row["trip_id"] = new_trip_id
            generated_trip_rows.append(row)

    trips_update = pd.DataFrame(generated_trip_rows)

    # Find representative stop_times
    frequency_stop_times = stop_times[stop_times["trip_id"].isin(representative_trips_ids)].copy()

    missing_stop_time_trips = representative_trips_ids - set(frequency_stop_times["trip_id"])
    if missing_stop_time_trips:
        raise ValueError(f"The follwoing trip_ids appear in frequencies.txt but have no stop_times: {sorted(missing_stop_time_trips)}")

    # Generate explicit stop_times
    generated_stop_time_rows = []

    for _, frequency in frequencies.iterrows():
        original_trip_id = frequency["trip_id"]
        frequency_id = frequency["frequency_id"]

        representative_stop_times = frequency_stop_times[frequency_stop_times["trip_id"] == original_trip_id].copy().sort_values("stop_sequence")

        if representative_stop_times.empty:
            continue

        # Convert representative arrival/departure time
        representative_stop_times["_arrival_secs"] = representative_stop_times["arrival_time"].apply(gtfs_utils.convert_to_seconds)
        representative_stop_times["_departure_secs"] = representative_stop_times["departure_time"].apply(gtfs_utils.convert_to_seconds)

        # The first departure is the reference point from which all stop-time offsets are calculated
        first_departure = representative_stop_times["_departure_secs"].iloc[0]

        representative_stop_times["_arrival_offset"] = representative_stop_times["_arrival_secs"] - first_departure
        representative_stop_times["_departure_offset"] = representative_stop_times["_departure_secs"] - first_departure

        # Generate every frequency instance
        for instance in range(int(frequency["total_trips"])):
            new_trip_id = f"{original_trip_id}_freq_{frequency_id}_{instance + 1}"
            shift = instance * frequency["headway_secs"]
            generated = representative_stop_times.copy()
            generated["trip_id"] = new_trip_id

            # Actual departure of this generated trip at the first stop
            new_start = frequency["start_time_secs"] + shift

            generated["_arrival_secs_new"] = new_start + generated["_arrival_offset"]
            generated["_departure_secs_new"] = new_start + generated["_departure_offset"]

            generated["arrival_time"] = generated["_arrival_secs_new"].apply(gtfs_utils.to_hhmmss)
            generated["departure_time"] = generated["_departure_secs_new"].apply(gtfs_utils.to_hhmmss)

            generated_stop_time_rows.append(generated)

    if generated_stop_time_rows:
        stop_times_update = pd.concat(generated_stop_time_rows, ignore_index=True)
    else:
        stop_times_update = pd.DataFrame(columns=stop_times.columns)

    # Remove helper columns
    helper_columns = ["_arrival_secs", "_departure_secs", "_arrival_offset",
                        "_departure_offset", "_arrival_secs_new", "_departure_secs_new"
                        ]

    stop_times_update = stop_times_update.drop(columns=[
        column for column in helper_columns if column in stop_times_update.columns
    ])

    # Remove representative trips
    trips = trips[~trips["trip_id"].isin(representative_trips_ids)].copy()
    stop_times = stop_times[~stop_times["trip_id"].isin(representative_trips_ids)].copy()

    # Restore order column
    trips_update = trips_update[trips.columns]
    stop_times_update = stop_times_update[stop_times.columns]

    # Add generated trips
    trips = pd.concat([trips, trips_update], ignore_index=True)
    stop_times = pd.concat([stop_times, stop_times_update], ignore_index=True)

    gtfs.files["trips.txt"] = trips
    gtfs.files["stop_times.txt"] = stop_times

    # Delete the frequencies.txt file from the GTFS object, as the information is now redundant
    gtfs.files.pop("frequencies.txt", None)

    return gtfs


def _trip_stop_signature(stop_times: pd.DataFrame, child_to_parent: dict[str,str]) -> pd.DataFrame:
    """
    Creates a hashed ordered stop signature for every trip, based on:
        - stop_id (mapped to parent station)
        - stop_sequence

    A shared hash is a necessary but not sufficient condition for two trips to match
    (extremely unlikely hash collision is possible) -> the true ordered sequence for surviving
    candidates needs to be verified. However, this saves RAM compared to the old version

    """

    # Only pull what's needed for the signature
    signature_input = stop_times[["trip_id", "stop_sequence", "stop_id"]]

    parent_station = signature_input["stop_id"].map(child_to_parent)
    missing_parent = parent_station.isna()
    if missing_parent.any():
        missing_stop_ids = sorted(
            signature_input.loc[missing_parent, "stop_id"].astype(str).unique()
        )
        raise ValueError(f"Some stop_ids in stop_times.txt could not be mapped to a parent station: {missing_stop_ids[:20]}")

    signature_input = signature_input.assign(parent_station=parent_station)
    signature_input = signature_input.sort_values(["trip_id", "stop_sequence"])

    def _hash_sequence(parents: pd.Series) -> int:
        joined = "\x1f".join(parents.astype(str))
        return hash(joined)

    signatures = (
        signature_input.groupby("trip_id", sort=False)["parent_station"]
        .agg(_hash_sequence)
        .reset_index()
        .rename(columns={"parent_station": "stop_signature"})
    )

    return signatures



def _match_trips(
    first_gtfs: "GTFS",
    second_gtfs: "GTFS",
    first_child_to_parent: dict[str, str],
    second_child_to_parent: dict[str, str],
    time_tolerance_sec: int = 60
) -> pd.DataFrame:
    """
    Finds trips that are present in both GTFS feeds.
    A trip is considered a match only if:
        1. The ordered stop_id sequence is identical
        2. The stop_sequence is identical
        3. Arrival and departure times differ by at most 'time_tolerance_sec' at every stop

    Returns:
        pd.DataFrame
            columns:
                first_trip_id
                second_trip_id
                stop_count
                max_arrival_difference
                max_departure_difference
    """

    first_stop_times = first_gtfs.files["stop_times.txt"]
    second_stop_times = second_gtfs.files["stop_times.txt"]

    first_signatures = _trip_stop_signature(first_stop_times, first_child_to_parent)
    second_signatures = _trip_stop_signature(second_stop_times, second_child_to_parent)

    empty_result = pd.DataFrame(columns=[
        "first_trip_id",
        "second_trip_id",
        "stop_count",
        "max_arrival_difference",
        "max_departure_difference",
    ])

    if first_signatures.empty or second_signatures.empty:
        return empty_result

    # Candidate pairs sharing a signature hash
    candidates = first_signatures.merge(
        second_signatures, on="stop_signature", how="inner", suffixes=("_first", "_second")
    )

    if candidates.empty:
        return empty_result

    needed_first = set(candidates["trip_id_first"])
    needed_second = set(candidates["trip_id_second"])

    # Restrict to the trips that are actually candidates instead of building
    # a DataFrame per trip for the whole feed (the old dict-of-groups
    # approach)
    first_subset = first_stop_times[first_stop_times["trip_id"].isin(needed_first)]
    second_subset = second_stop_times[second_stop_times["trip_id"].isin(needed_second)]

    first_subset = first_subset.assign(
        parent_station=first_subset["stop_id"].map(first_child_to_parent),
        _arr=gtfs_utils._vectorized_to_seconds(first_subset["arrival_time"]),
        _dep=gtfs_utils._vectorized_to_seconds(first_subset["departure_time"]),
    ).sort_values(["trip_id", "stop_sequence"])

    second_subset = second_subset.assign(
        parent_station=second_subset["stop_id"].map(second_child_to_parent),
        _arr=gtfs_utils._vectorized_to_seconds(second_subset["arrival_time"]),
        _dep=gtfs_utils._vectorized_to_seconds(second_subset["departure_time"]),
    ).sort_values(["trip_id", "stop_sequence"])

    first_groups = first_subset.groupby("trip_id", sort=False)[
        ["parent_station", "_arr", "_dep"]
    ].agg(list)
    second_groups = second_subset.groupby("trip_id", sort=False)[
        ["parent_station", "_arr", "_dep"]
    ].agg(list)

    matches = []
    for row in candidates.itertuples(index=False):
        first_trip_id = row.trip_id_first
        second_trip_id = row.trip_id_second

        first_row = first_groups.loc[first_trip_id]
        second_row = second_groups.loc[second_trip_id]

        # Guard against a hash collision from the signature merge above:
        # verify the true ordered stop sequence before trusting this pair.
        if first_row["parent_station"] != second_row["parent_station"]:
            continue

        arrival_differences = np.abs(
            np.asarray(first_row["_arr"]) - np.asarray(second_row["_arr"])
        )
        departure_differences = np.abs(
            np.asarray(first_row["_dep"]) - np.asarray(second_row["_dep"])
        )

        max_arrival_difference = int(arrival_differences.max())
        max_departure_difference = int(departure_differences.max())

        if max_arrival_difference > time_tolerance_sec:
            continue

        if max_departure_difference > time_tolerance_sec:
            continue

        matches.append({
            "first_trip_id": first_trip_id,
            "second_trip_id": second_trip_id,
            "stop_count": len(first_row["parent_station"]),
            "max_arrival_difference": max_arrival_difference,
            "max_departure_difference": max_departure_difference,
        })

    result = pd.DataFrame(matches)

    if result.empty:
        return empty_result

    if result["first_trip_id"].duplicated().any():
        raise ValueError("A trip from self matched multiple trips from other.")

    if result["second_trip_id"].duplicated().any():
        raise ValueError("A trip from other matched multiple trips from self.")

    return result.reset_index(drop=True)

def _merge_trip_data(
        self,
        other: "GTFS",
        matches: pd.DataFrame,
        stop_id_mapping: dict[str, str] | None = None,
        service_mapping: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge trips and stop_times from 'other' into 'self'. 'self' is authoritative for matched trips.

    stop_id_mapping (second_stop_id -> first_stop_id, from merge_stops) is
    applied to the stop_times of trips kept from 'other', since those trips
    may reference child stops (platforms) that were merged into self's
    stop_ids during merge_stops -- without this, stop_times would keep
    pointing at stop_ids that no longer exist in the merged stops.txt.
    """

    self_trips = self.files["trips.txt"]
    other_trips = other.files["trips.txt"]

    self_stop_times = self.files["stop_times.txt"]
    other_stop_times = other.files["stop_times.txt"]

    required_columns = {"first_trip_id", "second_trip_id"}
    missing = required_columns - set(matches.columns)
    if missing:
        raise ValueError(f"Trip matching dataframe is missing columns: {sorted(missing)}")

    if matches.empty:
        matched_other_trip_ids = set()

    else:
        if matches["first_trip_id"].duplicated().any():
                    raise ValueError("A trip from self matches multiple trips from other.")
        
        if matches["second_trip_id"].duplicated().any():
                    raise ValueError("A trip from other matches multiple trips from self.")
        
        matched_other_trip_ids = set(matches["second_trip_id"])

    # Find the unmatched trips and stop times
    other_unmatched_trips = other_trips[~other_trips["trip_id"].isin(matched_other_trip_ids)].copy()
    other_unmatched_stop_times = other_stop_times[
        ~other_stop_times["trip_id"].isin(matched_other_trip_ids)
    ].copy()

    # Replace the service IDs in the surviving trips
    if service_mapping:
        other_unmatched_trips["service_id"] = other_unmatched_trips["service_id"].replace(service_mapping)

    # Replace the stop IDs in the surviving stop times
    if stop_id_mapping:
        other_unmatched_stop_times["stop_id"] = other_unmatched_stop_times["stop_id"].replace(stop_id_mapping)
    
    # Merging trips and stop times
    merged_trips = pd.concat([self_trips, other_unmatched_trips], ignore_index=True)
    merged_stop_times = pd.concat([self_stop_times, other_unmatched_stop_times], ignore_index=True)

    return merged_trips, merged_stop_times



def merge_trips(self: "GTFS",
                other: "GTFS",
                trip_matches,
                stop_id_mapping: dict[str, str] | None = None,
                service_mapping: dict | None = None,
                return_matches: bool = False
    ):
    """
    Merge trips from another GTFS feed. The current object is authoritative.

    stop_id_mapping: full second->first stop_id mapping from merge_stops
    (stations + children), applied to other's surviving stop_times.
    """


    merged_trips, merged_stop_times = _merge_trip_data(self, other, trip_matches, stop_id_mapping=stop_id_mapping, service_mapping=service_mapping)

    if return_matches:
        return merged_trips, merged_stop_times, trip_matches

    return merged_trips, merged_stop_times


######## --------------------- ROUTES ----------------------- ############



def _match_routes(self: "GTFS", other: "GTFS", trip_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Identify routes that are present in both feeds based on previously matched trips.
    A route pair is considered a match if trips belonging to those routes have already been idenfified
    as identical
    """

    required_columns = {"first_trip_id", "second_trip_id"}
    missing = required_columns - set(trip_matches.columns)
    if missing:
        raise ValueError(f"trip_matches is missing columns: {sorted(missing)}")

    if trip_matches.empty:
        return pd.DataFrame(columns=[
            "first_route_id",
            "second_route_id",
            "matched_trip_count"
        ])

    first_trips = self.files["trips.txt"][
        ["trip_id", "route_id"]
    ].copy()

    second_trips = other.files["trips.txt"][
        ["trip_id", "route_id"]
    ].copy()

    # Associate each matched trip with its route
    matches = trip_matches[["first_trip_id", "second_trip_id"]].merge(
        first_trips, left_on="first_trip_id", right_on="trip_id", how="left"
    ).drop(columns="trip_id")

    matches = matches.merge(
        second_trips, left_on="second_trip_id", right_on="trip_id", how="left", suffixes=("_first", "_second")
    ).drop(columns="trip_id")

    # Check that every matched trip has a route
    if matches["route_id_first"].isna().any():
        raise ValueError("Some matched trips from self have no route_id")

    if matches["route_id_second"].isna().any():
        raise ValueError("Some matched trips from self have no route_id")

    # Count how many matched trips establish each route pair
    route_matches = matches.groupby(["route_id_first", "route_id_second"], as_index=False).agg(
        matched_trip_count=("first_trip_id", "count")
    ).rename(columns={
        "route_id_first": "first_route_id",
        "route_id_second": "second_route_id"
    })

    # A route should not map to several routes in the other feed
    first_ambiguous = route_matches[route_matches["first_route_id"].duplicated(keep=False)]
    second_ambiguous = route_matches[route_matches["second_route_id"].duplicated(keep=False)]

    if not first_ambiguous.empty:
        raise ValueError(f"A route from self maps to multiple routes in other: {first_ambiguous}")

    if not second_ambiguous.empty:
        raise ValueError(f"A route from other maps to multiple routes in self: {second_ambiguous}")

    return route_matches


def _merge_route_data(self: "GTFS", other: "GTFS", route_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Merge routes from 'other' into 'self'. 'self' is authoritative for matched routes.
    """

    self_routes = self.files["routes.txt"].copy()
    other_routes = other.files["routes.txt"].copy()

    # No matches
    if route_matches.empty:
        return pd.concat([self_routes, other_routes], ignore_index=True)

    required_columns = {"first_route_id", "second_route_id"}
    missing = required_columns - set(route_matches.columns)
    if missing:
        raise ValueError(f"routes_matches is missing columns: {sorted(missing)}")

    # Verify one-to-one mapping
    if route_matches["first_route_id"].duplicated().any():
        raise ValueError("A route from self matches multiple routes from other.")

    if route_matches["second_route_id"].duplicated().any():
        raise ValueError("A route from other matches multiple routes from self.")

    # Remove matched routes from other.routes.txt
    matched_other_route_ids = set(route_matches["second_route_id"])
    other_unmatched_routes = other_routes[~other_routes["route_id"].isin(matched_other_route_ids)].copy()

    # Combine routes
    merged_routes = pd.concat([self_routes, other_unmatched_routes], ignore_index=True)

    return merged_routes

def merge_routes(self: "GTFS", other: "GTFS", trip_matches: pd.DataFrame, return_matches: bool = False):
    """
    Merge routes from another GTFS feed. The current object is authoritative.
    """

    route_matches = _match_routes(self, other, trip_matches)

    merged_routes = _merge_route_data(self, other, route_matches)

    if return_matches:
        return merged_routes, route_matches

    return merged_routes


######## --------------------- SERVICES ----------------------- ############



def _service_dates(self: "GTFS") -> dict[str, set[str]]:
    """
    Return effective operating dates for every service_id
    Supports:
        - calendar.txt + calendar_dates.txt
        - calendar_dates.txt only
        
    """

    calendar = self.files.get("calendar.txt", pd.DataFrame())
    calendar_dates = self.files.get("calendar_dates.txt", pd.DataFrame())
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    services: dict[str, set[str]] = {}

    if not calendar.empty:
        cal = calendar.copy()
        cal["service_id"] = cal["service_id"].astype(str)
        cal["start_date"] = pd.to_datetime(cal["start_date"].astype(str), format="%Y%m%d")
        cal["end_date"] = pd.to_datetime(cal["end_date"].astype(str), format="%Y%m%d")
        flag_cols = cal[weekdays].to_numpy(dtype=int)

        for row, flags in zip(cal.itertuples(index=False), flag_cols):
            dr = pd.date_range(row.start_date, row.end_date, freq="D")
            mask = flags[dr.weekday] == 1
            services[row.service_id] = set(dr[mask].strftime("%Y%m%d"))

    if not calendar_dates.empty:
        cd = calendar_dates.copy()
        cd["service_id"] = cd["service_id"].astype(str)
        cd["date"] = cd["date"].astype(str)

        adds = cd[cd["exception_type"] == 1].groupby("service_id")["date"].apply(set)
        removes = cd[cd["exception_type"] == 2].groupby("service_id")["date"].apply(set)

        for service_id in set(adds.index) | set(removes.index):
            s = services.setdefault(service_id, set())
            s |= adds.get(service_id, set())
            s -= removes.get(service_id, set())

    return services


def _match_services(self: "GTFS", other: "GTFS", trip_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Match services between two feeds.
    A service pair is considered a match when:
        1. Their trips have been matched
        2. The effective operating dates are identical
        
    self is authoritative.
    """

    # Compare effective calendars
    self_dates = _service_dates(self)
    other_dates = _service_dates(other)

    empty_result = pd.DataFrame(columns=[
        "self_service_id",
        "other_service_id",
        "matched_trip_count"
    ])

    if trip_matches.empty:
        return empty_result, self_dates, other_dates

    required_columns = {"first_trip_id", "second_trip_id"}
    missing = required_columns - set(trip_matches.columns)
    if missing:
        raise ValueError(f"trip_matches is missing columns: {sorted(missing)}")

    self_trips = self.files["trips.txt"][["trip_id", "service_id"]].copy()
    other_trips = other.files["trips.txt"][["trip_id", "service_id"]].copy()

    matches = trip_matches[["first_trip_id", "second_trip_id"]].merge(
        self_trips, left_on="first_trip_id", right_on="trip_id", how="left"
    ).drop(columns="trip_id").merge(
        other_trips, left_on="second_trip_id", right_on="trip_id", how="left", suffixes=("_self", "_other")
    ).drop(columns="trip_id")

    if matches["service_id_self"].isna().any():
        raise ValueError("A matched trip from self has no service_id")

    if matches["service_id_other"].isna().any():
        raise ValueError("A matched trip from other has no service_id")

    candidates = matches.groupby(["service_id_self", "service_id_other"], as_index=False).agg(
        matched_trip_count=("first_trip_id", "count")
    )

    candidates["same_dates"] = candidates.apply(
        lambda row:
            self_dates.get(str(row["service_id_self"]), set())
            ==
            other_dates.get(str(row["service_id_other"]), set()),
            axis=1
    )

    candidates = candidates[candidates["same_dates"]].drop(columns="same_dates")

    # One-to-one constraint
    if candidates["service_id_self"].duplicated().any():
        raise ValueError("A service from self matches multiple services from other.")
    if candidates["service_id_other"].duplicated().any():
        raise ValueError("A service from other matches multiple services from self.")

    result = candidates.rename(columns={
        "service_id_self": "self_service_id",
        "service_id_other": "other_service_id"
    }
    )

    return result, self_dates, other_dates

def _service_mapping(self: "GTFS", service_matches: pd.DataFrame) -> dict[str, str]:
    """
    Return:
        other_service_id -> self_service_id
        
    for matched services

    Kept in the ORIGINAL dtype of trips.txt["service_id"], because it is used
    directly against the merged trips.txt column via .replace().
    """

    if service_matches.empty:
        return {}

    return dict(zip(
        service_matches["other_service_id"],
        service_matches["self_service_id"]
    ))



def _dates_to_calendar(service_id: str, dates: set[str]) -> tuple[dict, list[dict]]:
    """
    Convert an explicit set of operating dates into:
        calendar row
        calendar_dates exception row
        
    The generated calendar uses a weekday as a base service only when the service operates
    on every occurence of that weekday within the date range
    """

    if not dates:
        raise ValueError(f"Service {service_id} has no operating dates")

    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    timestamps = pd.to_datetime(sorted(dates), format="%Y%m%d")
    start_date, end_date = timestamps.min(), timestamps.max()

    operating_counts = np.bincount(timestamps.weekday, minlength=7)
    full_range = pd.date_range(start_date, end_date, freq="D")
    total_counts = np.bincount(full_range.weekday, minlength=7)

    weekly_flags = ((total_counts > 0) & (operating_counts == total_counts)).astype(int)
    weekly_service = dict(zip(weekdays, weekly_flags))

    calendar_row = {
        "service_id": service_id,
        **weekly_service,
        "start_date": int(start_date.strftime("%Y%m%d")),
        "end_date": int(end_date.strftime("%Y%m%d")),
    }

    full_dates_str = full_range.strftime("%Y%m%d")
    operates = full_dates_str.isin(dates)                       # vectorized set lookup
    normally_operates = weekly_flags[full_range.weekday] == 1

    added = full_dates_str[operates & ~normally_operates]
    removed = full_dates_str[~operates & normally_operates]

    exceptions_df = pd.concat([
        pd.DataFrame({"service_id": service_id, "date": added, "exception_type": 1}),
        pd.DataFrame({"service_id": service_id, "date": removed, "exception_type": 2}),
    ], ignore_index=True)

    return calendar_row, exceptions_df.to_dict("records")



def merge_services(
    self_dates: dict[str, set[str]],
    other_dates: dict[str, set[str]],
    merged_trips: pd.DataFrame,
    service_mapping: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge service IDs and produce the final:
        - trips
        - calendar
        - calendar_dates

    self is authoritative for matched services.
    """

    # String-keyed copy used only for comparisons against _service_dates()
    # output (which always stringifies service_id). Kept separate from
    # service_mapping because that dict must stay in trips.txt's native
    # dtype to match correctly inside .replace() below.
    service_mapping_str = {str(k): str(v) for k, v in service_mapping.items()}

    # Services that actually occur in the final trips table
    merged_service_ids = set(
        merged_trips["service_id"].astype(str)
    )


    final_service_dates = {}

    # Add services originating from self
    for service_id, dates in self_dates.items():

        service_id = str(service_id)

        if service_id in merged_service_ids:
            final_service_dates[service_id] = set(dates)

    # Add services originating from other
    #
    # If an other-service is matched to a self-service, the
    # self-service's dates are authoritative.
    # Otherwise, keep the other service and its dates.
    for other_service_id, dates in other_dates.items():


        final_service_id = service_mapping_str.get(
            other_service_id,
            other_service_id
        )

        # If this service does not occur in the final trips, there is no need to create a calendar entry for it.
        if final_service_id not in merged_service_ids:
            continue

        # Matched services are already represented by the authoritative self service.
        if other_service_id in service_mapping_str:
            continue

        # Unmatched service from other survives with its original service_id and dates.
        final_service_dates[final_service_id] = set(dates)

    # Safety check
    missing_service_dates = (
        merged_service_ids - set(final_service_dates)
    )

    if missing_service_dates:
        raise ValueError(
            "Could not determine operating dates for service IDs: "
            f"{sorted(missing_service_dates)[:20]}"
        )

    # Convert effective dates into calendar + exceptions
    calendar_rows = []
    calendar_dates_rows = []

    for service_id, dates in final_service_dates.items():

        calendar_row, exception_rows = _dates_to_calendar(service_id, dates)

        calendar_rows.append(calendar_row)
        calendar_dates_rows.extend(exception_rows)

    merged_calendar = pd.DataFrame(calendar_rows)

    merged_calendar_dates = pd.DataFrame(
        calendar_dates_rows,
        columns=[
            "service_id",
            "date",
            "exception_type"
        ]
    )

    return merged_calendar, merged_calendar_dates


######## --------------------- AGENCIES ----------------------- ############


def _match_agencies(self: "GTFS", other: "GTFS", route_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Identify agencies that correspond through matched routes.
    A pair of agencies is considered a match when routes belonging to those agencies have already been matched.
    """
    required_columns = {"first_route_id", "second_route_id"}
    missing = required_columns - set(route_matches.columns)
    if missing:
        raise ValueError(f"route_matches is missing columns: {sorted(missing)}")

    if route_matches.empty:
        return pd.DataFrame(columns=[
            "first_agency_id",
            "second_agency_id",
            "matched_route_count"
        ])

    first_routes = self.files["routes.txt"][["route_id", "agency_id"]].copy()
    second_routes = other.files["routes.txt"][["route_id", "agency_id"]].copy()

    # Associate each matched route with its agency
    matches = route_matches[["first_route_id", "second_route_id"]].merge(
        first_routes, left_on="first_route_id", right_on="route_id", how="left"
    ).drop(columns="route_id")

    matches = matches.merge(second_routes, left_on="second_route_id", right_on="route_id", how="left", suffixes=("_first", "_second")).drop(columns="route_id")

    if matches["agency_id_first"].isna().any():
        raise ValueError("Some matches routes from self have no agency_id")
    if matches["agency_id_second"].isna().any():
                raise ValueError("Some matches routes from other have no agency_id")

    agency_matches = matches.groupby(["agency_id_first", "agency_id_second"], as_index=False).agg(
        matched_route_count=("first_route_id", "count")
    ).rename(columns={
        "agency_id_first": "first_agency_id",
        "agency_id_second": "second_agency_id"
    })

    first_ambiguous = agency_matches[agency_matches["first_agency_id"].duplicated(keep=False)]
    second_ambiguous = agency_matches[agency_matches["second_agency_id"].duplicated(keep=False)]

    if not first_ambiguous.empty:
        raise ValueError(f"An agency from self maps to multiple agencies from other: {first_ambiguous}")
    if not second_ambiguous.empty:
        raise ValueError(f"An agency from other maps to multiple agencies from self: {second_ambiguous}")

    return agency_matches.reset_index(drop=True)


def _merge_agency_data(
    self: "GTFS",
    other: "GTFS",
    merged_routes: pd.DataFrame,
    agency_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    self_agencies = self.files["agency.txt"]
    other_agencies = other.files["agency.txt"]

    required_columns = {
        "first_agency_id",
        "second_agency_id",
    }

    missing = required_columns - set(agency_matches.columns)

    if missing:
        raise ValueError(f"agency_matches is missing columns: {sorted(missing)}")

    # Build agency mapping
    agency_mapping = dict(zip(agency_matches["second_agency_id"], agency_matches["first_agency_id"]))
    matched_other_agency_ids = set(agency_matches["second_agency_id"])

    # Find unmatched agencies
    other_unmatched_agencies = other_agencies[~other_agencies["agency_id"].isin(matched_other_agency_ids)].copy()

    self_agency_ids = set(self_agencies["agency_id"].astype(str))

    used_agency_ids = set(self_agency_ids)


    # Rename conflicting IDs from other
    for idx, row in other_unmatched_agencies.iterrows():

        old_id = str(row["agency_id"])

        if old_id not in used_agency_ids:
            used_agency_ids.add(old_id)
            continue

        base_id = f"other_{old_id}"
        new_id = base_id
        counter = 2

        while new_id in used_agency_ids:
            new_id = f"{base_id}_{counter}"
            counter += 1

        other_unmatched_agencies.at[idx, "agency_id"] = new_id

        agency_mapping[old_id] = new_id

        used_agency_ids.add(new_id)


    # Remap agency_id on the already-merged routes table, but only on the
    # rows that came from `other`
    other_route_ids = set(other.files["routes.txt"]["route_id"].astype(str))
    is_other_route = merged_routes["route_id"].astype(str).isin(other_route_ids)

    merged_routes = merged_routes.copy()
    merged_routes.loc[is_other_route, "agency_id"] = (
        merged_routes.loc[is_other_route, "agency_id"].astype(str).replace(agency_mapping)
    )

    # Merge agencies
    merged_agency = pd.concat([self_agencies, other_unmatched_agencies], ignore_index=True)

    # Final validation
    duplicate_agency_ids = merged_agency[merged_agency["agency_id"].duplicated(keep=False)]

    if not duplicate_agency_ids.empty:
        duplicates = sorted(
            duplicate_agency_ids["agency_id"]
            .astype(str)
            .unique()
        )

        raise ValueError(f"Duplicate agency_ids remain after agency merge: {duplicates[:20]}")

    return merged_routes, merged_agency


def merge_agencies(self: "GTFS", other: "GTFS", merged_routes: pd.DataFrame, route_matches: pd.DataFrame, return_matches: bool = False):
    """
    Merge agencies from another GTFS feed.
    Agencies are matched through previously matched routes. self is authoritative.
    """
    agency_matches = _match_agencies(self, other, route_matches)

    merged_routes, merged_agency = _merge_agency_data(self, other, merged_routes, agency_matches)

    if return_matches:
        return merged_routes, merged_agency, agency_matches

    return merged_routes, merged_agency 


######## --------------------- TRANSFERS ----------------------- ############


def merge_transfers(
    self: "GTFS",
    other: "GTFS",
    parent_mapping: dict[str, str],
    other_child_to_parent: dict[str, str],
    child_mapping: dict[str, str] | None = None,
    trip_matches: pd.DataFrame | None = None,
    return_matches: bool = False,
):
    """
    child_matches: matches from merge_stops(..., return_matches=True) for
    CHILD stops (platforms etc.). Needed so that:
    1. other's transfers referencing a merged child get rewritten onto
        self's child stop_id
    2. propagation in step 3 only clones self's transfer pattern onto
        children that are genuinely NEW (i.e. weren't matched into an
        existing self child) -- a matched child already IS a self stop_id,
        so it's already covered by self's own transfer rows.
    """

    self_raw = self.files.get("transfers.txt")
    other_raw = other.files.get("transfers.txt")

    if self_raw is None and other_raw is None:
        return (None, None) if return_matches else None

    self_transfers = gtfs_utils._normalize_transfers(self_raw)
    other_transfers = gtfs_utils._normalize_transfers(other_raw)

    child_mapping = child_mapping or {}
    stop_id_mapping = {**parent_mapping, **child_mapping}

    trip_id_mapping = {}
    if trip_matches is not None and not trip_matches.empty:
        trip_id_mapping = dict(zip(trip_matches["second_trip_id"], trip_matches["first_trip_id"]))

    # Remap other's transfers onto the merged stop hierarchy
    # Covers both matched parent and children stops
    other_transfers["from_stop_id"] = other_transfers["from_stop_id"].replace(stop_id_mapping)
    other_transfers["to_stop_id"] = other_transfers["to_stop_id"].replace(stop_id_mapping)

    if trip_id_mapping:
        other_transfers["from_trip_id"] = other_transfers["from_trip_id"].replace(trip_id_mapping)
        other_transfers["to_trip_id"] = other_transfers["to_trip_id"].replace(trip_id_mapping)

    # Split generic (station-level) vs trip-specific rows 
    is_trip_specific = self_transfers["transfer_type"].isin({"4", "5"})
    self_generic = self_transfers.loc[~is_trip_specific]

    # Propagate self's existing generic relationships onto new
    # children only (matched children are not new -- they already
    # exist as a self stop_id and are already covered by self's
    # own rows)
    propagated = pd.DataFrame(columns=GTFS_Schema.transfers_columns)

    if parent_mapping and not self_generic.empty:
        self_child_to_parent = gtfs_utils._build_child_to_parent_mapping(self.files["stops.txt"])


        matched_other_child_ids = set(child_mapping.keys())

        new_children_by_station: dict[str, list[str]] = {}

        for other_station_id, self_station_id in parent_mapping.items():

            children = [
                stop_id
                for stop_id, parent_id in other_child_to_parent.items()
                if parent_id == other_station_id
                and stop_id not in matched_other_child_ids
                and stop_id != other_station_id
            ]

            if children:
                new_children_by_station.setdefault(
                    self_station_id,
                    []
                ).extend(children)

        if new_children_by_station:
            augmented = self_generic.assign(
                _from_parent=self_generic["from_stop_id"].map(self_child_to_parent),
                _to_parent=self_generic["to_stop_id"].map(self_child_to_parent),
            )

            new_children_df = pd.DataFrame(
                [(station, child) for station, children in new_children_by_station.items() for child in children],
                columns=["_station_id", "_child_stop_id"],
            )

            # Inter-station propagation (Cornavin <-> Genève) 
            # Only where the other side belongs to a different station

            # same-station rows are handled separately later, so they
            # aren't double-cloned here.
            inter = augmented[augmented["_from_parent"] != augmented["_to_parent"]]

            from_side = inter.merge(new_children_df, left_on="_from_parent", right_on="_station_id")
            from_side = from_side.assign(from_stop_id=from_side["_child_stop_id"])

            to_side = inter.merge(new_children_df, left_on="_to_parent", right_on="_station_id")
            to_side = to_side.assign(to_stop_id=to_side["_child_stop_id"])

            inter_propagated = pd.concat([from_side, to_side], ignore_index=True)

            # Internal-station propagation (platform <-> platform within the same merged station)
            
            # Existing internal rows describe the station's own transfer
            # pattern (e.g. transfer_type/min_transfer_time between its
            # platforms). New children get that same pattern against
            # every existing platform and against each other
            internal = augmented[
                (augmented["_from_parent"] == augmented["_to_parent"])
                & augmented["_from_parent"].isin(new_children_by_station)
            ]

            internal_propagated_frames = []
            for station_id, children in new_children_by_station.items():
                station_internal = internal[internal["_from_parent"] == station_id]
                if station_internal.empty:
                    continue

                # Heuristic: use the most common (transfer_type,
                # min_transfer_time) pair among the station's existing
                # internal rows as the template for newly generated pairs.
                # A station can in principle mix multiple internal transfer
                # types (e.g. most platforms are type 2 but one pair is
                # type 3 "not possible") -- this picks the dominant pattern
                # rather than trying to infer per-pair semantics that
                # aren't derivable from the new children alone.
                template = (
                    station_internal.groupby(["transfer_type", "min_transfer_time"])
                    .size()
                    .idxmax()
                )
                template_type, template_min_time = template

                existing_platforms = pd.unique(
                    pd.concat([station_internal["from_stop_id"], station_internal["to_stop_id"]])
                )

                pair_targets = list(existing_platforms) + [
                    c for c in children  # new children against each other too
                ]

                rows = []
                for child in children:
                    for target in pair_targets:
                        if target == child:
                            continue
                        rows.append({
                            "from_stop_id": child, "to_stop_id": target,
                            "from_route_id": "", "to_route_id": "",
                            "from_trip_id": "", "to_trip_id": "",
                            "transfer_type": template_type,
                            "min_transfer_time": template_min_time,
                        })
                        rows.append({
                            "from_stop_id": target, "to_stop_id": child,
                            "from_route_id": "", "to_route_id": "",
                            "from_trip_id": "", "to_trip_id": "",
                            "transfer_type": template_type,
                            "min_transfer_time": template_min_time,
                        })

                if rows:
                    internal_propagated_frames.append(pd.DataFrame(rows, columns=GTFS_Schema.transfers_columns))

            internal_propagated = (
                pd.concat(internal_propagated_frames, ignore_index=True)
                if internal_propagated_frames
                else pd.DataFrame(columns=GTFS_Schema.transfers_columns)
            )

            helper_columns = ["_from_parent", "_to_parent", "_station_id", "_child_stop_id"]
            propagated = pd.concat(
                [
                    inter_propagated.drop(columns=helper_columns),
                    internal_propagated,
                ],
                ignore_index=True,
            )

    # Combine and de-duplicate 
    merged = pd.concat([self_transfers, propagated, other_transfers], ignore_index=True)

    key_columns = ["from_stop_id", "to_stop_id", "from_route_id", "to_route_id", "from_trip_id", "to_trip_id"]
    merged = merged.drop_duplicates(subset=key_columns, keep="first").reset_index(drop=True)

    if return_matches:
        return merged, propagated

    return merged

######## --------------------- MERGING ----------------------- ############


def merge(
    self: "GTFS",
    other: "GTFS",
    distance_threshold_m: float = 250.0,
    time_tolerance_sec: int = 60,
    create_transfers: bool = True,
    output_path: str = None
) -> "GTFS":
    """
    Merge another GTFS feed into this GTFS.

    self is authoritative for matched stops, trips, routes and services.

    Returns
    -------
    GTFS
        A new merged GTFS object.
    """
    print("Start to merge stops")
    # Merge stops
    merged_stops, parent_mapping, child_mapping, child_to_parent_self, stop_id_mapping = merge_stops(
        self,
        other,
        distance_threshold_m=distance_threshold_m,
        return_matches=True,
    )
    print("Start to match parent mapping")
    # Build the stop -> parent mapping for the original other feed
    # (needed for merge_transfers)
    child_to_parent_other = gtfs_utils._build_child_to_parent_mapping(other.files["stops.txt"])
    # Map children from other to parent id of self (necessary to get the same
    # hash signature across both feeds)
    child_to_parent_other_for_matching = {
        child_id: parent_mapping.get(parent_id, parent_id)
        for child_id, parent_id in child_to_parent_other.items()
    }

    print("Start to expand frequencies to trips and stop times")
    self = frequencies_to_trips(self)
    other = frequencies_to_trips(other)

    print("Start to match trips")
    trip_matches = _match_trips(
            self, other,
            first_child_to_parent=child_to_parent_self,
            second_child_to_parent=child_to_parent_other_for_matching,
            time_tolerance_sec=time_tolerance_sec,
        )

    print("Start to match services")
    # Merge services
    service_matches, self_dates, other_dates = _match_services(self, other, trip_matches)
    service_mapping = _service_mapping(self, service_matches)


    print("Start to merge trips")
    # Merge trips
    merged_trips, merged_stop_times, trip_matches = merge_trips(
        self,
        other,
        trip_matches=trip_matches,
        stop_id_mapping=stop_id_mapping,
        service_mapping=service_mapping,
        return_matches=True,
    )


    print("Start to merge routes")
    # Merge routes
    merged_routes, route_matches = merge_routes(
        self,
        other,
        trip_matches=trip_matches,
        return_matches=True,
    )
    
    print("Start to merge agencies")
    merged_routes, merged_agency = merge_agencies(
        self,
        other,
        merged_routes,
        route_matches,
    )

    print("Start to merge services")
    merged_calendar, merged_calendar_dates = merge_services(
        self_dates=self_dates,
        other_dates=other_dates,
        merged_trips=merged_trips,
        service_mapping=service_mapping
    )

    # Create a new GTFS object
    merged_files = {
        "agency.txt": merged_agency,
        "routes.txt": merged_routes,
        "stops.txt": merged_stops,
        "stop_times.txt": merged_stop_times,
        "trips.txt": merged_trips,
        "calendar.txt": merged_calendar,
        "calendar_dates.txt": merged_calendar_dates,
    }
    print("Start to merge transfers")
    merged_transfers = merge_transfers(
        self,
        other,
        parent_mapping=parent_mapping,
        other_child_to_parent=child_to_parent_other,
        child_mapping=child_mapping,
        trip_matches=trip_matches,
    )

    if merged_transfers is not None:
        merged_files["transfers.txt"] = merged_transfers

    merged_gtfs = GTFS(merged_files)

    if create_transfers:
        print("Start to create intermetropolitan transfers")
        new_transfers = create_intercity_transfers(merged_gtfs)
        if merged_transfers is not None:
            final_transfers = pd.concat([merged_transfers, new_transfers], ignore_index=True)
        # only if the two feed have no transfers (i.e.merged_transfers returns None)
        else:
            final_transfers = new_transfers
        merged_gtfs.files["transfers.txt"] = final_transfers

    
    if output_path is not None:
        print("Starting the export")
        GTFS.export_gtfs(merged_gtfs, output_path)

    return merged_gtfs


@staticmethod
def merge_multiple_gtfs(
    authoritative: "GTFS | str | Path",
    others: list["GTFS"],
    distance_threshold_m: float = 250.0,
    time_tolerance_sec: int = 60,
    create_transfers: bool = True,
    output_path: str | Path | None = None,
) -> "GTFS":
    """
    Fold an arbitrary number of national GTFS feeds into one, without ever
    holding more than two feeds' worth of tables in memory at once.

    `authoritative` wins over every subsequent feed for matched IDs; among
    the `others`, each one is authoritative over feeds that come after it
    in the list.

    Args:
        authoritative: base feed — a GTFS object, or a path to a .zip.
        others: remaining feeds to fold in, in order. Prefer passing paths
                save RAM.
        output_path: if given, the final merged feed is written here.

    Returns:
        The final merged GTFS.
    """

    def _load(feed) -> "GTFS":
        return feed if isinstance(feed, GTFS) else GTFS.from_zip(feed)

    accumulator = _load(authoritative)

    for other_feed in others:
        print(f"Starting adding {other_feed}")
        other = _load(other_feed)

        merged = merge(
            accumulator,
            other,
            distance_threshold_m=distance_threshold_m,
            time_tolerance_sec=time_tolerance_sec,
            create_transfers=False
        )

        # Drop the feed that was just merged to save RAM.
        del other
        del accumulator
        gc.collect()

        accumulator = merged

    if create_transfers:
        print("Start to create intermetropolitan transfers.")
        old_transfers = accumulator.files.get("transfers.txt")
        new_transfers = create_intercity_transfers(accumulator)

        if old_transfers is not None:
            final_transfers = pd.concat([old_transfers, new_transfers], ignore_index=True)
        else:
            final_transfers = new_transfers

        accumulator.files["transfers.txt"] = final_transfers

    if output_path is not None:
        print("Start the export")
        accumulator.export_gtfs(output_path)

    return accumulator