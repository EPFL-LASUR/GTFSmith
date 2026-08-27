import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import gc
from geopy.geocoders import Nominatim
import unicodedata
import re

from .gtfs_class import GTFS
from .gtfs_schema import GTFS_Schema



def convert_to_seconds(value):
    """Convert hhmmss to seconds"""

    # Check that the type is str
    if not isinstance(value, str):
        print("Unexpected value:", repr(value), type(value))
        raise TypeError(
            f"Expected a GTFS time string, got "
            f"{type(value).__name__}: {value!r}"
        )

    h, m, s = value.split(":")
    return (int(h) * 3600 + int(m) * 60 + int(s))


def _vectorized_to_seconds(col: pd.Series) -> np.ndarray:
    """
    Vectorized equivalent of gtfs_utils.convert_to_seconds applied over a whole
    column. Used in hot paths where the row-wise .apply(gtfs_utils.convert_to_seconds)
    would be re-run unnecessarily and allocate a Python int per row.
    """
    parts = col.str.split(":", n=2, expand=True)
    hours = parts[0].astype("int32")
    minutes = parts[1].astype("int32")
    seconds = parts[2].astype("int32")

    return (hours * 3600 + minutes * 60 + seconds).to_numpy()


def to_hhmmss(seconds: int) -> str:
    """
    Convert seconds to hhmmss
    """
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"





def _normalize_stop_name(name: "GTFS") -> str:
    """
    Normalize a stop name before comparing it
    """

    if pd.isna(name):
        return ""

    # Remove accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(char for char in name if not unicodedata.combining(char))

    # Replace punctuation with spaces
    name = re.sub(r"\s+", " ", name).strip()

    return name

def _normalize_platform_code(value) -> str | None:
    """
    Normalize a platform_code for comparison: strip whitespace, uppercase.
    Returns None for missing/empty codes.
    """
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    return value if value else None


def _normalize_transfers(transfers: pd.DataFrame | None) -> pd.DataFrame:
    """
    Reindex a transfers.txt frame onto the full canonical column set 
    to safely concatenate files from different feeds.
    """
    if transfers is None:
        return pd.DataFrame(columns=GTFS_Schema.transfers_columns)

    transfers = transfers.reindex(columns=GTFS_Schema.transfers_columns)
    transfers[["from_route_id", "to_route_id", "from_trip_id", "to_trip_id"]] = (
        transfers[["from_route_id", "to_route_id", "from_trip_id", "to_trip_id"]].fillna("")
    )
    transfers["transfer_type"] = transfers["transfer_type"].fillna("0").astype(str)
    return transfers



def _get_parent_stations(stops: pd.DataFrame) -> pd.DataFrame:
    """
    Return only parent stations (location_type = 1).
    """

    if "location_type" not in stops.columns:
        raise ValueError("stops.txt does not contain location_type")

    location_type = pd.to_numeric(stops["location_type"], errors="coerce")

    return stops.loc[location_type == 1].copy()


def _prepare_coordinates(stops: pd.DataFrame) -> pd.DataFrame:
    """
    Convert stop_lat/stop_lon to numeric values and remove rows
    with missing coordinates.
    """

    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")

    return stops.dropna(
        subset=["stop_lat", "stop_lon"]
    )


def _latlon_to_xy(stops: pd.DataFrame, reference_lat: float) -> np.ndarray:
    """
    Convert latitude/longitude to an approximate Cartesian coordinate system in meters.
    This approximation is enough for matching stations within a few hundred meters.
    """

    meters_per_degree_lat = 111200.0
    meters_per_degree_lon = meters_per_degree_lat * np.cos(np.radians(reference_lat))

    x = stops["stop_lat"].to_numpy() * meters_per_degree_lat
    y = stops["stop_lon"].to_numpy() * meters_per_degree_lon

    return np.column_stack([x, y])


def _build_child_to_parent_mapping(stops: pd.DataFrame) -> dict[str, str]:
    """
    Build a mapping from every stop_id to the station it belongs to.
        For a parent station: 8500010 -> 8500010
        For a platform: 8500010:0:18 -> 8500010
    """

    result = {}

    for _, row in stops.iterrows():
        stop_id = row["stop_id"]
        location_type = pd.to_numeric(row.get("location_type"), errors="coerce")

        if location_type == 1:
            result[stop_id] = stop_id
            continue

        parent_station = row.get("parent_station")

        if pd.notna(parent_station) and str(parent_station) != "":
            result[stop_id] = parent_station

    return result


def _build_parent_station_mapping(matches: pd.DataFrame) -> dict[str, str]:
    """"
    Build:
        second-feed parent station ID
                    ->
        first-feed parent station ID
        
    """

    if matches.empty:
        return {}

    return dict(zip(
        matches["second_stop_id"], matches["first_stop_id"]
    ))


def _find_nearby_parent_stations(
    stops: pd.DataFrame,
    radius_m: float = 500.0,
) -> pd.DataFrame:
    """
    Find pairs of parent stations that are within radius_m.

    Each pair is returned only once.

    Returns:
        first_stop_id
        second_stop_id
        distance_m
    """

    parents = _get_parent_stations(stops)
    parents = _prepare_coordinates(parents)

    if parents.empty:
        return pd.DataFrame(columns=["first_stop_id", "second_stop_id", "distance_m"])

    # Preserve stops.txt ordering.
    parents = parents.reset_index(drop=True)

    reference_lat = parents["stop_lat"].mean()
    xy = _latlon_to_xy(parents, reference_lat)

    tree = cKDTree(xy)

    matches = []

    for first_position, first_point in enumerate(xy):

        candidate_positions = tree.query_ball_point(first_point, radius_m)

        for second_position in candidate_positions:

            # Don't match the station with itself. Only keep each pair once.
            if second_position <= first_position:
                continue

            second_point = xy[second_position]

            distance_m = float(np.linalg.norm(first_point - second_point))

            matches.append({
                "first_stop_id": parents.iloc[first_position]["stop_id"],
                "second_stop_id": parents.iloc[second_position]["stop_id"],
                "distance_m": distance_m,
            })

    return pd.DataFrame(matches)


def _build_parent_merge_mapping(
    stops: pd.DataFrame,
    nearby_parents: pd.DataFrame,
) -> dict[str, str]:
    """
    Build:

        losing_parent_id -> winning_parent_id

    Nearby parent stations are treated as connected components.

    The parent station appearing first in stops.txt is selected
    as the winner of each component.
    """

    if nearby_parents.empty:
        return {}

    # Position of every stop in the original stops.txt.
    stop_order = {
        stop_id: position
        for position, stop_id in enumerate(stops["stop_id"])
    }

    # Build an undirected graph.
    graph: dict[str, set[str]] = {}

    for _, row in nearby_parents.iterrows():

        first_id = row["first_stop_id"]
        second_id = row["second_stop_id"]

        graph.setdefault(first_id, set()).add(second_id)
        graph.setdefault(second_id, set()).add(first_id)

    mapping: dict[str, str] = {}
    visited: set[str] = set()

    for start_id in graph:

        if start_id in visited:
            continue

        component = []
        stack = [start_id]

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            component.append(current)

            for neighbour in graph.get(current, set()):
                if neighbour not in visited:
                    stack.append(neighbour)

        if len(component) <= 1:
            continue

        # The first parent in stops.txt wins.
        winning_parent = min(
            component,
            key=lambda stop_id: stop_order.get(stop_id, float("inf")),
        )

        for parent_id in component:

            if parent_id != winning_parent:
                mapping[parent_id] = winning_parent

    return mapping


def _remap_parent_references(
    gtfs,
    parent_mapping: dict[str, str],
) -> None:
    """
    Remap references to losing parent station IDs.
    Children keep their own stop_ids.
    """

    if not parent_mapping:
        return gtfs

    # stops.txt
    stops = gtfs.files["stops.txt"]

    if "parent_station" in stops.columns:

        stops["parent_station"] = stops["parent_station"].replace(parent_mapping)

    gtfs.files["stops.txt"] = stops

    return gtfs


def valid_id(column, valid_ids):
    """Checks that an ID is valid"""

    return (
        column.isna()
        | column.eq("")
        | column.isin(valid_ids)
    )


def get_coordinates(place):
    """
    Some feeds are missing coordinates, especially for stops outside of "their" country.
    This function geocodes the station names to allow stop matching while merging feeds.
    """

    geolocator = Nominatim(user_agent="city_mapper", timeout=10)
    try:
        location = geolocator.geocode(place)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"Error with {place}: {e}")
    return None, None

import pandas as pd



def normalize_route_type(x):
    """
    Swiss GTFS have more detailed description of their service than
    for example french GTFS. Therefore, the description of swiss services
    is tailord down to match the french ones.
    """
    x = int(x)
    if 100 <= x < 200:
        return 2   # rail
    if 400 <= x < 500:
        return 1   # metro
    if 700 <= x < 800:
        return 3   # bus
    if 900 <= x < 1000:
        return 0   # tram
    if x == 1300:
        return 6   # gondola / téléphérique
    if x == 1400:
        return 7   # funicular
    if x == 1000:
        return 4   # ferry
    if x==1303:
        return 6   # gondola / téléphérique
    if x==1500:
        return 3   # bus (1500 is "taxi" but we don't have a better option)
    if x==202:
        return 3  # bus (202 is "intercity bus" but we don't have a better option)
    return x