from pathlib import Path
import subprocess
import tempfile

from gtfsmith import GTFS
from gtfsmith import merge
from gtfsmith import merge_duplicated_physical_stops, create_parent_stations, prefix_stop_ids

# Helper to download GTFS feeds
def download_feed(url: str, destination: Path) -> None:
    subprocess.run(
        ["curl", "-L", "--fail", "--output", str(destination), url],
        check=True,
    )


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    renfe_zip = tmp / "renfe.zip"
    sncf_zip = tmp / "sncf.zip"
    output_path = tmp / "merged.zip"

    # Download GTFS feeds for the long distance services of the SNCF and Renfe
    download_feed("https://ssl.renfe.com/gtransit/Fichero_AV_LD/google_transit.zip", renfe_zip)
    download_feed("https://eu.ftp.opendatasoft.com/sncf/plandata/Export_OpenData_SNCF_GTFS_NewTripId.zip", sncf_zip)

    # Load them as GTFS objects
    renfe = GTFS.from_zip(renfe_zip)
    sncf = GTFS.from_zip(sncf_zip)

    # Merged french stops with the same physical location but two appearances in stops.txt
    sncf = merge_duplicated_physical_stops(sncf)

    # Create parent stations for the spanish stops
    renfe = create_parent_stations(renfe)

    # Prefix the stop IDs of the Renfe to make them unique
    renfe = prefix_stop_ids(renfe, "ES")



    # Merge the two feeds
    merged_gtfs = merge(sncf, renfe)

    # Export the merged feeds
    merged_gtfs.export_gtfs(output_path)