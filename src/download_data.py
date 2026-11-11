"""
Download FEMA NFIP data from official OpenFEMA sources into data/raw/.

Usage (from the project folder, in PowerShell):
    .venv\\Scripts\\python.exe src\\download_data.py fields        # field lists (small)
    .venv\\Scripts\\python.exe src\\download_data.py claims        # all NFIP claims
    .venv\\Scripts\\python.exe src\\download_data.py policies-va   # Virginia policies
    .venv\\Scripts\\python.exe src\\download_data.py all           # everything above

What this script does for you:
  * Retries on network hiccups (waits a little longer after each failure).
  * Skips anything that's already fully downloaded, so you can safely re-run it.
  * Resumes large API downloads page by page if it gets interrupted.
  * Prints progress so you can see it's working.

Sources: only fema.gov (OpenFEMA). See https://www.fema.gov/about/openfema/api
"""

import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

API_BASE = "https://www.fema.gov/api/open"
# An honest description of who is calling. No personal information.
HEADERS = {"User-Agent": "flood-risk-explorer/0.1 (student portfolio project; python-requests)"}

MAX_ATTEMPTS = 5          # how many times to try one request before giving up
PAGE_SIZE = 10_000        # OpenFEMA's maximum rows per API request
PAUSE_SECONDS = 0.5       # polite pause between API requests
TIMEOUT_SECONDS = 120     # give up on a single request after this long

# Official full-file download for claims (FEMA's catalog lists this URL).
CLAIMS_PARQUET_URL = "https://www.fema.gov/about/reports-and-data/openfema/v3/NfipClaimsV3.parquet"

# The policy columns we realistically need for modelling exposure (37 of 87).
POLICY_COLUMNS = [
    # identifiers and dates
    "id", "policyEffectiveDate", "policyTerminationDate", "policyTermIndicator",
    "originalNBDate", "cancellationDateOfFloodPolicy",
    # exposure: how much is insured, and how many units
    "policyCount", "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage",
    "buildingDeductibleCode", "contentsDeductibleCode",
    "totalInsurancePremiumOfThePolicy", "policyCost", "buildingReplacementCost",
    # location
    "propertyState", "reportedZipCode", "reportedCity", "censusGeoid",
    "latitude", "longitude", "nfipRatedCommunityNumber", "nfipCommunityNumberCurrent",
    "nfipCommunityName", "crsClassCode",
    # flood zone and rating
    "ratedFloodZone", "floodZoneCurrent", "rateMethod", "elevationDifference",
    "elevatedBuildingIndicator", "postFIRMConstructionIndicator",
    # building characteristics
    "occupancyType", "primaryResidenceIndicator", "numberOfFloorsInInsuredBuilding",
    "basementEnclosureCrawlspaceType", "foundationType", "originalConstructionDate",
    "condominiumCoverageTypeCode",
]


class FatalHTTPError(Exception):
    """A response that retrying won't fix (for example 403 Forbidden or 404 Not Found)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def mb(num_bytes):
    """Format a byte count as megabytes, e.g. '123.4 MB'."""
    return f"{num_bytes / 1_048_576:,.1f} MB"


def get_with_retry(url, params=None, stream=False):
    """
    GET a URL, retrying on temporary problems.

    Temporary problems = connection errors, timeouts, HTTP 429 (too many
    requests) and 5xx (server errors). Between attempts we wait 2, 4, 8, 16
    seconds ("exponential backoff"), which gives the server time to recover.
    Other errors like 403/404 won't fix themselves, so we stop right away.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, headers=HEADERS,
                                    timeout=TIMEOUT_SECONDS, stream=stream)
            if response.status_code == 200:
                return response
            if response.status_code == 429 or response.status_code >= 500:
                problem = f"HTTP {response.status_code}"
            else:
                raise FatalHTTPError(f"HTTP {response.status_code} for {response.url}")
        except (requests.ConnectionError, requests.Timeout) as err:
            problem = type(err).__name__

        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(f"Gave up after {MAX_ATTEMPTS} attempts ({problem}): {url}")
        wait = 2 ** attempt
        print(f"    {problem}; retrying in {wait}s (attempt {attempt}/{MAX_ATTEMPTS})")
        time.sleep(wait)


# ---------------------------------------------------------------------------
# 1. Field lists (FEMA's official descriptions of each column)
# ---------------------------------------------------------------------------
def get_field_list(dataset, version):
    """Download FEMA's official field list for a dataset; cache it as JSON."""
    dest = RAW_DIR / f"{dataset}V{version}_fields.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))

    print(f"Fetching field list for {dataset} v{version} ...")
    params = {"$filter": f"openFemaDataSet eq '{dataset}' and datasetVersion eq {version}",
              "$top": 1000}
    data = get_with_retry(f"{API_BASE}/v1/DataSetFields", params=params).json()
    fields = [{"name": f["name"], "title": f["title"], "type": f["type"],
               "description": f["description"]}
              for f in sorted(data["DataSetFields"], key=lambda f: f["sortOrder"])]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(fields, indent=2), encoding="utf-8")
    print(f"  saved {len(fields)} fields -> {dest.relative_to(PROJECT_ROOT)}")
    return fields


def arrow_type(fema_type):
    """Translate a FEMA field type (e.g. 'decimal(12,2)') into a Parquet column type."""
    if fema_type in ("smallint", "integer", "bigint"):
        return pa.int64()
    if fema_type.startswith("decimal"):
        return pa.float64()
    if fema_type == "boolean":
        return pa.bool_()
    if fema_type in ("date", "datetime", "datetimez"):
        return pa.timestamp("ms", tz="UTC")
    return pa.string()  # text, uuid, and anything unexpected


def records_to_table(records, schema):
    """
    Turn one page of API records into a typed Arrow table.

    Using FEMA's official types for every page means all pages have the same
    column types and can be stacked together, even if one page happens to have
    a column that is entirely empty.
    """
    df = pd.DataFrame.from_records(records).reindex(columns=schema.names)
    for field in schema:
        col = df[field.name]
        if pa.types.is_timestamp(field.type):
            df[field.name] = pd.to_datetime(col, utc=True, errors="coerce")
        elif pa.types.is_integer(field.type):
            df[field.name] = pd.to_numeric(col, errors="coerce").astype("Int64")
        elif pa.types.is_floating(field.type):
            df[field.name] = pd.to_numeric(col, errors="coerce").astype("float64")
        elif pa.types.is_boolean(field.type):
            df[field.name] = col.astype("boolean")
        else:
            df[field.name] = col.astype("string")
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


# ---------------------------------------------------------------------------
# 2. Download a ready-made file (streamed, with progress)
# ---------------------------------------------------------------------------
def download_file(url, dest):
    """Stream a file to dest. Skips it if dest already exists."""
    dest = Path(dest)
    if dest.exists():
        print(f"{dest.name}: already downloaded ({mb(dest.stat().st_size)}), skipping.")
        return dest

    print(f"Downloading {url}")
    response = get_with_retry(url, stream=True)
    total = int(response.headers.get("Content-Length", 0))
    partial = dest.with_suffix(dest.suffix + ".part")   # unfinished download
    done, next_report = 0, 0
    with open(partial, "wb") as out:
        for chunk in response.iter_content(chunk_size=1_048_576):
            out.write(chunk)
            done += len(chunk)
            if done >= next_report:
                pct = f" ({done / total:.0%})" if total else ""
                print(f"  {mb(done)}{pct}")
                next_report += 50 * 1_048_576            # report every 50 MB
    partial.replace(dest)   # only rename once the file is complete
    print(f"  finished: {dest.name} ({mb(dest.stat().st_size)})")
    return dest


# ---------------------------------------------------------------------------
# 3. Download through the API, one page at a time
# ---------------------------------------------------------------------------
def download_api(dataset, version, dest, filter=None, select=None,
                 page_size=PAGE_SIZE, pause=PAUSE_SECONDS):
    """
    Download a dataset through the OpenFEMA API and save it as one Parquet file.

    The API returns at most `page_size` rows per request, so we ask for page 1
    ($skip=0), then page 2 ($skip=10000), and so on ("pagination"). Each page
    is saved to its own small file first, so an interrupted download resumes
    where it stopped instead of starting over.
    """
    dest = Path(dest)
    if dest.exists():
        print(f"{dest.name}: already downloaded ({mb(dest.stat().st_size)}), skipping.")
        return dest

    # Build a fixed column schema from FEMA's official field types.
    fields = get_field_list(dataset, version)
    wanted = select or [f["name"] for f in fields]
    types = {f["name"]: f["type"] for f in fields}
    missing = [c for c in wanted if c not in types]
    if missing:
        raise ValueError(f"These columns aren't in {dataset} v{version}: {missing}")
    schema = pa.schema([(c, arrow_type(types[c])) for c in wanted])

    url = f"{API_BASE}/v{version}/{dataset}"
    base_params = {"$orderby": "id"}           # a fixed order keeps pages consistent
    if filter:
        base_params["$filter"] = filter
    if select:
        base_params["$select"] = ",".join(select)

    # Ask for 1 row just to learn the total row count.
    first = get_with_retry(url, params={**base_params, "$top": 1, "$count": "true"}).json()
    total_rows = first["metadata"]["count"]
    n_pages = -(-total_rows // page_size)          # ceiling division
    print(f"{dataset} v{version}: {total_rows:,} rows -> {n_pages} pages of {page_size:,}")

    page_dir = RAW_DIR / "_pages" / dest.stem
    page_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for page in range(n_pages):
        page_file = page_dir / f"page_{page + 1:05d}.parquet"
        if page_file.exists():
            continue                               # done on an earlier run
        params = {**base_params, "$top": page_size, "$skip": page * page_size,
                  "$metadata": "off"}
        records = get_with_retry(url, params=params).json()[dataset]
        pq.write_table(records_to_table(records, schema), page_file)

        elapsed = time.time() - start
        print(f"  page {page + 1}/{n_pages}: {len(records):,} rows "
              f"({(page + 1) / n_pages:.0%}, {elapsed / 60:.1f} min elapsed)")
        time.sleep(pause)                          # be polite to FEMA's servers

    # Combine all pages into one file, check the row count, then clean up.
    print("  combining pages ...")
    page_files = sorted(page_dir.glob("page_*.parquet"))
    partial = dest.with_suffix(".parquet.part")
    rows_written = 0
    with pq.ParquetWriter(partial, schema, compression="zstd") as writer:
        for page_file in page_files:
            table = pq.read_table(page_file, schema=schema)
            writer.write_table(table)
            rows_written += table.num_rows

    if rows_written != total_rows:
        partial.unlink()
        raise RuntimeError(f"Row count mismatch: got {rows_written:,}, API says {total_rows:,}. "
                           f"Pages kept in {page_dir} so you can investigate.")
    partial.replace(dest)
    shutil.rmtree(page_dir)
    print(f"  finished: {dest.name}: {rows_written:,} rows, {mb(dest.stat().st_size)}")
    return dest


# ---------------------------------------------------------------------------
# The specific downloads for this project
# ---------------------------------------------------------------------------
def download_claims():
    """All NFIP claims. Try FEMA's full Parquet file first; fall back to the API."""
    dest = RAW_DIR / "NfipClaimsV3.parquet"
    if dest.exists():
        print(f"{dest.name}: already downloaded ({mb(dest.stat().st_size)}), skipping.")
        return dest
    try:
        return download_file(CLAIMS_PARQUET_URL, dest)
    except FatalHTTPError as err:
        print(f"  Full-file download refused ({err}).")
        print("  Falling back to the official OpenFEMA API (slower, same data).")
        return download_api("NfipClaims", 3, dest)


def download_policies_va():
    """NFIP policies for Virginia only, with only the columns we need."""
    return download_api("NfipPolicies", 3, RAW_DIR / "NfipPoliciesV3_VA.parquet",
                        filter="propertyState eq 'VA'", select=POLICY_COLUMNS)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what not in ("fields", "claims", "policies-va", "all"):
        print(__doc__)
        sys.exit(1)

    if what in ("fields", "all"):
        get_field_list("NfipClaims", 3)
        get_field_list("NfipPolicies", 3)
    if what in ("claims", "all"):
        download_claims()
    if what in ("policies-va", "all"):
        download_policies_va()

    print("\nFiles in data/raw:")
    for path in sorted(RAW_DIR.glob("*")):
        if path.is_file():
            print(f"  {path.name:35s} {mb(path.stat().st_size):>12s}")


if __name__ == "__main__":
    main()
