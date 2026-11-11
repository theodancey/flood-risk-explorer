"""
Download FEMA NFIP data from official OpenFEMA sources into data/raw/.

Usage (from the project folder, in PowerShell):
    .venv\\Scripts\\python.exe src\\download_data.py fields        # field lists (small)
    .venv\\Scripts\\python.exe src\\download_data.py claims        # all NFIP claims
    .venv\\Scripts\\python.exe src\\download_data.py policies-va   # Virginia policies
    .venv\\Scripts\\python.exe src\\download_data.py counties      # county names (Census, small)
    .venv\\Scripts\\python.exe src\\download_data.py cpi           # CPI-U inflation index (BLS, small)
    .venv\\Scripts\\python.exe src\\download_data.py coastal       # coastline county list (Census, small)
    .venv\\Scripts\\python.exe src\\download_data.py all           # everything above

What this script does for you:
  * Retries on network hiccups (waits a little longer after each failure).
  * Skips anything that's already fully downloaded, so you can safely re-run it.
  * Resumes large API downloads page by page if it gets interrupted.
  * Prints progress so you can see it's working.

Sources: fema.gov (OpenFEMA, see https://www.fema.gov/about/openfema/api), plus small public
reference files: Census Bureau county names and coastline counties, and the CPI-U inflation
index from the Bureau of Labor Statistics (BLS) public API.
None of these need an account or API key.
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

# Census Bureau list of every US county and its FIPS code (about 0.1 MB, "|"-separated text).
CENSUS_COUNTY_URL = "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt"

# Census Bureau list of "coastline counties" (counties touching the ocean, Gulf, Great Lakes,
# or a major estuary such as the Chesapeake Bay), from its 2018 "Coastline America" release.
CENSUS_COASTLINE_URL = "https://www2.census.gov/library/stories/2018/08/coastline-counties-list.xlsx"

# CPI-U (Consumer Price Index for All Urban Consumers, all items, US city average, not
# seasonally adjusted, 1982-84 = 100), monthly, straight from the Bureau of Labor Statistics'
# free public API (version 1 needs no key; limit 10 years per request, 25 requests per day).
# BLS series ID CUUR0000SA0 is the same series FRED calls CPIAUCNS. FRED's download URL timed
# out for Python scripts (it answered curl), so we use BLS, the original source, instead.
BLS_API_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
BLS_CPI_SERIES = "CUUR0000SA0"
CPI_YEARS = (2006, 2025)      # must cover 2010-2025; requested in 10-year chunks

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

    The API returns at most `page_size` rows per request, so we fetch the data
    in pages ("pagination"), sorted by id: each request asks for the next rows
    after the last id we received. Each page is saved to its own small file
    first, so an interrupted download resumes where it stopped instead of
    starting over.
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

    # Resume: count the pages saved on earlier runs and find the last id we got.
    page_dir = RAW_DIR / "_pages" / dest.stem
    page_dir.mkdir(parents=True, exist_ok=True)
    for leftover in page_dir.glob("*.tmp"):
        leftover.unlink()                          # half-written page from a crash
    saved = sorted(page_dir.glob("page_*.parquet"))
    rows_done = sum(pq.ParquetFile(p).metadata.num_rows for p in saved)
    last_id = (pq.read_table(saved[-1], columns=["id"]).column("id").to_pylist()[-1]
               if saved else None)
    if saved:
        print(f"  resuming: {len(saved)} pages ({rows_done:,} rows) already saved")

    # "Keyset" pagination: instead of "skip the first 400,000 rows" (which
    # gets slower the deeper we go, because the server still has to count past
    # them), we ask for "the next rows whose id is greater than the last id we
    # received". That stays fast no matter how far into the data we are.
    page = len(saved)
    start = time.time()
    while rows_done < total_rows:
        page_filter = filter
        if last_id is not None:
            id_filter = f"id gt {last_id}"
            page_filter = f"({filter}) and {id_filter}" if filter else id_filter
        params = {**base_params, "$top": page_size, "$metadata": "off"}
        if page_filter:
            params["$filter"] = page_filter
        records = get_with_retry(url, params=params).json()[dataset]
        if not records:
            break                                  # nothing left to fetch

        page += 1
        page_file = page_dir / f"page_{page:05d}.parquet"
        tmp_file = page_file.with_suffix(".tmp")   # write, then rename = no half files
        pq.write_table(records_to_table(records, schema), tmp_file)
        tmp_file.replace(page_file)
        last_id = records[-1]["id"]
        rows_done += len(records)

        elapsed = time.time() - start
        print(f"  page {page}/{n_pages}: {len(records):,} rows "
              f"({rows_done / total_rows:.0%}, {elapsed / 60:.1f} min elapsed)")
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


def download_county_names():
    """County FIPS code -> county name lookup from the Census Bureau (used to label charts)."""
    return download_file(CENSUS_COUNTY_URL, RAW_DIR / "census_counties_2020.txt")


def download_cpi():
    """Monthly CPI-U index from the BLS public API, saved as data/raw/cpi_u_bls.csv.

    Used to convert dollars to 2025 dollars. Columns: year, month, value, footnote.
    A month BLS couldn't publish has a blank value and a footnote saying why
    (e.g. October 2025, during the federal government shutdown)."""
    dest = RAW_DIR / "cpi_u_bls.csv"
    if dest.exists():
        print(f"{dest.name}: already downloaded, skipping.")
        return dest
    rows = []
    first, last = CPI_YEARS
    for start in range(first, last + 1, 10):            # BLS v1: at most 10 years per request
        end = min(start + 9, last)
        print(f"Requesting CPI-U {start}-{end} from the BLS public API")
        body = {"seriesid": [BLS_CPI_SERIES], "startyear": str(start), "endyear": str(end)}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = requests.post(BLS_API_URL, json=body, headers=HEADERS, timeout=TIMEOUT_SECONDS)
                reply = resp.json()
                if resp.status_code == 200 and reply.get("status") == "REQUEST_SUCCEEDED":
                    break
                problem = f"HTTP {resp.status_code}, {reply.get('status')}: {reply.get('message')}"
            except (requests.RequestException, ValueError) as err:
                problem = type(err).__name__
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"BLS API failed after {MAX_ATTEMPTS} attempts ({problem})")
            print(f"    {problem}; retrying in {2 ** attempt}s")
            time.sleep(2 ** attempt)
        for d in reply["Results"]["series"][0]["data"]:
            if not d["period"].startswith("M") or d["period"] == "M13":   # M13 = annual average
                continue
            notes = "; ".join(f["text"] for f in d.get("footnotes", []) if f.get("text"))
            value = pd.to_numeric(d["value"], errors="coerce")             # "-" = not published
            rows.append({"year": int(d["year"]), "month": int(d["period"][1:]),
                         "value": value, "footnote": notes})
        time.sleep(PAUSE_SECONDS)
    table = pd.DataFrame(rows).sort_values(["year", "month"])
    partial = dest.with_suffix(".csv.part")
    table.to_csv(partial, index=False)
    partial.replace(dest)                               # only appears once complete
    print(f"  finished: {dest.name} ({len(table)} months)")
    return dest


def download_coastline_counties():
    """Census list of coastline counties (used for the 'coastal county' feature)."""
    return download_file(CENSUS_COASTLINE_URL, RAW_DIR / "census_coastline_counties.xlsx")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what not in ("fields", "claims", "policies-va", "counties", "cpi", "coastal", "all"):
        print(__doc__)
        sys.exit(1)

    if what in ("fields", "all"):
        get_field_list("NfipClaims", 3)
        get_field_list("NfipPolicies", 3)
    if what in ("claims", "all"):
        download_claims()
    if what in ("policies-va", "all"):
        download_policies_va()
    if what in ("counties", "all"):
        download_county_names()
    if what in ("cpi", "all"):
        download_cpi()
    if what in ("coastal", "all"):
        download_coastline_counties()

    print("\nFiles in data/raw:")
    for path in sorted(RAW_DIR.glob("*")):
        if path.is_file():
            print(f"  {path.name:35s} {mb(path.stat().st_size):>12s}")


if __name__ == "__main__":
    main()
