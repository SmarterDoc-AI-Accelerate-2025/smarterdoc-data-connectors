##############################################################
# Fivetran Custom Connector SDK: NPI Registry Provider Data (Connector A)
# Purpose:
# Fetches NPI (National Provider Identifier) data for healthcare providers from the
# NPPES NPI Registry API (CMS). This connector is designed to be highly configurable
# to target specific states, postal code areas, and medical specialties.

# Best Practices Implementation:
# 1. Sharding: Uses multi-dimensional looping over postal prefixes and taxonomy codes
#    to overcome the NPI API's 1200-record per-query limit.
# 2. Incrementality: Uses the NPI 'last_updated_epoch' field as a bookmark for
#    efficient incremental synchronization.
# 3. Resilience: Includes exponential backoff and retry logic for transient API failures.
# 4. Resumption: Uses 'last_processed_shard_key' to ensure syncs can resume exactly
#    where they left off after a failure.
#############################################################

import json
from datetime import datetime, timezone
import time
import typing as t
import requests as rq

from fivetran_connector_sdk import Connector
from fivetran_connector_sdk import Operations as op
from fivetran_connector_sdk import Logging as log

# ------- config -------

API_VERSION = "2.1"
# doctor practice location
ADDRESS_PURPOSE = "location"
DEFAULT_MAX_RETRIES = 5
RETRY_BACKOFF = 2.0
DEFAULT_STATE = "NY"

# ------ helpers ------


def utc_now_iso():
    """Returns the current UTC time in ISO 8601 format."""
    # This function is needed by the SDK's internal logging or state processing.
    return datetime.now(timezone.utc).isoformat()


def maybe_list(v) -> t.List[str]:
    if v is None: return []
    if isinstance(v, (list, tuple)): return [str(x) for x in v]
    # This line handles the deserialization:
    return [p for p in str(v).replace(",", " ").split() if p]


def _safe(d: dict, path: t.List[t.Union[str, int]], default=None):
    cur = d
    try:
        for p in path:
            cur = cur[p]
        return cur
    except (KeyError, IndexError, TypeError):
        return default


def choose_primary_taxonomy(taxonomies: t.Optional[list]) -> t.Optional[dict]:
    if not taxonomies: return None
    for txy in taxonomies:
        # NPI marks primary taxonomy; value can be True/"Y"/"true"
        val = str(txy.get("primary", "")).lower()
        if val in ("true", "1", "y", "yes"):
            return txy
    return taxonomies[0]


def extract_primary_desc(r: dict) -> t.Optional[str]:
    txy = choose_primary_taxonomy(r.get("taxonomies"))
    return txy.get("desc") if txy else None


def choose_location_address(addresses: t.Optional[list]) -> t.Optional[dict]:
    if not addresses: return None
    loc, mail = None, None
    for a in addresses:
        purpose = (a.get("address_purpose") or "").upper()
        if purpose == "LOCATION" and loc is None:
            loc = a
        elif purpose == "MAILING" and mail is None:
            mail = a
    return loc or mail or addresses[0]


def extract_city(r: dict) -> t.Optional[str]:
    a = choose_location_address(r.get("addresses"))
    return a.get("city") if a else None


def extract_state(r: dict) -> t.Optional[str]:
    a = choose_location_address(r.get("addresses"))
    return a.get("state") if a else None


def extract_zip(r: dict) -> t.Optional[str]:
    a = choose_location_address(r.get("addresses"))
    return a.get("postal_code") if a else None  # ZIP or ZIP+4


def extract_first_name(r: dict) -> t.Optional[str]:
    return _safe(r, ["basic", "first_name"])


def extract_last_name(r: dict) -> t.Optional[str]:
    return _safe(r, ["basic", "last_name"])


def extract_last_updated_epoch(r: dict) -> t.Optional[int]:
    return r.get("last_updated_epoch")


def epoch_to_iso(epoch: t.Optional[int]) -> t.Optional[str]:
    if epoch is None: return None
    try:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()
    except Exception:
        return None


def is_after_bookmark(record_epoch: t.Optional[int],
                      bookmark_iso: t.Optional[str]) -> bool:
    if not bookmark_iso: return True
    if record_epoch is None: return True
    try:
        rec_dt = datetime.fromtimestamp(int(record_epoch), tz=timezone.utc)
        bm = datetime.fromisoformat(bookmark_iso)
        return rec_dt > bm
    except Exception:
        return True


def _build_params(page_size: int, skip: int, state: str,
                  postal_prefix: t.Optional[str],
                  taxonomy_code: t.Optional[str]) -> t.Dict[str, str]:
    """
    Dynamically builds the API request parameters based on shard filters.
    """
    params = {
        "version": API_VERSION,
        "limit": str(page_size),
        "skip": str(skip),
        "address_purpose": ADDRESS_PURPOSE,
        "state": state,  # mandatory for sharding
    }

    if postal_prefix:
        params["postal_code"] = f"{postal_prefix}*"

    # NPI API accepts 'taxonomy_description' OR 'taxonomy',
    if taxonomy_code:
        params["taxonomy"] = taxonomy_code

    return params


def _request_with_retries(url: str, params: t.Dict[str, str], timeout_s: int,
                          max_retries: int, retry_backoff: float,
                          shard_label: str) -> rq.Response:
    """HTTP request with exponential retry mechanism """
    headers = {
        "User-Agent":
        "SmarterDoc NPI Connector (contact: yian.chen261@gmail.com) "
    }

    last_err = None
    for i in range(max_retries):
        try:
            resp = rq.get(url,
                          params=params,
                          headers=headers,
                          timeout=timeout_s)
            resp.raise_for_status()
            return resp
        except rq.HTTPError as e:
            if e.response.status_code == 400:
                log.severe(
                    f"NPI API 400 Error on shard {shard_label}: Query is too broad or invalid. Cannot retry."
                )
                raise

            # Transient error: 429 (Rate Limit) or 5xx (Server Error). Retry.
            last_err = e
            log.warning(
                f"Transient error ({e.response.status_code}) on shard {shard_label}. Retrying in {retry_backoff ** i:.1f}s..."
            )
            time.sleep(retry_backoff**i)
        except Exception as e:
            # Connection/Timeout error. Retry.
            last_err = e
            log.warning(
                f"Connection error on shard {shard_label}. Retrying in {retry_backoff ** i:.1f}s..."
            )
            time.sleep(retry_backoff**i)

    # If the loop finishes without success, raise the last error
    raise RuntimeError(
        f"Request failed for shard {shard_label} after {max_retries} retries: {last_err}"
    )


# only want NYC doctors
def _fetch_page_filtered(
    api_base: str,
    page_size: int,
    skip: int,
    timeout_s: int,
    max_retries: int,
    retry_backoff: float,
    state: str,
    postal_prefix: t.Optional[str],
    taxonomy_code: t.Optional[str],
) -> list:
    """
    Filtered call using practice LOCATION, postal_code wildcard, and taxonomy_description.
    API params reference (v2.1): city, state, postal_code(wildcard allowed), address_purpose, taxonomy_description, limit/skip.
    """
    url = api_base.rstrip("/") + "/"

    params = _build_params(page_size, skip, state, postal_prefix,
                           taxonomy_code)

    shard_label = f"ZIP={postal_prefix or 'ALL'}, TAX={taxonomy_code or 'ALL'}"

    headers = {
        "User-Agent":
        "SmarterDoc NPI Connector (contact: yian.chen261@gmail.com)"
    }

    resp = _request_with_retries(url=url,
                                 params=params,
                                 timeout_s=timeout_s,
                                 max_retries=max_retries,
                                 retry_backoff=retry_backoff,
                                 shard_label=shard_label)

    payload = resp.json() or {}
    return payload.get("results", []) or []


# ---------- fivetran SDK functions----------


def schema(configuration: dict):
    """
    Declare table + PK; Fivetran infers other columns & types
    """
    return [
        {
            "table": "npi_providers",
            "primary_key": ["npi"]
        },
    ]


def update(configuration: dict, state: dict):
    """
    update: This function is called by Fivetran in every sync.
    Gets the 

    Args:
        configuration (dict): dict containing connection details
        state (dict): dict containing state information from previous runs. 
        State dictionary is empty for the first sync of full re-sync
    """
    log.info("npi_registry: starting sync")

    api_base = configuration.get("api_base",
                                 "https://npiregistry.cms.hhs.gov/api/")
    page_size = int(configuration.get("page_size", 200))
    timeout_s = int(configuration.get("request_timeout_seconds", 30))
    max_retries = int(configuration.get("max_retries", DEFAULT_MAX_RETRIES))
    retry_backoff = float(
        configuration.get("retry_backoff_factor", RETRY_BACKOFF))
    state_filter = configuration.get("state_filter", DEFAULT_STATE)
    backoff_s = float(configuration.get("request_backoff_seconds", 0.25))

    # Optional limit
    max_pages = configuration.get("max_pages_per_sync")
    max_pages = int(max_pages) if max_pages not in (None, "") else None

    # Optional configurable fields
    shard_zip_prefixes = maybe_list(configuration.get("postal_prefixes",
                                                      "")) or [""]
    shard_taxonomy_codes = maybe_list(configuration.get("taxonomy_codes",
                                                        "")) or [""]

    bookmark_iso = state.get("last_updated_at")
    total_rows = 0
    shards_seen = 0
    pages = 0

    # store state of outer loop for mid-synced failures to resume
    last_processed_shard = state.get("last_processed_shard_key", None)

    is_resuming = last_processed_shard is not None

    # split connections into manageable units
    for zip_prefix in shard_zip_prefixes:
        for tcode in shard_taxonomy_codes:
            shards_seen += 1

            # create descriptive unique key to resume from
            current_shard_key = f"{state_filter}-{zip_prefix or 'ALL'}-{tcode or 'ALL'}"

            # If resuming, skip shards until we reach the last successful one
            if is_resuming and current_shard_key != last_processed_shard:
                log.info(f"Skipping previous shard: {current_shard_key}")
                continue

            # Start processing from this point forward
            log.info(f"Starting shard: {current_shard_key}")

            skip = 0
            while True:
                if max_pages is not None and pages >= max_pages:
                    log.warning(
                        f"Stopping sync due to max_pages_per_sync={max_pages} limit."
                    )
                    # persist current shard as checkpoint
                    op.checkpoint(
                        state={
                            "last_updated_at": bookmark_iso,
                            "last_processed_shard_key": current_shard_key
                        })
                    return  # exit sync

                results = _fetch_page_filtered(api_base, page_size, skip,
                                               timeout_s, max_retries,
                                               retry_backoff, state_filter,
                                               zip_prefix, tcode)
                pages += 1

                if not results:
                    break  # end of current shard

                for r in results:
                    rec_epoch = extract_last_updated_epoch(r)

                    # incremental sync check
                    if bookmark_iso and not is_after_bookmark(
                            rec_epoch, bookmark_iso):
                        continue  # Skip old record

                    record_iso = epoch_to_iso(rec_epoch)
                    row = {
                        "npi": str(r.get("number") or ""),
                        "first_name": extract_first_name(r),
                        "last_name": extract_last_name(r),
                        "primary_specialty": extract_primary_desc(r),
                        "city": extract_city(r),
                        "state": extract_state(r),
                        "zip": extract_zip(r),
                        "last_updated_at": record_iso,
                        "raw": r,
                    }
                    if not row["npi"]:
                        continue

                    # upsert row
                    op.upsert(table="npi_providers", data=row)
                    total_rows += 1

                    # advance bookmark
                    if record_iso and (bookmark_iso is None
                                       or record_iso > bookmark_iso):
                        bookmark_iso = record_iso

                # pagination check
                if len(results) < page_size:
                    break  # end of this shard

                skip += page_size
                time.sleep(backoff_s)  # Respect API Rate Limits

            # Checkpoint the successful completion of the current shard
            op.checkpoint(
                state={
                    "last_updated_at": bookmark_iso,
                    "last_processed_shard_key": current_shard_key
                })
            log.info(f"Finished shard: {current_shard_key}")

            # Reset resumption key after the successful shard completion
            last_processed_shard = None

    # final checkpoint
    op.checkpoint(state={
        "last_updated_at": bookmark_iso,
        "last_processed_shard_key": None
    })
    log.info(
        f"npi_registry: Sync complete. Shards seen={shards_seen}, pages={pages}, rows={total_rows}, final bookmark={bookmark_iso}"
    )


connector = Connector(update=update, schema=schema)

if __name__ == "__main__":
    try:
        with open("configuration.json", "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}
    # allows testing connector
    connector.debug(configuration=cfg)
