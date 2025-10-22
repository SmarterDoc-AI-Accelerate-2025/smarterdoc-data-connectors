##############################################################
# Fivetran Custom Connector SDK: Profiles Enrichment (Connector B)

# Purpose:
# This connector acts as a data consumer and producer. It pulls seed data (NPIs)
# from the raw BigQuery tables loaded by Connectors A and B, executes LLM enrichment
# (Gemini/Grounding) and embedding generation, and streams the final, enriched wide
# table back to the destination.

# Best Practices Implementation:
# 1. Orchestration: Uses BigQuery as a source to target recently updated NPIs.
# 2. Resilience: Robust try/except blocks around high-latency LLM/Embedding calls.
# 3. Schema Management: Explicitly defines the complex ARRAY and JSON column types
#    to guide Fivetran's schema creation in BigQuery.
# 4. Consumption Control: Uses LIMIT clauses in BQ queries to manage Vertex AI spending
#    and control sync runtime (Max Doctors per Sync).

##############################################################

import json
import typing as t
from datetime import datetime, timedelta, timezone
import re
from typing import List, Dict, Any, Tuple
from fivetran_connector_sdk import Connector
from fivetran_connector_sdk import Operations as op
from fivetran_connector_sdk import Logging as log

from google.cloud import bigquery
from google.cloud import aiplatform
from google import genai
from google.genai import types
from google.genai.errors import APIError

from pydantic import BaseModel, Field


# Pydantic schema
class RatingRecord(BaseModel):
    """Schema for a single rating/review record."""
    source: str = Field(
        description="Name of the review platform. (e.g. ZocDoc)")
    score: float = Field(description="The numerical rating (e.g. 4.5, 5.0).")
    count: int = Field(
        description=
        "The total number of patient reviews counted from this source.")
    link: str = Field(description="URL to the original review page.")


class ApiEnrichedProfileData(BaseModel):
    """The final structured data object to be extracted by the LLM."""
    years_experience: int = Field(
        description=
        "Total years of clinical practice since residency/fellowship completion, calculated by LLM."
    )
    bio_text_consolidated: str = Field(
        description=
        "Comprehensive biographical paragraph summarizing the doctor's experience, education, and past/current medical focus."
    )
    publications: List[str] = Field(
        description=
        "A list of titles of 3-5 key professional publications or research papers."
    )
    ratings_summary: List[RatingRecord] = Field(
        description=
        "List of structured rating records from all unique platforms found.")

    testimonial_summary_text: str = Field(
        description=
        "Summary of key patient testimonials and overall feedback to help new patients"
    )
    practice_address: str = Field(
        description=
        "The full street address of the doctor's primary practice location.")
    practice_phone: str = Field(
        description="The primary practice phone number.")

    latitude: float = Field(
        description=
        "The decimal latitude coordinate of the primary practice location.")
    longitude: float = Field(
        description=
        "The decimal longitude coordinate of the primary practice location.")
    education: List[str] = Field(
        description="Medical schools, residencies, fellowships.")
    hospitals: List[str] = Field(
        description="Current hospital or clinical affiliations.")
    certifications: List[str] = Field(description="Board certifications.")


# ------ helpers ---------


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_days(v, default=30):
    try:
        return int(v)
    except Exception:
        return default


def maybe_list(v) -> t.List[str]:
    if v is None: return []
    if isinstance(v, (list, tuple)): return [str(x) for x in v]
    # accept comma/space separated
    return [p for p in str(v).replace(",", " ").split() if p]


def build_pubmed_links(pmids: t.Iterable[str]) -> t.List[str]:
    return [f"https://pubmed.ncbi.nlm.nih.gov/{p}/" for p in pmids if p]


def _clean_llm_artifacts(text: str) -> str:
    """
    Strips out known artifacts (like token indices) that the LLM occasionally 
    embeds into text fields, which corrupt the output. (Copied from gemini_client.py)
    """
    if not text:
        return text

    # Pattern 1: Matches [INDEX 1, 2, 3, ...] or [INDEX 1] artifacts
    text = re.sub(r'\[INDEX\s+\d+(?:,\s*\d+)*\]',
                  '',
                  text,
                  flags=re.IGNORECASE)

    # Pattern 2: Matches artifact text like INDEX_1256 that occasionally appears
    text = re.sub(r'INDEX_\d+', '', text, flags=re.IGNORECASE)

    return text.strip()


# ------- Vertex / Gemini helpers ---------

_GCP_PROJECT = None
_GCP_LOCATION = None
_GEMINI_CLIENT: t.Optional[genai.Client] = None


def init_vertex(gcp_project: str, gcp_location: str):
    """
    Initializes the required GCP client configurations (Gen AI Client pointing to Vertex).
    """
    global _GCP_PROJECT, _GCP_LOCATION, _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        _GCP_PROJECT = gcp_project
        _GCP_LOCATION = gcp_location

        _GEMINI_CLIENT = genai.Client(
            vertexai=True,
            project=gcp_project,
            location=gcp_location,
        )
        log.info(
            f"Google Gen AI Client (Vertex mode) initialized for project {_GCP_PROJECT} in {_GCP_LOCATION}."
        )


def _call_gemini_structured_grounded(
        prompt_instruction: str, schema: dict, model_name: str
) -> t.Tuple[t.Dict[str, Any], t.List[t.Dict[str, str]]]:
    """
    Internal function to call Gemini with Google Search Grounding and JSON Schema.
    """
    if _GEMINI_CLIENT is None:
        raise RuntimeError("Gen AI Client must be initialized.")

    empty_result = {}, []

    config = types.GenerateContentConfig(
        max_output_tokens=16384,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.1,
    )

    try:
        response = _GEMINI_CLIENT.models.generate_content(
            model=model_name,
            contents=prompt_instruction,
            config=config,
        )
    except APIError as e:
        log.severe(f"Gemini API call failed. Error: {e}")
        return empty_result
    except Exception as e:
        log.severe(f"Gemini API call failed (Unexpected): {e}")
        return empty_result

    if not response.candidates:
        log.warning(f"Gemini returned no candidates.")
        return empty_result

    candidate = response.candidates[0]
    json_str = candidate.content.parts[0].text.strip()

    # Pydantic validation and artifact cleaning
    if not json_str:
        log.warning(
            f"Gemini returned empty JSON string. Reason: {candidate.finish_reason.name}"
        )
        return empty_result

    json_str = _clean_llm_artifacts(json_str)

    try:
        extracted_dict = ApiEnrichedProfileData.model_validate_json(
            json_str).model_dump()

        extracted_dict['bio_text_consolidated'] = _clean_llm_artifacts(
            extracted_dict.get('bio_text_consolidated', ''))
        extracted_dict['testimonial_summary_text'] = _clean_llm_artifacts(
            extracted_dict.get('testimonial_summary_text', ''))

    except Exception as e:
        log.severe(
            f"Pydantic validation failed for Grounding output: {e}. Raw JSON: {json_str[:200]}..."
        )
        return empty_result

    # Extract grounding metadata for sources
    sources = []
    if candidate.grounding_metadata:
        source_list_container = getattr(candidate.grounding_metadata,
                                        'attributions', None)
        if source_list_container:
            for attribution in source_list_container:
                if attribution.web:
                    sources.append({
                        'url': attribution.web.uri,
                        'title': attribution.web.title,
                    })

    # no need sources list for final BQ here
    return extracted_dict, sources


def embed_text(text: str,
               model_name: str = "text-embedding-004") -> t.List[float]:
    """Uses the genai.Client for text embedding."""
    if _GEMINI_CLIENT is None:
        # Allow this to run even if the client wasn't fully initialized if we only want embedding
        # but in this connector, init_vertex runs first.
        log.severe("Gen AI Client not initialized for embedding.")
        return [0.0] * 3072

    embedding_model_name = model_name

    try:
        response = _GEMINI_CLIENT.models.embed_content(
            model=embedding_model_name,
            contents=text,
        )

        embedding_value = response.embedding.values if response.embedding else []
        return embedding_value

    except APIError as e:
        log.severe(f"Gen AI Embedding failed (APIError): {e}")
        return [0.0] * 3072
    except Exception as e:
        log.severe(f"Gen AI Embedding failed (General): {e}")
        return [0.0] * 3072


def enrich_profile_with_llm(
        doctor: t.Dict[str, Any],
        summary_model: str) -> t.Tuple[t.Dict[str, Any], str]:
    """
    The main enrichment logic that performs the grounded, structured extraction.
    Returns: (extracted_pydantic_dict, text_to_embed)
    """
    name = f"{doctor['first_name']} {doctor['last_name']}"
    specialty = doctor.get('primary_specialty', 'Physician')

    prompt = (f"""
        As a medical data expert, find the official profile and review information for 
        Dr. {name}, a specialist in {specialty}. Extract their years of experience, 
        average patient ratings, certifications, education, hospital affiliations, and key professional publications.
        Provide comprehensive biographical summary of the doctor's experiences, education, and medical focus including research, conferences, and talks.
        Find the doctor's primary practice location, its **full street address**, and its **phone number**.
        For the 'education' list, include the names of ALL medical schools, residencies, and fellowships.
        For the 'hospitals' list, include the names of ALL hospital and major clinical affiliations.
        For example:
        Dr. Jane Smith (Cardiology)
        education: ["Harvard Medical School (MD, 2005)", "Massachusetts General Hospital (Residency)"]
        hospitals: ["Massachusetts General Hospital", "Brigham and Women's Hospital"]
        certifications: ["Board Certified in Cardiology"]
        years_experience: 18
        average_rating: 4.7
        practice_address: 212 W 51st, New York, NY
        practice_phone: 7178952314
        publications: ["Cardiac Outcomes in Post-Surgical Patients", "Advances in Echocardiography"]
        Now, repeat this process for Dr. {name}.
        Find the doctor's primary practice location and use Google Search to find its **precise latitude and longitude coordinates**.
        
        Summarize key patient testimonials to help new patients make informed decisions.
        
        Use Google Search as your tool. Calculate years of experience from their graduation or residency end date.
        """)

    # Get the JSON schema for the extraction
    schema = ApiEnrichedProfileData.model_json_schema()

    # structured extraction with grounding
    extracted_dict, _ = _call_gemini_structured_grounded(
        prompt_instruction=prompt, schema=schema, model_name=summary_model)

    # Determine the text to embed (fallback to a basic string if extraction failed)
    text_to_embed = extracted_dict.get(
        'bio_text_consolidated',
        f"{name} specializing in {specialty} in {doctor['city']}")

    return extracted_dict, text_to_embed


# -----------------------
# BigQuery helpers
# -----------------------
def get_bq_client_for_connector(project: str) -> bigquery.Client:
    """
    Initializes the BigQuery client for the Fivetran connector.
    """
    try:
        client = bigquery.Client(project=project)
        return client
    except Exception as e:
        log.severe(
            f"Failed to initialize BigQuery client for project {project}: {e}")
        raise RuntimeError(f"BigQuery client initialization failed: {e}")


def query_seed_doctors(
    client: bigquery.Client,
    project: str,
    dataset: str,
    npi_table: str,
    updated_since_days: int,
    specialties: t.List[str],
    state: str,
    max_doctors: int,
):
    """
    Pull a small seed set of NPIs to enrich from the NPI table already loaded by Connector A.
    """
    since_ts = (datetime.now() -
                timedelta(days=updated_since_days)).isoformat() + "Z"
    specialties_list = ",".join([f"'{s}'" for s in specialties])

    sql = f"""
    -- Query pulls doctors from NPI table, filtered by recency, state, and specialty
    -- NOTE: Removing complex JSON extraction for address, relying on LLM to find it.
    WITH base AS (
      SELECT
        t1.npi,
        t1.first_name,
        t1.last_name,
        t1.primary_specialty_desc,
        t1.city,
        t1.state,
        t1.zip,
        t1.last_updated_at
        -- Removed: address_line_1, address_line_2, phone JSON extractions
      FROM `{project}.{dataset}.{npi_table}` AS t1
      WHERE t1.state = @state
        AND t1.primary_specialty IN ({specialties_list})
        AND (t1.last_updated_at IS NULL OR t1.last_updated_at >= @since)
    )
    SELECT *
    FROM base
    -- Best Practice: Order by oldest records first to enrich low-priority/old data first.
    ORDER BY last_updated_at ASC NULLS FIRST
    LIMIT @limit
    """
    try:
        job = client.query(
            sql,
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("state", "STRING", state),
                bigquery.ScalarQueryParameter("since", "STRING", since_ts),
                bigquery.ScalarQueryParameter("limit", "INT64", max_doctors),
            ]),
        )
        return list(job.result())
    except Exception as e:
        log.severe(f"Failed to execute BQ seed query: {e}")
        # Critical failure: stop sync if seed data cannot be retrieved
        raise RuntimeError("BQ seed query failed.")


def load_pmids_for(client: bigquery.Client,
                   project: str,
                   dataset: str,
                   provider_publications_table: str,
                   npi: str,
                   max_pmids: int = 25) -> t.List[str]:
    """
    Optional: attach PubMed links from your PubMed/ORCID connector output.
    """
    sql = f"""
    SELECT t2.pmid
    FROM `{project}.{dataset}.{provider_publications_table}` AS t1
    JOIN `{project}.{dataset}.pubmed_articles` AS t2 ON t1.pmid = t2.pmid
    WHERE t1.npi = @npi
    LIMIT @limit
    """
    try:
        job = client.query(
            sql,
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("npi", "STRING", npi),
                bigquery.ScalarQueryParameter("limit", "INT64", max_pmids),
            ]),
        )
        return [row["pmid"] for row in job.result()]
    except Exception as e:
        log.warning(f"Failed to fetch PMIDs for NPI {npi}: {e}")
        return []


# Fivetran SDK functions


def schema(configuration: dict):
    # One wide table for your app to consume
    return [
        {
            "table": "doctor_profiles",
            "primary_key": ["npi"],
        },
    ]


def update(configuration: dict, state: dict):
    log.info("profiles_enrichment: start")

    bq_project = configuration.get("bq_project")
    bq_dataset = configuration.get("bq_dataset")
    bq_npi_table = configuration.get("bq_npi_table", "npi_providers")

    if not (bq_project and bq_dataset):
        raise ValueError(
            "profiles_enrichment requires 'bq_project' and 'bq_dataset'")

    # --- Vertex config
    gcp_project = configuration.get("gcp_project") or bq_project
    gcp_location = configuration.get("gcp_location", "us-central1")
    embedding_model = configuration.get("embedding_model",
                                        "gemini-embedding-001")
    summary_model = configuration.get("summary_model", "gemini-2.5-flash")

    # --- Selection knobs / usage caps
    state_filter = configuration.get("state_filter", "NY")
    specialties = maybe_list(
        configuration.get("specialties",
                          "Reproductive Endocrinology,Orthopaedic Surgery"))
    updated_since = parse_days(configuration.get("updated_since_days", 30))
    max_doctors = int(configuration.get("max_doctors_per_sync", 200))
    # attach_pmids = bool(configuration.get("attach_pubmed_links", True))
    # max_pmids = int(configuration.get("max_pmids_per_doctor", 25))
    # Placeholder URL for profile picture
    PROFILE_PIC_URL = 'https://storage.googleapis.com/smarterdoc-profile-media-bucket/headshots/12345.png'

    # --- Initialize clients
    client = get_bq_client_for_connector(bq_project)
    init_vertex(gcp_project, gcp_location)

    # --- Pull a small seed set from NPI table
    rows = query_seed_doctors(
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        npi_table=bq_npi_table,
        updated_since_days=updated_since,
        specialties=specialties,
        state=state_filter,
        max_doctors=max_doctors,
    )
    log.info(
        f"profiles_enrichment: retrieved {len(rows)} doctors for processing.")

    processed = 0
    for row in rows:
        # base fields
        npi = str(row["npi"])
        first_name = row["first_name"]
        last_name = row["last_name"]
        primary_specialty = row["primary_specialty"]
        city = row["city"]
        state_abbr = row["state"]
        zip_code = row["zip"]

        # Initialize final record data
        enriched_data: t.Dict[str, Any] = {}
        vector: t.List[float] = [0.0] * 3072

        # 2) Perform LLM Structured Enrichment (with Grounding)
        try:
            extracted_data, text_to_embed = enrich_profile_with_llm(
                row, summary_model)

            if not extracted_data:
                log.warning(
                    f"Structured extraction failed for NPI={npi}. Skipping LLM steps."
                )
                continue  # Skip to next doctor

            enriched_data = extracted_data

        except Exception as e:
            log.severe(f"FATAL LLM enrichment failure for NPI={npi}: {e}")
            continue  # Skip to next doctor

        # --- 2) Generate Embedding Vector (Multi-Field Vectorization) ---
        # CHANGE: Build the Composite Text for rich embedding (publications are now in extracted_data)
        composite_embed_text = " ".join(
            filter(None, [
                f"SPECIALTY: {primary_specialty}",
                f"BIO: {extracted_data.get('bio_text_consolidated', '')}",
                f"SUMMARY: {extracted_data.get('testimonial_summary_text', '')}",
                f"PUBS: {', '.join(extracted_data.get('publications', []))}",
                f"CERTS: {', '.join(extracted_data.get('certifications', []))}",
                f"EDUCATION: {', '.join(extracted_data.get('education', []))}",
                f"HOSPITALS: {', '.join(extracted_data.get('hospitals', []))}"
            ]))

        try:
            vector = embed_text(composite_embed_text, embedding_model)
        except Exception as e:
            log.warning(
                f"Embedding failed for NPI={npi}. Using fallback vector: {e}")
            vector: t.List[float] = [0.0] * 3072

        # --- 3) Publication Link Aggregation ---
        # Publications are now solely the array of strings extracted by
        pub_links: t.List[str] = enriched_data.get('publications', [])

        # Ensure ratings is a list of dicts (Pydantic converts them, but we ensure list consistency for BQ JSON type)
        ratings_list = [
            r.model_dump() if isinstance(r, BaseModel) else r
            for r in enriched_data.get('ratings_summary', [])
        ]

        # --- 4) Compose FINAL RECORD ---
        final_record = {
            "npi":
            npi,
            "first_name":
            first_name,
            "last_name":
            last_name,
            "primary_specialty":
            primary_specialty,
            "city":
            city,
            "state":
            state_abbr,
            "zip":
            zip_code,

            # LLM-EXTRACTED ADDRESS FIELDS MAPPED
            "practice_address":
            extracted_data.get("practice_address"),
            "practice_phone":
            extracted_data.get("practice_phone"),
            "bio":
            extracted_data.get('bio_text_consolidated', text_to_embed),
            "years_experience":
            extracted_data.get('years_experience'),
            "testimonial_summary_text":
            extracted_data.get('testimonial_summary_text'),
            "profile_picture_url":
            PROFILE_PIC_URL,
            "latitude":
            extracted_data.get('latitude'),
            "longitude":
            extracted_data.get('longitude'),
            "education":
            extracted_data.get('education', []),
            "hospitals":
            extracted_data.get('hospitals', []),
            "ratings":
            ratings_list,  # Array of JSON objects
            "publications":
            pub_links,  # Array of strings (Titles/URLs/Identifiers)
            "certifications":
            extracted_data.get('certifications', []),
            "embedding":
            vector,  # Array of floats
            "enriched_at":
            utc_now_iso(),
        }

        # --- 5) Upsert (Stream to Fivetran) ---
        op.upsert("doctor_profiles", final_record)
        processed += 1

    # Final checkpoint
    op.checkpoint(state={"last_run_at": utc_now_iso()})
    log.info(
        f"profiles_enrichment: processed={processed}, last_run_at={state.get('last_run_at')}"
    )


# SDK connector object
connector = Connector(update=update, schema=schema)

if __name__ == "__main__":
    try:
        with open("../main-configuration.json", "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}
    connector.debug(configuration=cfg)
