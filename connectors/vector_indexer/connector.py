##############################################################
# Fivetran Custom Connector SDK: Vector Search Indexer (Connector C)

# Purpose:
# This connector acts as a batch processor designed for maintenance. It reads
# enriched provider data from the `doctor_profiles` table, regenerates the dense
# embedding vector using the latest model (e.g., gemini-embedding-001), and
# upserts the new vector back into the same table. It is crucial for scheduled
# model migrations and re-indexing tasks.

# Best Practices Implementation:
# 1. Orchestration: Reads from and writes to the same BigQuery destination table (`doctor_profiles`).
# 2. Incrementality: Uses `last_vector_sync_time` state to process only records enriched since the last vector job.
# 3. Batching: Processes data in configurable batches (`batch_size`) to manage Vertex AI cost and throughput.
# 4. Resilience: Robust try/except blocks wrap high-latency calls to the Gemini embedding service.
#############################################################
import json
import time
import typing as t
from datetime import datetime, timezone, timedelta
import re

from fivetran_connector_sdk import Connector
from fivetran_connector_sdk import Operations as op
from fivetran_connector_sdk import Logging as log

from google.cloud import bigquery
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Configuration Constants ---
DEFAULT_BATCH_SIZE = "50"
DEFAULT_DIMENSION = "3072"


# --- State Management Helper ---
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# BigQuery Helpers
# ----------------------------------------------------------------------

# The BigQuery Client must be initialized here for the source query.
_BQ_CLIENT: t.Optional[bigquery.Client] = None


def get_bq_client_for_connector(project: str) -> bigquery.Client:
    """Initializes the BigQuery client."""
    global _BQ_CLIENT
    if _BQ_CLIENT is None:
        try:
            _BQ_CLIENT = bigquery.Client(project=project)
            log.info(f"BigQuery client initialized for project {project}.")
        except Exception as e:
            log.error(
                f"Failed to initialize BigQuery client for project {project}: {e}"
            )
            raise RuntimeError(f"BigQuery client initialization failed: {e}")
    return _BQ_CLIENT


def query_doctors_for_vector_indexing(
    client: bigquery.Client,
    project: str,
    dataset: str,
    table: str,
    last_vector_sync_time: t.Optional[str],
    batch_size: int,
):
    """
    Pulls records that have been enriched (or updated) since the last vector sync.
    """
    # Fallback to a very old timestamp if no state is found (full historical sync)
    since_ts = last_vector_sync_time or (
        datetime.now() - timedelta(days=9000)).isoformat() + "Z"

    sql = f"""
    -- Fetch NPIs and fields needed to generate the vector
    SELECT
        t1.npi,
        t1.primary_specialty_desc,
        t1.bio, -- LLM-extracted bio text
        t1.testimonial_summary_text,
        t1.publications,
        t1.certifications,
        t1.education,
        t1.hospitals,
        t1.enriched_at -- Use LLM enrichment time as the source change marker
    FROM `{project}.{dataset}.{table}` AS t1
    -- Filter 1: Only re-index if the profile has been enriched SINCE the last vector sync.
    WHERE t1.enriched_at >= @since
    -- Filter 2: Only doctors that have been enriched (i.e., bio is not NULL)
      AND t1.bio IS NOT NULL
    -- Best Practice: Process the oldest records first to catch up on backlog.
    ORDER BY t1.enriched_at ASC
    LIMIT @limit
    """
    log.info(f"Querying BQ for records enriched since: {since_ts}")

    try:
        job = client.query(
            sql,
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("since", "STRING", since_ts),
                bigquery.ScalarQueryParameter("limit", "INT64", batch_size),
            ]),
        )
        # Use job.to_dataframe().to_dict('records') for large results if necessary
        return [dict(row.items()) for row in job.result()]
    except Exception as e:
        log.error(
            f"Failed to execute BQ seed query for vector re-indexing: {e}")
        # Critical failure: stop sync
        raise RuntimeError("BQ vector seed query failed.")


# ----------------------------------------------------------------------
# Gemini/Vertex AI Helpers (Adapted from connector_profile.py)
# ----------------------------------------------------------------------

_GCP_PROJECT = None
_GCP_LOCATION = None
_GEMINI_CLIENT: t.Optional[genai.Client] = None


def init_vertex(gcp_project: str, gcp_location: str):
    """Initializes the Gen AI Client pointing to Vertex."""
    global _GCP_PROJECT, _GCP_LOCATION, _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        _GCP_PROJECT = gcp_project
        _GCP_LOCATION = gcp_location

        # Note: aiplatform import needed for credential handling in Vertex mode
        from google.cloud import aiplatform
        aiplatform.init(project=gcp_project, location=gcp_location)

        _GEMINI_CLIENT = genai.Client(
            vertexai=True,
            project=gcp_project,
            location=gcp_location,
        )
        log.info(
            f"Google Gen AI Client (Vertex mode) initialized for project {_GCP_PROJECT} in {_GCP_LOCATION}."
        )


def generate_embeddings_batch(
        texts: t.List[str],
        model_name: str = "text-embedding-004",
        dimension: int = DEFAULT_DIMENSION) -> t.List[t.List[float]]:
    """Generates a batch of embeddings using the Gen AI client."""
    if _GEMINI_CLIENT is None:
        log.error("Gen AI Client not initialized for embedding.")
        return [[0.0] * dimension] * len(texts)

    try:
        response = _GEMINI_CLIENT.models.batch_embed_content(
            model=model_name,
            contents=texts,
        )

        vectors = []
        for embedding in response.embeddings:
            vectors.append(embedding.values if embedding.values else [0.0] *
                           dimension)

        return vectors

    except APIError as e:
        log.error(f"Gen AI Batch Embedding failed (APIError): {e}")
        return [[0.0] * dimension] * len(texts)
    except Exception as e:
        log.error(f"Gen AI Batch Embedding failed (General): {e}")
        return [[0.0] * dimension] * len(texts)


def build_composite_text(doc: t.Dict[str, t.Any]) -> str:
    """
    Builds the composite text string by concatenating relevant doctor profile fields.
    This is the exact logic from your re_indexer.py.
    """

    def safe_join(field):
        value = doc.get(field)
        if isinstance(value, list):
            # Check for non-string types in list and convert/filter
            return ', '.join(filter(None, [str(x) for x in value]))
        # Use 'bio' instead of 'bio_text_consolidated' as that is the BQ column name
        return str(value) if value else ''

    composite_embed_text = " ".join(
        filter(None, [
            f"SPECIALTY: {safe_join('primary_specialty_desc')}",
            f"BIO: {safe_join('bio')}",
            f"SUMMARY: {safe_join('testimonial_summary_text')}",
            f"PUBS: {safe_join('publications')}",
            f"CERTS: {safe_join('certifications')}",
            f"EDUCATION: {safe_join('education')}",
            f"HOSPITALS: {safe_join('hospitals')}",
        ]))
    return composite_embed_text.strip()


# ----------------------------------------------------------------------
# Fivetran SDK Functions
# ----------------------------------------------------------------------


def schema(configuration: dict):
    """
    Defines the BigQuery destination table. We must declare all columns
    to ensure Fivetran knows the full structure for the upsert operation.
    """
    # NOTE: This connector only modifies the 'embedding' column, but we must
    # declare the full schema to operate on the existing table.
    return [
        {
            "table":
            configuration.get("bq_profile_table", "doctor_profiles"),
            "primary_key": ["npi"],
            # Define all columns explicitly to ensure type consistency, matching C's schema
            "columns": [
                {
                    "name": "npi",
                    "type": "string"
                },
                {
                    "name": "first_name",
                    "type": "string"
                },
                {
                    "name": "last_name",
                    "type": "string"
                },
                {
                    "name": "primary_specialty_desc",
                    "type": "string"
                },
                {
                    "name": "city",
                    "type": "string"
                },
                {
                    "name": "state",
                    "type": "string"
                },
                {
                    "name": "zip",
                    "type": "string"
                },
                {
                    "name": "practice_address",
                    "type": "string"
                },
                {
                    "name": "practice_phone",
                    "type": "string"
                },
                {
                    "name": "bio",
                    "type": "string"
                },
                {
                    "name": "years_experience",
                    "type": "int"
                },
                {
                    "name": "testimonial_summary_text",
                    "type": "string"
                },
                {
                    "name": "profile_picture_url",
                    "type": "string"
                },
                {
                    "name": "latitude",
                    "type": "float"
                },
                {
                    "name": "longitude",
                    "type": "float"
                },
                {
                    "name": "education",
                    "type": "array",
                    "item_type": "string"
                },
                {
                    "name": "hospitals",
                    "type": "array",
                    "item_type": "string"
                },
                {
                    "name": "ratings",
                    "type": "array",
                    "item_type": "json"
                },
                {
                    "name": "publications",
                    "type": "array",
                    "item_type": "string"
                },
                {
                    "name": "certifications",
                    "type": "array",
                    "item_type": "string"
                },
                {
                    "name": configuration.get("vector_attribute_name",
                                              "embedding"),
                    "type": "array",
                    "item_type": "float"
                },
                {
                    "name": "enriched_at",
                    "type": "timestamp"
                },
                # Add a new column to track when the vector was last updated
                {
                    "name": "vector_updated_at",
                    "type": "timestamp"
                }
            ]
        },
    ]


def update(configuration: dict, state: dict):
    """
    Runs the incremental vector re-indexing job.
    """
    log.info("vector_indexer: starting sync")

    # --- Config Extraction ---
    bq_project = configuration.get("bq_project")
    bq_dataset = configuration.get("bq_dataset")
    bq_profile_table = configuration.get("bq_profile_table", "doctor_profiles")
    vector_attr = configuration.get("vector_attribute_name", "embedding")

    gcp_project = configuration.get("gcp_project") or bq_project
    gcp_location = configuration.get("gcp_location", "us-central1")
    embedding_model = configuration.get("embedding_model",
                                        "text-embedding-004")
    embedding_dimension = int(
        configuration.get("embedding_dimension", DEFAULT_DIMENSION))
    batch_size = int(configuration.get("batch_size", DEFAULT_BATCH_SIZE))

    # --- State Management: Incremental Sync Marker ---
    # last_vector_sync_time tracks when the last vector sync completed successfully.
    # This is the key for incremental sync.
    last_vector_sync_time = state.get("last_vector_sync_time")

    if not (bq_project and bq_dataset):
        raise ValueError(
            "vector_indexer requires 'bq_project' and 'bq_dataset'")

    # --- Initialize clients ---
    client = get_bq_client_for_connector(bq_project)
    init_vertex(gcp_project, gcp_location)

    total_processed = 0
    # Process the first batch, then loop until no more records are returned
    while True:
        # Get the current time to set as the high-water mark for the *next* sync
        current_sync_time = utc_now_iso()

        # 1. Pull Batch of Data from BQ (Incremental)
        rows = query_doctors_for_vector_indexing(
            client=client,
            project=bq_project,
            dataset=bq_dataset,
            table=bq_profile_table,
            last_vector_sync_time=last_vector_sync_time,
            batch_size=batch_size,
        )

        if not rows:
            log.info(
                "No new enriched doctors found for re-indexing. Finishing.")
            break

        log.info(f"Retrieved {len(rows)} doctors for vector generation.")

        # 2. Prepare Texts for Batch Embedding
        texts_to_send = [build_composite_text(row) for row in rows]

        # 3. Generate Embeddings (The high-cost step)
        try:
            new_vectors = generate_embeddings_batch(
                texts_to_send,
                model_name=embedding_model,
                dimension=embedding_dimension)
        except Exception as e:
            log.error(f"Fatal batch embedding error. Aborting sync: {e}")
            raise  # Stop the sync so Fivetran retries later

        # 4. Prepare and Upsert Records
        records_to_upsert = []
        for i, row in enumerate(rows):
            vector = new_vectors[i]

            # The record only needs the PK (npi), the new vector, and the update time
            # The vector_attr is dynamically pulled from config (e.g., 'embedding')
            records_to_upsert.append({
                "npi": str(row["npi"]),
                vector_attr: vector,
                "vector_updated_at":
                current_sync_time  # Mark when the vector was created
            })

            # 5. Upsert to Fivetran Destination (Streaming)
            op.upsert(bq_profile_table, records_to_upsert[-1])
            total_processed += 1

        # Checkpoint is performed after successfully processing a batch
        # Update the state to the enrichment time of the LATEST record processed in this batch
        # This ensures we don't skip records if the batch contains records with the same enriched_at time
        latest_enriched_at = max(row["enriched_at"] for row in rows)

        op.checkpoint(state={"last_vector_sync_time": latest_enriched_at})
        log.info(
            f"Batch completed. Upserted {len(rows)} vectors. Checkpoint set to: {latest_enriched_at}"
        )

        # If the number of rows is less than the batch size, it means we reached the end of the available data
        if len(rows) < batch_size:
            break

    # Final checkpoint is implied by the last batch checkpoint, but we log the total
    log.info(
        f"vector_indexer: Sync complete. Total processed: {total_processed} doctors."
    )


connector = Connector(update=update, schema=schema)

if __name__ == "__main__":
    try:
        with open("../main-configuration.json", "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}
    connector.debug(configuration=cfg)
