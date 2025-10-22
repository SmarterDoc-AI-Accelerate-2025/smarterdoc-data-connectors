# SmarterDoc AI Data Connectors (Fivetran SDK Pipeline)

This repository contains the Fivetran Custom Connector SDK code that powers the machine learning data pipeline for the SmarterDoc application. These connectors are responsible for extracting, enriching, and vectorizing healthcare provider data for use in recommendation and semantic search services.

With **Fivetran custom sdk**, we're able to build a highly optimized, three-stage pipeline designed to transform raw public data into a clean, vectorized, and AI-ready asset in BigQuery.The pipeline also ensures incremental synchronization of complex AI features (vectors and structured LLM output) into BigQuery.

---

## Pipeline Overview (3 Connectors, 2 Tables)

The pipeline runs in a sequential dependency chain:

1.  **Connector A: NPI Registry** (Source)
2.  **Connector B: Profiles Enrichment** (Source & Producer)
3.  **Connector C: Vector Indexer** (Batch Processor)

| Connector               | Source                | Destination Table     | Purpose                                                                                                                                      |
| :---------------------- | :-------------------- | :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **NPI Registry**        | CMS NPPES API         | `raw.npi_providers`   | Ingests raw provider metadata using sharding and incremental updates.                                                                        |
| **Profiles Enrichment** | `raw.npi_providers`   | `raw.doctor_profiles` | Executes Google Gemini (Grounding) and Embedding calls to generate bios, publications, and the initial `embedding` vector.                   |
| **Vector Indexer**      | `raw.doctor_profiles` | `raw.doctor_profiles` | Handles scheduled batch regeneration of the vector field (`embedding`) to support model migration (e.g., $768 \rightarrow 3072$ dimensions). |

---

## Configurable Fields for Adaptability

---

### NPI Connector

- **Time-Based Bookmark:** The connector saves the most recent `last_updated_at` timestamp from the NPI records it has successfully processed into the Fivetran `state` object. On the next sync, the connector only queries the NPI API for records that have been modified after that saved timestamp. This minimizes API calls and greatly reduces sync latency.
- **Sharding Resumption:** The connector saves a `last_processed_shard_key` in the checkpoint, ensuring that if a job fails, the Fivetran retry mechanism restarts the connector exactly from the last saved shard key, preventing the loss of progress and guaranteeing a robust sync.

### Profile Enrichment Connector

- **AI Asset Creation (Vectorization):** This connector calls the **Vertex AI Embedding service** to generate the $3072$-dimension `embedding` (vector) directly within the pipeline, treating it as a standard column. This transforms the data pipeline into a managed service for creating the core AI asset needed for vector search.
- **Structured Grounding:** The connector uses **Google Gemini (with Grounding)** and a strict **Pydantic schema** to extract and validate complex fields from the web, such as patient `ratings`, `education`, and `publications`. This ensures perfectly structured data is delivered straight to the `doctor_profiles` table.
- **Orchestration and Consumption Control:** The connector reads from the `raw.npi_providers` table and uses configurable limits (`max_doctors_per_sync`) to manage the volume of data processed per run, thereby controlling Vertex AI costs and sync duration.

### Vector Indexer Connector

- **Batch Re-indexing and Model Migration:** This maintenance connector is designed to handle model upgrades (e.g., switching from an older vector model to the $3072$-dimension `gemini-embedding-001`). It recalculates the embeddings for existing profiles without running the entire costly LLM enrichment process again.
- **Vector Incremental Sync:** It uses its own dedicated state (`last_vector_sync_time`) to process only those `doctor_profiles` that have been _enriched_ since the last vector maintenance job. This ensures the batch job is highly targeted and efficient.
- **Targeted Upsert:** The connector performs an `op.upsert()` that updates _only_ the `embedding` column in the `doctor_profiles` table, demonstrating granular control over the data asset.

---

## Connector Reusability and Domain Adaptability

This entire suite of Fivetran Custom Connectors is designed for reuse by any organization needing a structured, AI-ready database of healthcare providers from the [NPI (National Provider Identifier) Registry](https://npiregistry.cms.hhs.gov/search).

Users building a doctor database can leverage this pipeline by simply modifying the externalized configurations:

**Custom Sharding**: By changing the configuration strings for postal_prefixes and taxonomy_codes, users can instantly pivot the NPI Registry Connector to target providers in any state, region, or medical specialty, eliminating the need to rewrite the core API fetching logic.

**LLM Flexibility**: Users can easily adjust the summary_model (e.g., from gemini-2.5-flash to a newer model) and the embedding_model for vectorization, ensuring the pipeline remains compatible with evolving Generative AI services without redeploying code.

**Data Governance**: The clear separation of the raw NPI table and the final enriched doctor_profiles table ensures data lineage and governance are maintained, regardless of the target application.

---

## Configuration and Deployment

All connectors share configuration defined in the main `main-configuration.json` file.

### Prerequisites:

1.  **GCP Access:** Project must be authorized for Vertex AI and BigQuery access.
2.  **Fivetran Setup:** BigQuery Destination must be created and authorized in GCP IAM.

### Local Testing:

To run any connector locally against the configuration file:

```bash
# Example: Running the Enrichment Connector
cd connectors/profiles_enrichment
python connector.py
```

---

## Important Disclaimers and Usage Policy

#### Source Data and Liability:

- Public Information Only: All healthcare provider profile information (bios, publications, ratings, and addresses) extracted by the Profiles Enrichment Connector is gathered from publicly accessible internet sources using Google Search Grounding tools.
- No Verification Guarantee: The data is processed using Generative AI (LLMs) and is not guaranteed to be $100\%$ accurate, current, or verified.
- Non-Commercial Use: This Fivetran SDK connector pipeline and the resulting BigQuery database are intended strictly for demonstration, research, and non-commercial educational purposes (such as hackathons and developer challenges).
- Commercial Prohibition: This project is not licensed or intended for use in any commercial application that involves making medical decisions, validating credentials, providing patient referrals, or replacing official licensed data sources. Users assume all responsibility for any use outside of its stated academic/demonstration purpose.
