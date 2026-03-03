# Reflection — Construction Arbitration Document Intelligence Pipeline

## 1. Scaling from 50 Segments to 500 Segments and 200 Active Cases

With 50 segments, the pipeline runs smoothly in a single process with thread pooled execution model. However, scaling to:

- 500 segments per case
- 200 concurrent active cases

introdces architectural pressure.

### What Changes

**a) Concurrency Model**

Segment-level parallelism (ThreadPoolExecutor) becomes insufficient.  
At scale, we need:

- Distributed task queues (e.g., message broker + worker fleet)
- Per-case isolation
- Rate-limit-aware LLM scheduling

LLM calls would need to be throttled across cases to avoid API bottlenecks.

**b) State Based, Per-Segment Processing Model**

Phases 2 and 3 (Step 2 in each) are designed as **per-segment state transitions**, not monolithic batch jobs.

Each segment independently transitions through:

- Indexed → Classified (Phase 2)
- Indexed → Claim-Mapped (Phase 3)

Outputs are written in a flattened, segment-scoped structure
(e.g., `SegmentClaimMapping` rows).

This design has several scaling advantages:

- A failure in one segment does not fail the entire phase
- Work can be retried at segment granularity
- Horizontal workers can process segments independently
- Partial results can be safely persisted

At 50 segments, this runs inside a ThreadPoolExecutor.
At 500 segments, this model maps cleanly to a distributed task queue.


**c) Storage and Checkpointing**

Currently, outputs are file-based JSON artifacts.

At scale:

- Replace file outputs with object storage (S3/GCS)
- Store structured outputs in a relational database
- Maintain per-segment and per-phase status tracking

Checkpointing becomes database-driven rather than file-driven.


### What breaks first

The likely first pressure points:

- LLM API rate limits
- Latency accumulation in Phase 3 Step 2
- Memory growth in Phase 4 coverage matrix generation

The current architecture is modular and clean,
but horizontal scalability requires distributed orchestration.

## 2. Parallel Execution of Phase 2 and Phase 3

Yes, Phases 2 and 3 were implemented to run in parallel using a fork-join model:

- Both depend only on Phase 1 output
- Each phase internally runs Step 1 → Step 2 sequentially
- A shared ThreadPoolExecutor (max_workers=5) limits LLM concurrency

This structure provides:
- Controlled parallelism
- Rate-limit safety
- Deterministic outputs


## 3. Redesign for Serverless (Stateless Containers)

### What Breaks

1. **Long-lived orchestrator process** — `main()` in `orchestrator.py` runs all phases within a single process. Serverless containers have hard execution time limits (typically 5–15 min). A full case with 500 segments would exceed this.

2. **`ThreadPoolExecutor` parallelism** — The shared thread pool (`max_workers=5`) assumes a persistent process. Serverless containers freeze or terminate between invocations; background threads are killed.

3. **Local filesystem checkpointing** — Patterns like `output_path.exists()` across Phase 1–3 assume persistent local disk. Serverless containers have ephemeral filesystems that are wiped between invocations.

4. **In-memory state sharing** — Phase 2 and Phase 3 fork-join coordination currently relies on `concurrent.futures` within a single process. No shared memory exists across serverless invocations.

### Redesign

**1. Replace the monolithic orchestrator with a workflow engine**
- Use AWS Step Functions, GCP Workflows, to define the phase graph declaratively.
- Each phase step becomes a separate serverless function invocation.
- The workflow engine handles sequencing (Phase 1 → Phase 2/3 → Phase 4), fan-out, fan-in, and retries.

**2. Replace filesystem state with external storage**

- OCR inputs → object storage (S3/GCS bucket).
- Structured outputs (index, classifications, claim mappings) → relational database or document store.
- Final reports → object storage.
- Checkpointing becomes a database status column (`pending`, `in_progress`, `complete`, `failed`) per segment per phase.

**3. Decompose into per-segment serverless functions**

- Phase 1: One invocation per segment (`extract_segment_index` is already stateless and segment-scoped).
- Phase 2 Step 2: One invocation per segment (`_classify_one` is already isolated).
- Phase 3 Step 2: One invocation per segment (per-segment claim mapping is already independent).
- Phase 2 Step 1 and Phase 3 Step 1: Single invocations (taxonomy discovery, claim extraction).
- Phase 4: Split into per-claim-head analysis invocations + one final assembly invocation.


## 4. Prompt Engineering Decisions for Messy OCR Text

The following prompt engineering decisions were made to handle this:


**1. Instructed the LLM to act as a domain-expert reader, not a literal parser**

- Prompts explicitly frame the task as interpretation, not extraction.
- Example: "You are a construction arbitration expert reviewing scanned tribunal documents" — this primes the model to tolerate OCR noise and infer meaning from context rather than relying on exact string matches.

**2. Provided the full OCR text as-is, without pre-cleaning**

- No regex-based cleanup or heuristic preprocessing was applied to the OCR text before sending it to the LLM.
- LLMs handle noisy natural language better than rule-based cleaners. Pre-cleaning risks stripping meaningful content (e.g., table fragments, currency symbols, legal references that look like artifacts).

**3. Used structured output schemas (Pydantic models) to force clean responses**

- Every LLM call returns a Pydantic-validated JSON object (e.g., `SegmentIndex`, `SegmentClassification`, `SegmentClaimMapping`).
- This forces the LLM to distill messy OCR into clean, typed fields — the schema acts as a noise filter.
- If the LLM hallucinates or returns malformed JSON, Pydantic validation catches it immediately.

**4. Included explicit instructions to handle common OCR failure modes**

- Prompts include guidance like: "The text may contain OCR artifacts, broken tables, or garbled characters. Extract the best interpretation."
- For tables: instructed the model to reconstruct tabular data logically even if column alignment is lost.
- For mixed languages: instructed the model to identify and preserve non-English content (e.g., Arabic party names, project references) without attempting translation.

**5. Few-shot examples in taxonomy and classification prompts**

- Phase 2 Step 1 (taxonomy discovery) and Step 2 (classification) include examples of expected output format.
- This anchors the LLM's output structure even when the input is noisy.

### Where the LLM Struggled

**1. Tabular financial data**
- OCR frequently breaks table structures — columns merge, rows split across lines, currency values lose decimal points.
- The LLM sometimes misattributes amounts to wrong line items or invents column headers.
- This is most impactful in Phase 3 (claim mapping), where quantum values matter.

**2. Lengthy Documents**
- For around 5 documents in this set, the character length reached upto 600k. 
- This became a problem generating the json as it would hit the max token limit

**3. Broken Dates and Monetary Amounts**
- The OCR text for various key dates and monetary amounts was malformed.
- This made the LLM come up with suspicious and highly infalted numbers that didn't fit in this case. 

**4. Confidence Calibration**
- The LLM does not reliably signal when OCR quality is too poor to extract meaningful content.
- It tends to produce plausible-sounding outputs even when the source text is largely unreadable — a known LLM failure mode (hallucination under uncertainty).


## 5. Use of AI Tools During the Assignment

I used AI tools primarily for:

- Architectural brainstorming
- Prompt refinement
- Edge-case reasoning
- Validating design trade-offs
- Pydantic schema design
- Code review and debugging

What worked well:
- Iterative prompt refinement for claim abstraction
- Parallel orchestration design patterns
- Identifying upstream/downstream data quality coupling

What did not work well:
- Overly verbose auto-generated plans
- Tendency toward overengineering
- Claim head generation quality
- Confidence score differentiation

