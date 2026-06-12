# CSE 590A Mini-Project: 4-5 Hour Execution Plan

**Project:** Autonomous Post-Discharge Patient Follow-Up Agent
**Authors:** Sai Sunku, Salma Hajian
**Stack:** vLLM + Llama-3.1-8B-Instruct on Colab A100 GPU + SQLite + Synthea Coherent

---

## 0. Strategic deviations from the proposal (state these in the paper)

1. **Data:** Switch from MIMIC-IV Demo (excludes free-text notes per PhysioNet README) to **Synthea Coherent Data Set** (synthetic, open-access, includes SOAP-style notes + full longitudinal FHIR history). Justify as: "real MIMIC-IV-Note requires credentialed access not feasible in our project window; Synthea provides equivalent structure under an open license."
2. **Hardware:** Switch from Cloud TPU to **Colab A100 GPU**. Systems analysis principles (prefill/decode dichotomy, prefix caching, continuous batching) are hardware-agnostic. State this in the paper.
3. **Scope:** Three workers instead of five (medication, problem-list, follow-up plan) to keep the build tractable. Drop separate red-flag and patient-instruction agents; fold their key behaviors into the follow-up worker.

---

## 1. Time budget

| Phase | Time | Output |
|---|---|---|
| 1. Setup + data | 45 min | Colab notebook, vLLM running, SQLite DB loaded |
| 2. Agent implementation | 90 min | Working orchestrator + 3 workers + tool calls |
| 3. Benchmarking | 60 min | Latency/quality numbers, 3 configs |
| 4. Paper writing | 90 min | 6-page draft |
| **Total** | **~4.75 h** | |

If anything goes sideways, **cut workers before cutting benchmarks** — a clean 2-worker pipeline with rigorous numbers beats a 5-worker mess.

---

## 2. Data setup (45 min, first thing)

### 2.1 Download Synthea sample

Use the pre-generated 100-patient sample from MITRE (not the 9GB Coherent set — way too big). Direct URL:

```bash
wget https://synthea.mitre.org/downloads/synthea_sample_data_csv_apr2020.zip
unzip synthea_sample_data_csv_apr2020.zip -d synthea_data/
```

This gives you CSV files: `patients.csv`, `encounters.csv`, `conditions.csv`, `medications.csv`, `observations.csv`, `procedures.csv`, `careplans.csv`. Far easier to work with than FHIR JSON.

### 2.2 Build SQLite history database

Single Python script: `build_db.py` loads CSVs into a SQLite database with these tables:
- `patients(id, birthdate, gender, ...)`
- `encounters(id, patient_id, start, stop, encounter_class, reason_description)`
- `conditions(patient_id, start, stop, code, description)`
- `medications(patient_id, start, stop, code, description, reason_description)`
- `observations(patient_id, date, code, description, value, units)`  (this is your labs/vitals)

Filter to patients with **≥3 encounters and ≥5 medication records** to ensure meaningful history. You'll have plenty.

### 2.3 Generate the "discharge note" inputs

Synthea's CSVs don't include free-text discharge summaries directly. Two options, **pick one**:

**Option A (faster, recommended):** Generate templated discharge notes from structured data using a Python template. For each patient's most recent inpatient encounter, fill a template:

```
DISCHARGE SUMMARY
Patient: [age]-year-old [gender]
Admission: [start_date] - [stop_date]
Chief Complaint: [reason_description from encounters]

History of Present Illness:
The patient presented with [conditions started at this encounter].
[narrative from observations]

Hospital Course:
[procedures performed]
[medications started]

Discharge Diagnosis:
[active conditions]

Discharge Medications:
[active medications at discharge]

Discharge Plan:
Follow-up in [varies] weeks with primary care.
```

This gives 2-4K tokens of input per note — perfect for the prefix-caching story. Build a target of **30 discharge notes** total.

**Option B (slower, more realistic):** Use a frontier LLM (Claude API if you have credits) to write notes from the structured data. Skip if tight on time.

### 2.4 Quick validation

Pick 3 patients. Manually verify:
- The note contains medications that are in the SQL `medications` table for that patient
- The note's conditions match the SQL `conditions` table
- The patient has ≥2 prior encounters for the tools to find

That's your dataset.

---

## 3. Architecture

### 3.1 File structure

```
project/
├── build_db.py              # Load Synthea CSVs → SQLite
├── generate_notes.py        # Build 30 discharge notes
├── tools.py                 # SQL-backed tool functions
├── agent.py                 # Orchestrator + workers
├── benchmark.py             # Run all configs, collect metrics
├── evaluate.py              # Compute application-level metrics
├── data/
│   ├── synthea.db
│   ├── notes/               # 30 .txt discharge notes
│   └── ground_truth.json    # For evaluation
├── results/
│   ├── metrics.csv
│   └── plots/
└── paper.md
```

### 3.2 Tools (in `tools.py`)

Five Python functions, each ~10 lines wrapping a SQL query. Each returns a structured dict.

```python
def get_medication_history(patient_id, lookback_days=365):
    """Return list of medications the patient was on, with start/stop dates."""

def get_lab_trend(patient_id, lab_name, lookback_days=180):
    """Return time-series of values for a lab/vital."""

def get_prior_diagnoses(patient_id):
    """Return list of active and resolved conditions."""

def get_prior_encounters(patient_id, n=5):
    """Return recent encounter summaries."""

def check_drug_interactions(drug_list):
    """Mock or call RxNav. For speed, mock with a static interaction dictionary
       of common pairs (warfarin+NSAIDs, ACE-i+K-sparing, etc.). Mock is fine."""
```

**Critical: mock the RxNav call.** Real RxNav adds 100-300ms per call and won't change the systems story. Mock it with a 5-entry hardcoded interaction table. Acknowledge this in the limitations section.

### 3.3 Agent architecture

```
                    [Discharge Note + Patient ID]
                                |
                                v
                        [Orchestrator]
                  (parses note, identifies patient_id,
                   decides tools to call upfront)
                                |
                                v
                    [Tool Execution Phase]
              (parallel: 4-5 SQL queries via asyncio)
                                |
                                v
                    [Shared context built]
                                |
                    +-----------+-----------+
                    v           v           v
              [Medication]  [Problem]  [Follow-up]
                Worker      Worker     Worker
              (parallel, all see shared context)
                    \\          |          /
                     \\         |         /
                      v         v        v
                       [Final Assembler]
                    (combines into structured plan)
                                |
                                v
                       [Validation/Reflection]
                  (checks consistency, may rerun a worker)
                                |
                                v
                    [Structured Follow-Up Plan]
```

### 3.4 Key implementation choice — prompt structure for prefix caching

This is the single most important code-level decision. **All three workers must share an identical prompt prefix.** Structure each worker's prompt as:

```
[SHARED PREFIX — identical across all workers]
You are part of a post-discharge follow-up system.
Here is the patient's discharge note:
<<DISCHARGE_NOTE>>

Here is the retrieved patient history:
<<TOOL_RESULTS>>

[WORKER-SPECIFIC SUFFIX]
Your role: medication reconciliation.
Output a JSON object with fields: ...
```

vLLM's automatic prefix caching keys on exact token-prefix match. If you accidentally vary even one token in the shared prefix (different system prompt per worker, different formatting), the cache misses. Verify by checking vLLM's `gpu_prefix_cache_hit_rate` metric — should be high (>60%) on workers 2 and 3.

### 3.5 Worker outputs (JSON schema, use `guided_json`)

```python
medication_output = {
    "discharge_medications": [{"name": str, "dose": str, "indication": str, "status": "new"|"continued"|"changed"}],
    "reconciliation_issues": [{"issue": str, "severity": "low"|"medium"|"high"}],
    "interaction_warnings": [str]
}

problem_output = {
    "active_problems": [{"name": str, "type": "chronic"|"acute"|"new"}],
    "resolved_problems": [str]
}

followup_output = {
    "appointments": [{"specialty": str, "timeframe_days": int, "reason": str}],
    "monitoring": [{"what_to_watch": str, "when_to_call": str}],
    "red_flags": [str]
}
```

Use vLLM's `guided_json` parameter with these schemas. Guarantees parseable output.

---

## 4. vLLM serving setup (15 min of the first 45)

### 4.1 Colab notebook setup

```python
# Cell 1: install
!pip install vllm==0.6.3
!pip install outlines  # for guided JSON

# Cell 2: launch vLLM server in background
import subprocess
server = subprocess.Popen([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "meta-llama/Llama-3.1-8B-Instruct",
    "--dtype", "bfloat16",
    "--max-model-len", "8192",
    "--enable-prefix-caching",  # toggle for ablation
    "--port", "8000",
    "--gpu-memory-utilization", "0.9",
])
# wait ~60s for server boot
```

### 4.2 Client uses OpenAI-compatible API

```python
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Parallel workers via asyncio.gather:
results = await asyncio.gather(*[
    call_worker(client, "medication", note, tools_results),
    call_worker(client, "problem", note, tools_results),
    call_worker(client, "followup", note, tools_results),
])
```

### 4.3 Metrics scraping

vLLM exposes Prometheus metrics at `http://localhost:8000/metrics`. Scrape before/after each run:

```python
import requests
def get_metrics():
    r = requests.get("http://localhost:8000/metrics").text
    # parse:
    # vllm:prompt_tokens_total
    # vllm:generation_tokens_total
    # vllm:gpu_cache_usage_perc
    # vllm:time_to_first_token_seconds (histogram)
    # vllm:gpu_prefix_cache_hit_rate
    # vllm:num_requests_running
```

Plus wall-clock timing in client code: `t0 = time.time(); ...; latency = time.time() - t0`.

---

## 5. Benchmarking (60 min, after agent works on 1 patient)

Three configurations, run all 30 patients through each:

**Config A — Baseline (single-shot, no decomposition).** One LLM call per patient: stuff the note + ALL tool data into one prompt, ask for the full structured plan. No fan-out, no parallelism. Establishes "is the agentic architecture worth it" baseline.

**Config B — Full agent, prefix caching OFF.** Relaunch vLLM without `--enable-prefix-caching`. Run the full 3-worker pipeline. Establishes the agentic architecture's cost.

**Config C — Full agent, prefix caching ON.** Relaunch vLLM with `--enable-prefix-caching`. Same pipeline. This is your headline optimization.

For each config, record per-patient:
- End-to-end wall-clock latency
- Total prompt tokens
- Total generation tokens
- Per-worker latency (for B/C)
- Prefix cache hit rate (for C)

For each config, record over the full run:
- Mean throughput (patients/min)
- p50 / p95 / p99 latency
- GPU cache utilization peak

**You want one headline plot:** end-to-end latency per patient (x-axis: input note length in tokens, y-axis: latency seconds), three series (A, B, C). Expected shape: B > A > C for long notes (B does redundant prefill 3x, C amortizes it). This is your money figure.

---

## 6. Evaluation (built into benchmarking, no separate phase)

### 6.1 Ground truth construction

For each of the 30 patients, build a ground-truth JSON from the SQLite database (you have the data — Synthea is fully observable):

```python
ground_truth[patient_id] = {
    "true_medications": <set of active medications at discharge>,
    "true_active_conditions": <set>,
    "should_have_followup_for": <list of conditions requiring followup>,
}
```

### 6.2 Metrics computed automatically

For each (config, patient) pair:
- **Medication recall:** fraction of true_medications appearing in worker output
- **Medication precision:** fraction of agent's medications that are real (hallucination inverse)
- **Problem recall / precision:** same idea
- **JSON validity:** does the worker output parse and match schema
- **Followup appropriateness:** did the followup worker flag the right specialties for the diagnoses (simple rule-based check: diabetes→endo, CHF→cards, etc.)

Aggregate to: macro-F1 per worker per config. Report a 3x3 table (worker × config).

### 6.3 Sanity check

Run Config B and Config C side-by-side, *verify their outputs are essentially identical*. Prefix caching shouldn't change outputs — only latency. If they differ, something's wrong with the cache.

---

## 7. Paper outline (90 min, written in parallel with running benchmarks)

Workshop-style, ~6 pages double-column. Use IEEE or ACM template. Skip if you don't have time — single-column markdown is acceptable if the prof didn't specify.

**Section 1 — Motivation (1/2 page).** Hospital readmission as preventable harm. Existing post-discharge workflows are labor-intensive. Opportunity: an agent that synthesizes the discharge plan with patient history to produce structured follow-up recommendations. Frame as a real workflow with real systems characteristics: long shared context, autonomous reasoning, structured output.

**Section 2 — Agent Design (1.5 pages).** The orchestrator-workers + tool use pattern. Describe each worker, the tools available, the assembler/validator. Include a system architecture figure (you can ask Claude Code to generate an SVG or use mermaid). Justify each design choice in terms of the agentic patterns covered in lecture: planning, tool use, multi-agent, reflection.

**Section 3 — Implementation (1 page).** vLLM + Llama-3.1-8B + Colab A100. Synthea Coherent dataset. SQLite tools. Guided JSON. Mention the prompt-structure design decision for prefix-cache compatibility — that's a non-obvious detail graders will appreciate.

**Section 4 — Evaluation Methodology (1/2 page).** Three configurations (A, B, C). 30 patients. Application metrics: per-worker F1 against Synthea ground truth, JSON validity. Systems metrics: end-to-end latency, prefill/decode token counts, prefix cache hit rate, GPU cache utilization.

**Section 5 — Results (1.5 pages).** The headline plot (latency vs note length, three series). The 3x3 F1 table. The Pareto plot (latency vs F1). Key findings:
- Multi-agent decomposition costs Nx latency vs single-shot but yields M-point F1 improvement
- Prefix caching reduces multi-agent latency by Y% on long notes
- Quality is unchanged by caching (sanity confirmed)
- Tail latency dominated by longest-generating worker

Tie each result to the systems concept that explains it: prefill is compute-bound (roofline), redundant prefill is wasted compute-bound work, prefix caching eliminates it. Reference the lectures.

**Section 6 — Optimization Opportunities (1 page).** Discuss optimizations *not* implemented:
- Pipelined batching across patients (request-level pipeline parallelism)
- Speculative decoding for templated JSON outputs (high acceptance rate expected)
- Prefill-decode disaggregation for multi-tenant deployment
- FP8 quantization for workers, BF16 for the validator (heterogeneous precision)
- Dynamic worker routing (skip workers when their domain isn't in the note)
- Tool-call parallelism (batch multiple SQL queries within a worker)

Frame each as: "in a production deployment, X would address Y bottleneck we observed."

**Section 7 — Limitations and Conclusion (1/2 page).** Synthetic data, mocked drug-interaction API, 30-patient corpus, single model, single hardware. The systems-level findings generalize; the application-level findings are bounded to the Synthea distribution. No clinical safety claims. Future work: real MIMIC-IV-Note evaluation, larger corpus, real RxNav calls.

---

## 8. Concrete handoff to Claude Code

Hand Claude Code this plan plus these instructions:

> Build the project described in `build_plan.md`. Implement files in this order:
> 1. `build_db.py` — load Synthea CSVs into SQLite (download URL in plan)
> 2. `tools.py` — five tool functions per the spec
> 3. `generate_notes.py` — templated discharge note generator, save 30 notes
> 4. `agent.py` — orchestrator + 3 workers + assembler, async, calls vLLM via OpenAI client
> 5. `benchmark.py` — runs three configs over 30 patients, dumps results CSV
> 6. `evaluate.py` — computes ground truth and F1 metrics from results
>
> Use vLLM 0.6.x. Use `guided_json` for worker outputs. Structure all worker prompts with identical shared prefix (note + tool results) followed by worker-specific suffix — this is critical for the prefix-caching ablation. Mock RxNav with a static 5-entry interaction table.
>
> After building, run all three configs and save:
> - `results/metrics.csv` (per-patient, per-config metrics)
> - `results/aggregate.json` (means, medians, percentiles)
> - `results/plots/latency_vs_length.png` (the headline figure)
> - `results/plots/pareto.png` (quality vs latency)
>
> Then generate `paper.md` following the outline in Section 7 of the plan.

---

## 9. Risk mitigations / fallback plans

**If vLLM won't install or boot on Colab in 20 min:** fall back to running Llama 3.2 3B (smaller, faster install) or use TGI as a substitute. Note this in the paper. The systems story is the same.

**If the A100 isn't available on Colab:** use T4 with Qwen2.5-3B-Instruct quantized to INT8 via vLLM's `--quantization` flag. Cite this as a hardware constraint.

**If guided JSON is flaky:** drop to plain JSON prompting with explicit instructions, parse with try/except, and report a "JSON validity rate" as one of your metrics. This is actually MORE interesting from an evaluation standpoint.

**If Synthea data parsing eats too much time:** generate 30 patients programmatically with a Python script using ICD-10 / RxNorm seed lists. Acknowledge as a methodological limitation; ground truth is even cleaner this way.

**If you only get 2 of 3 configurations done:** ship with Config B vs Config C (the prefix caching comparison). That's the headline anyway. Drop Config A and rewrite Section 5 to lead with caching.

**If the paper writing crunches:** prioritize Sections 1, 2, 4, 5. Sections 3 and 6 can be 2-3 paragraphs each. Section 7 can be 4 sentences.

---

## 10. Things to do RIGHT NOW (before starting Claude Code)

1. Activate Colab Pro (if not already) and confirm A100 access in the runtime menu
2. Download the Synthea sample zip to a local folder and inspect a few CSVs to confirm structure
3. Get a HuggingFace access token (Llama 3.1 8B is gated — you need to accept the license on the HF model page first)
4. Open a Colab notebook with A100 selected as the runtime
5. Then hand Claude Code this plan and have it execute

Estimated total wall-clock from "now" to "done": 5 hours if no major surprises, 6 hours if vLLM fights you.