"""agent.py — Orchestrator + 3 workers + assembler + reflection.

Pipeline (build_plan.md 3.3):
    [note + patient_id]
        -> tool execution (tools.build_tool_results)
        -> 3 workers fired CONCURRENTLY (asyncio.gather), each sharing a
           byte-identical prompt prefix (the prefix-caching contract)
        -> assembler -> one-round reflection -> structured follow-up plan

Key guarantees:
  * build_worker_prompt(worker, note, tool_results) constructs
    SHARED_PREFIX + WORKER_SUFFIX, where SHARED_PREFIX (system preamble + note +
    tool_results) is byte-identical across all three workers for a given patient.
    Only the suffix (role instructions + schema description) varies.
  * guided_json is attempted first; on parse/schema failure we retry once with
    plain prompting + an explicit "valid JSON only" instruction. Per-call
    validity is recorded for the benchmark.
  * Every LLM call is timed and its prompt/completion token usage recorded.

The real client is openai.AsyncOpenAI(base_url=.../v1). MockClient mirrors that
surface so orchestration wiring can be verified locally without a GPU.
"""
import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace

import tools

MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# --------------------------------------------------------------------------- #
# Prompt construction — the prefix-caching contract
# --------------------------------------------------------------------------- #
SYSTEM_PREAMBLE = (
    "You are part of an autonomous post-discharge patient follow-up system. "
    "You are given a hospital discharge note and structured patient history "
    "retrieved from the medical record. Reason ONLY from the information "
    "provided below. Do not invent medications, conditions, or facts that are "
    "not supported by the note or the retrieved history."
)

WORKER_SUFFIXES = {
    "medication": (
        "=== YOUR TASK: MEDICATION RECONCILIATION ===\n"
        "You are the medication reconciliation specialist. Reconcile the "
        "patient's discharge medications against the retrieved medication "
        "history and flag interactions and discrepancies.\n"
        "Output a single JSON object with EXACTLY these fields:\n"
        "  discharge_medications: array of objects "
        "{name: string, dose: string, indication: string, "
        "status: one of [new, continued, changed]}\n"
        "  reconciliation_issues: array of objects "
        "{issue: string, severity: one of [low, medium, high]}\n"
        "  interaction_warnings: array of strings\n"
        "Only include medications supported by the discharge note. "
        "Output JSON only, no prose."
    ),
    "problem": (
        "=== YOUR TASK: PROBLEM LIST ===\n"
        "You are the problem-list specialist. Build the patient's active and "
        "resolved problem list from the note and history.\n"
        "Output a single JSON object with EXACTLY these fields:\n"
        "  active_problems: array of objects "
        "{name: string, type: one of [chronic, acute, new]}\n"
        "  resolved_problems: array of strings\n"
        "Output JSON only, no prose."
    ),
    "followup": (
        "=== YOUR TASK: FOLLOW-UP PLAN ===\n"
        "You are the follow-up planning specialist. Produce the post-discharge "
        "follow-up plan. Every appointment reason must correspond to a "
        "documented active problem. Fold in red-flag and patient-instruction "
        "guidance.\n"
        "Output a single JSON object with EXACTLY these fields:\n"
        "  appointments: array of objects "
        "{specialty: string, timeframe_days: integer, reason: string}\n"
        "  monitoring: array of objects "
        "{what_to_watch: string, when_to_call: string}\n"
        "  red_flags: array of strings\n"
        "Output JSON only, no prose."
    ),
}

PLAIN_JSON_INSTRUCTION = (
    "\n\nIMPORTANT: Output VALID JSON ONLY, exactly matching the field schema "
    "above. No prose, no explanation, no markdown code fences."
)


def shared_prefix(note, tool_results):
    """The byte-identical leading context shared by all three workers."""
    return (
        SYSTEM_PREAMBLE
        + "\n\n=== DISCHARGE NOTE ===\n"
        + note
        + "\n\n=== RETRIEVED PATIENT HISTORY ===\n"
        + tool_results
        + "\n\n"
    )


def build_worker_prompt(worker_name, note, tool_results, correction=None):
    """Return {messages, prefix, suffix} for a worker.

    prefix is byte-identical across workers for a given (note, tool_results).
    Only suffix varies. A reflection `correction` is appended AFTER the suffix
    so the shared prefix (and thus the cache) is preserved.
    """
    prefix = shared_prefix(note, tool_results)
    suffix = WORKER_SUFFIXES[worker_name]
    if correction:
        suffix = suffix + "\n\n=== CORRECTION REQUIRED ===\n" + correction
    content = prefix + suffix
    messages = [{"role": "user", "content": content}]
    return {"messages": messages, "prefix": prefix, "suffix": suffix}


# --------------------------------------------------------------------------- #
# JSON schemas (build_plan.md 3.5)
# --------------------------------------------------------------------------- #
SCHEMAS = {
    "medication": {
        "type": "object",
        "properties": {
            "discharge_medications": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dose": {"type": "string"},
                    "indication": {"type": "string"},
                    "status": {"type": "string",
                               "enum": ["new", "continued", "changed"]},
                },
                "required": ["name", "dose", "indication", "status"],
            }},
            "reconciliation_issues": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "issue": {"type": "string"},
                    "severity": {"type": "string",
                                 "enum": ["low", "medium", "high"]},
                },
                "required": ["issue", "severity"],
            }},
            "interaction_warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["discharge_medications", "reconciliation_issues",
                     "interaction_warnings"],
    },
    "problem": {
        "type": "object",
        "properties": {
            "active_problems": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string",
                             "enum": ["chronic", "acute", "new"]},
                },
                "required": ["name", "type"],
            }},
            "resolved_problems": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["active_problems", "resolved_problems"],
    },
    "followup": {
        "type": "object",
        "properties": {
            "appointments": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "specialty": {"type": "string"},
                    "timeframe_days": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["specialty", "timeframe_days", "reason"],
            }},
            "monitoring": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "what_to_watch": {"type": "string"},
                    "when_to_call": {"type": "string"},
                },
                "required": ["what_to_watch", "when_to_call"],
            }},
            "red_flags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["appointments", "monitoring", "red_flags"],
    },
}

EMPTY_OUTPUT = {
    "medication": {"discharge_medications": [], "reconciliation_issues": [],
                   "interaction_warnings": []},
    "problem": {"active_problems": [], "resolved_problems": []},
    "followup": {"appointments": [], "monitoring": [], "red_flags": []},
}


def validate_json(instance, schema):
    """Minimal JSON-Schema validator (stdlib only): object/array/string/
    integer/number/boolean, properties, required, items, enum."""
    t = schema.get("type")
    if t == "object":
        if not isinstance(instance, dict):
            return False
        for req in schema.get("required", []):
            if req not in instance:
                return False
        for k, sub in schema.get("properties", {}).items():
            if k in instance and not validate_json(instance[k], sub):
                return False
        return True
    if t == "array":
        if not isinstance(instance, list):
            return False
        item = schema.get("items")
        return all(validate_json(x, item) for x in instance) if item else True
    if t == "string":
        if not isinstance(instance, str):
            return False
        return "enum" not in schema or instance in schema["enum"]
    if t == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if t == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if t == "boolean":
        return isinstance(instance, bool)
    return True


def _extract_json(text):
    """Parse a JSON object from a model response, tolerating ```json fences."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    # Fall back to the outermost {...} span.
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
        raise


# --------------------------------------------------------------------------- #
# Timed LLM call + per-call records
# --------------------------------------------------------------------------- #
@dataclass
class CallRecord:
    worker: str
    attempt: str            # "guided" | "fallback"
    is_correction: bool
    guided: bool
    json_valid: bool
    schema_valid: bool
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    ttft_s: float = 0.0     # time-to-first-token (== latency if not streamed)


async def _one_call(client, model, messages, schema, worker, attempt,
                    guided, is_correction, records, stream=False,
                    max_tokens=1024):
    """Make one timed LLM call; parse + validate; append a CallRecord.

    When stream=True the response is streamed so time-to-first-token (TTFT) is
    measured; otherwise ttft_s == latency_s. Returns (parsed_or_None, raw_text).
    """
    extra_body = {"guided_json": schema} if guided else None
    t0 = time.perf_counter()
    ttft = 0.0

    if stream:
        text_parts = []
        p_tok = c_tok = 0
        first = True
        resp_stream = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.0,
            max_tokens=max_tokens, extra_body=extra_body, stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in resp_stream:
            usage = getattr(chunk, "usage", None)
            if usage:
                p_tok = getattr(usage, "prompt_tokens", None) or p_tok
                c_tok = getattr(usage, "completion_tokens", None) or c_tok
            choices = getattr(chunk, "choices", None)
            if choices:
                piece = getattr(choices[0].delta, "content", None)
                if piece:
                    if first:
                        ttft = time.perf_counter() - t0
                        first = False
                    text_parts.append(piece)
        text = "".join(text_parts)
        latency = time.perf_counter() - t0
    else:
        resp = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.0,
            max_tokens=max_tokens, extra_body=extra_body,
        )
        latency = time.perf_counter() - t0
        ttft = latency
        text = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        p_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        c_tok = getattr(usage, "completion_tokens", 0) if usage else 0

    json_valid, schema_valid, parsed = False, False, None
    try:
        parsed = _extract_json(text)
        json_valid = True
        schema_valid = validate_json(parsed, schema)
    except Exception:
        parsed = None

    records.append(CallRecord(
        worker=worker, attempt=attempt, is_correction=is_correction,
        guided=guided, json_valid=json_valid, schema_valid=schema_valid,
        prompt_tokens=p_tok, completion_tokens=c_tok, latency_s=latency,
        ttft_s=ttft,
    ))
    return (parsed if (json_valid and schema_valid) else None), text


async def call_worker(client, worker, note, tool_results, model, records,
                      use_guided=True, correction=None, stream=False):
    """Run one worker: guided_json first, then plain-JSON fallback."""
    schema_key = worker
    schema = SCHEMAS[schema_key]
    prompt = build_worker_prompt(worker, note, tool_results, correction=correction)
    is_corr = correction is not None

    if use_guided:
        parsed, _ = await _one_call(
            client, model, prompt["messages"], schema, worker, "guided",
            guided=True, is_correction=is_corr, records=records, stream=stream)
        if parsed is not None:
            return parsed

    # Fallback: plain prompting + explicit JSON-only instruction.
    msgs = [dict(m) for m in prompt["messages"]]
    msgs[-1]["content"] += PLAIN_JSON_INSTRUCTION
    parsed, _ = await _one_call(
        client, model, msgs, schema, worker, "fallback",
        guided=False, is_correction=is_corr, records=records, stream=stream)
    return parsed if parsed is not None else EMPTY_OUTPUT[schema_key]


# --------------------------------------------------------------------------- #
# Assembler
# --------------------------------------------------------------------------- #
def assemble(patient_id, outputs):
    """Combine the three worker outputs into a unified follow_up_plan JSON."""
    return {
        "patient_id": patient_id,
        "medication_reconciliation": outputs["medication"],
        "problem_list": outputs["problem"],
        "follow_up": outputs["followup"],
    }


# --------------------------------------------------------------------------- #
# Reflection (one round)
# --------------------------------------------------------------------------- #
def _norm(s):
    return "".join(c if c.isalnum() else " " for c in s.lower()).split()


def _name_in_note(name, note_lower):
    """A med name is supported if any of its alphabetic tokens (len>=4) is in
    the note (drug names; ignore dose/number tokens)."""
    toks = [t for t in _norm(name) if len(t) >= 4 and not t.isdigit()]
    return any(t in note_lower for t in toks)


def _problem_matches(reason, problem_names_norm):
    """reason matches a problem if they share a content token (len>=4)."""
    rtoks = {t for t in _norm(reason) if len(t) >= 4}
    return any(rtoks & ptoks for ptoks in problem_names_norm)


def reflection_check(outputs, note):
    """Detect inconsistencies. Returns {offending: {worker: correction_str},
    findings: [str]} so callers can log the decision."""
    findings = []
    offending = {}
    note_lower = note.lower()

    # Check 1: medication worker flagged a drug "new" that isn't in the note.
    bad_new = []
    for med in outputs["medication"].get("discharge_medications", []):
        if med.get("status") == "new" and not _name_in_note(med.get("name", ""),
                                                             note_lower):
            bad_new.append(med.get("name", "?"))
    if bad_new:
        findings.append(
            "MEDICATION: drug(s) marked 'new' but absent from the discharge "
            "note: " + ", ".join(bad_new))
        offending["medication"] = (
            "These medications are marked status 'new' but do not appear in the "
            "discharge note: " + ", ".join(bad_new) + ". Remove medications not "
            "supported by the note, or correct their status. Re-output the full "
            "JSON object.")
    else:
        findings.append("MEDICATION: all 'new' medications are supported by the note. OK")

    # Check 2: every follow-up appointment reason maps to an active problem.
    problem_names_norm = [
        {t for t in _norm(p.get("name", "")) if len(t) >= 4}
        for p in outputs["problem"].get("active_problems", [])
    ]
    bad_reasons = []
    for appt in outputs["followup"].get("appointments", []):
        reason = appt.get("reason", "")
        if not _problem_matches(reason, problem_names_norm):
            bad_reasons.append(reason)
    if bad_reasons:
        findings.append(
            "FOLLOW-UP: appointment reason(s) not in the problem list: "
            + ", ".join(bad_reasons))
        offending["followup"] = (
            "These follow-up appointment reasons do not correspond to any active "
            "problem in the problem list: " + ", ".join(bad_reasons) + ". Align "
            "every appointment reason to a documented active problem, or remove "
            "it. Re-output the full JSON object.")
    else:
        findings.append("FOLLOW-UP: all appointment reasons map to active problems. OK")

    return {"offending": offending, "findings": findings}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class AgentResult:
    patient_id: str
    plan: dict
    outputs: dict
    records: list = field(default_factory=list)
    reflection: dict = field(default_factory=dict)
    wall_clock_s: float = 0.0


async def run_agent(client, patient_id, note, tool_results=None, model=MODEL,
                    use_guided=True, reflect=True, stream=False):
    """Full 3-worker pipeline for one patient (Configs B/C)."""
    if tool_results is None:
        tool_results = tools.build_tool_results(patient_id)
    records = []
    t0 = time.perf_counter()

    workers = ["medication", "problem", "followup"]
    results = await asyncio.gather(*[
        call_worker(client, w, note, tool_results, model, records,
                    use_guided=use_guided, stream=stream)
        for w in workers
    ])
    outputs = dict(zip(workers, results))

    reflection = {"findings": [], "offending": {}, "corrected": []}
    if reflect:
        chk = reflection_check(outputs, note)
        reflection["findings"] = chk["findings"]
        reflection["offending"] = list(chk["offending"].keys())
        if chk["offending"]:
            # Re-invoke ONLY the offending workers, once, concurrently.
            offenders = list(chk["offending"].items())
            worker_map = {"medication": "medication", "problem": "problem",
                          "followup": "followup"}
            fixed = await asyncio.gather(*[
                call_worker(client, worker_map[w], note, tool_results, model,
                            records, use_guided=use_guided, correction=corr,
                            stream=stream)
                for w, corr in offenders
            ])
            for (w, _), parsed in zip(offenders, fixed):
                outputs[w] = parsed
            reflection["corrected"] = [w for w, _ in offenders]

    plan = assemble(patient_id, outputs)
    result = AgentResult(
        patient_id=patient_id, plan=plan, outputs=outputs, records=records,
        reflection=reflection, wall_clock_s=time.perf_counter() - t0,
    )
    return result


# --------------------------------------------------------------------------- #
# Config A: single-shot baseline (no decomposition)
# --------------------------------------------------------------------------- #
FULL_SCHEMA = {
    "type": "object",
    "properties": {
        "medication_reconciliation": SCHEMAS["medication"],
        "problem_list": SCHEMAS["problem"],
        "follow_up": SCHEMAS["followup"],
    },
    "required": ["medication_reconciliation", "problem_list", "follow_up"],
}

SINGLE_SHOT_SUFFIX = (
    "=== YOUR TASK: COMPREHENSIVE POST-DISCHARGE PLAN ===\n"
    "Produce the complete post-discharge follow-up plan in ONE response. "
    "Output a single JSON object with EXACTLY these fields:\n"
    "  medication_reconciliation: {discharge_medications: array of "
    "{name, dose, indication, status in [new|continued|changed]}, "
    "reconciliation_issues: array of {issue, severity in [low|medium|high]}, "
    "interaction_warnings: array of strings}\n"
    "  problem_list: {active_problems: array of {name, type in "
    "[chronic|acute|new]}, resolved_problems: array of strings}\n"
    "  follow_up: {appointments: array of {specialty, timeframe_days, reason}, "
    "monitoring: array of {what_to_watch, when_to_call}, red_flags: array of "
    "strings}\n"
    "Output JSON only, no prose."
)


async def run_single_shot(client, patient_id, note, tool_results=None,
                          model=MODEL, use_guided=True, stream=False):
    """One LLM call producing the full plan (Config A baseline)."""
    if tool_results is None:
        tool_results = tools.build_tool_results(patient_id)
    records = []
    t0 = time.perf_counter()
    content = shared_prefix(note, tool_results) + SINGLE_SHOT_SUFFIX
    messages = [{"role": "user", "content": content}]

    parsed = None
    if use_guided:
        parsed, _ = await _one_call(
            client, model, messages, FULL_SCHEMA, "single_shot", "guided",
            guided=True, is_correction=False, records=records, stream=stream,
            max_tokens=2048)
    if parsed is None:
        msgs = [dict(m) for m in messages]
        msgs[-1]["content"] += PLAIN_JSON_INSTRUCTION
        parsed, _ = await _one_call(
            client, model, msgs, FULL_SCHEMA, "single_shot", "fallback",
            guided=False, is_correction=False, records=records, stream=stream,
            max_tokens=2048)
    if parsed is None:
        parsed = {"medication_reconciliation": EMPTY_OUTPUT["medication"],
                  "problem_list": EMPTY_OUTPUT["problem"],
                  "follow_up": EMPTY_OUTPUT["followup"]}

    plan = {"patient_id": patient_id, **parsed}
    return AgentResult(
        patient_id=patient_id, plan=plan,
        outputs={"single_shot": parsed}, records=records,
        reflection={"findings": ["(single-shot: no reflection)"],
                    "offending": [], "corrected": []},
        wall_clock_s=time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------- #
# MockClient — mirrors openai.AsyncOpenAI surface for local wiring tests
# --------------------------------------------------------------------------- #
def _resp(content, prompt_text):
    """Build an OpenAI-shaped response object."""
    usage = SimpleNamespace(
        prompt_tokens=len(prompt_text) // 4,
        completion_tokens=len(content) // 4,
        total_tokens=(len(prompt_text) + len(content)) // 4,
    )
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice], usage=usage)


# Canned outputs tailored to note_27 (CHF/AFib patient). The medication and
# follow-up first-pass outputs each contain ONE deliberate inconsistency so the
# reflection checks fire; the correction pass returns clean output.
_MOCK_MED_FIRST = {
    "discharge_medications": [
        {"name": "Furosemide", "dose": "40 MG", "indication": "CHF", "status": "continued"},
        {"name": "Metoprolol succinate", "dose": "100 MG", "indication": "CHF", "status": "continued"},
        {"name": "Digoxin", "dose": "0.125 MG", "indication": "Atrial fibrillation", "status": "continued"},
        {"name": "Warfarin", "dose": "5 MG", "indication": "Atrial fibrillation", "status": "continued"},
        {"name": "Lisinopril", "dose": "10 MG", "indication": "CHF", "status": "new"},
    ],
    "reconciliation_issues": [
        {"issue": "Warfarin requires INR monitoring after discharge", "severity": "medium"},
    ],
    "interaction_warnings": [
        "Furosemide + Digoxin: diuretic-induced hypokalemia increases digoxin toxicity risk.",
    ],
}
_MOCK_MED_CORRECTED = {
    "discharge_medications": [m for m in _MOCK_MED_FIRST["discharge_medications"]
                             if m["name"] != "Lisinopril"],
    "reconciliation_issues": _MOCK_MED_FIRST["reconciliation_issues"],
    "interaction_warnings": _MOCK_MED_FIRST["interaction_warnings"],
}
_MOCK_PROBLEM = {
    "active_problems": [
        {"name": "Chronic congestive heart failure", "type": "chronic"},
        {"name": "Atrial Fibrillation", "type": "chronic"},
        {"name": "Anemia", "type": "chronic"},
        {"name": "Prediabetes", "type": "chronic"},
    ],
    "resolved_problems": ["Acute viral pharyngitis", "Viral sinusitis"],
}
_MOCK_FOLLOWUP_FIRST = {
    "appointments": [
        {"specialty": "cardiology", "timeframe_days": 14, "reason": "Chronic congestive heart failure"},
        {"specialty": "cardiology", "timeframe_days": 14, "reason": "Atrial Fibrillation"},
        {"specialty": "endocrinology", "timeframe_days": 30, "reason": "Prediabetes"},
        {"specialty": "hematology", "timeframe_days": 30, "reason": "Hyperlipidemia"},
    ],
    "monitoring": [
        {"what_to_watch": "Daily weight", "when_to_call": "Weight gain > 2 kg in 2 days"},
        {"what_to_watch": "INR", "when_to_call": "INR outside 2.0-3.0"},
    ],
    "red_flags": ["Worsening shortness of breath", "Chest pain", "Syncope"],
}
_MOCK_FOLLOWUP_CORRECTED = {
    "appointments": [a for a in _MOCK_FOLLOWUP_FIRST["appointments"]
                     if a["reason"] != "Hyperlipidemia"]
    + [{"specialty": "hematology", "timeframe_days": 30, "reason": "Anemia"}],
    "monitoring": _MOCK_FOLLOWUP_FIRST["monitoring"],
    "red_flags": _MOCK_FOLLOWUP_FIRST["red_flags"],
}
_MOCK_SINGLE_SHOT = {
    "medication_reconciliation": _MOCK_MED_CORRECTED,
    "problem_list": _MOCK_PROBLEM,
    "follow_up": _MOCK_FOLLOWUP_CORRECTED,
}


async def _mock_stream(content, prompt_text):
    """Async generator emulating an OpenAI/vLLM streamed completion."""
    mid = max(1, len(content) // 2)
    for piece in (content[:mid], content[mid:]):
        delta = SimpleNamespace(content=piece)
        choice = SimpleNamespace(delta=delta)
        yield SimpleNamespace(choices=[choice], usage=None)
    usage = SimpleNamespace(
        prompt_tokens=len(prompt_text) // 4,
        completion_tokens=len(content) // 4,
        total_tokens=(len(prompt_text) + len(content)) // 4,
    )
    yield SimpleNamespace(choices=[], usage=usage)


class _MockCompletions:
    async def create(self, model, messages, temperature=0.0, max_tokens=1024,
                     extra_body=None, stream=False, stream_options=None):
        await asyncio.sleep(0)  # behave like a coroutine that yields
        content = messages[-1]["content"]
        low = content.lower()
        is_correction = "correction required" in low

        if "comprehensive post-discharge plan" in low:
            out = _MOCK_SINGLE_SHOT
        elif "medication reconciliation specialist" in low:
            out = _MOCK_MED_CORRECTED if is_correction else _MOCK_MED_FIRST
        elif "problem-list specialist" in low:
            out = _MOCK_PROBLEM
        elif "follow-up planning specialist" in low:
            out = _MOCK_FOLLOWUP_CORRECTED if is_correction else _MOCK_FOLLOWUP_FIRST
        else:
            out = {}
        text = json.dumps(out)
        if stream:
            return _mock_stream(text, content)
        return _resp(text, content)


class _MockChat:
    def __init__(self):
        self.completions = _MockCompletions()


class MockClient:
    """Drop-in stand-in for openai.AsyncOpenAI for local wiring tests."""
    def __init__(self):
        self.chat = _MockChat()


def records_to_dicts(records):
    return [asdict(r) for r in records]
