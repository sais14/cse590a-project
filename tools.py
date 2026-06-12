"""tools.py — SQL-backed tool functions over data/synthea.db.

Five tools (per build_plan.md 3.2) plus an aggregator that concatenates their
output into the unified `tool_results` block the workers see.

DETERMINISM CONTRACT (critical for the prefix-caching ablation):
  The same patient queried twice MUST return byte-identical strings.
  - No wall-clock time: `lookback_days` is measured back from the patient's
    discharge date (max encounter date in the DB), which is data-derived and
    fixed. We never call datetime.now().
  - No float reformatting: observation values are echoed as the verbatim text
    stored in the DB, so we never depend on Python's float repr.
  - Stable ordering: every query uses an explicit ORDER BY with full tiebreakers
    AND results are re-sorted in Python, so order never depends on SQLite.
  - No random IDs, no dict-iteration-order leakage into output.

Return type: every tool returns a STRING (a formatted block ready to embed in a
prompt). Concatenating them is how the shared tool_results block is built.
"""
import datetime as _dt
import os
import sqlite3

DB_PATH = os.path.join("data", "synthea.db")


def set_db_path(path):
    global DB_PATH
    DB_PATH = path


def _open():
    """Fresh read-only connection per call (thread-safe under asyncio.to_thread)."""
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    return con


def _reference_date(con, patient_id):
    """Patient's discharge date = latest encounter date (YYYY-MM-DD). Deterministic."""
    row = con.execute(
        """SELECT MAX(substr(COALESCE(NULLIF(stop,''), start), 1, 10))
           FROM encounters WHERE patient_id=?""",
        (patient_id,),
    ).fetchone()
    return row[0] if row and row[0] else None


def _cutoff(ref_date, lookback_days):
    """ISO date string `lookback_days` before ref_date (lexical-comparable)."""
    d = _dt.date.fromisoformat(ref_date) - _dt.timedelta(days=lookback_days)
    return d.isoformat()


def _active_at(stop, ref_date):
    return stop is None or stop == "" or stop[:10] >= ref_date


# --------------------------------------------------------------------------- #
# Tool 1: medication history
# --------------------------------------------------------------------------- #
def get_medication_history(patient_id, lookback_days=365):
    """Medications in the lookback window, aggregated per drug with status."""
    con = _open()
    try:
        ref = _reference_date(con, patient_id)
        if ref is None:
            return "MEDICATION HISTORY:\n  (no encounters on record)"
        cutoff = _cutoff(ref, lookback_days)
        rows = con.execute(
            """SELECT description, substr(start,1,10), stop, reason_description
               FROM medications
               WHERE patient_id=? AND substr(start,1,10)>=? AND substr(start,1,10)<=?
               ORDER BY description ASC, start ASC""",
            (patient_id, cutoff, ref),
        ).fetchall()
    finally:
        con.close()

    agg = {}  # description -> {first_start, ongoing, reasons{reason:count}}
    for desc, start, stop, reason in rows:
        a = agg.setdefault(desc, {"first": start, "ongoing": False, "reasons": {}})
        if start < a["first"]:
            a["first"] = start
        if _active_at(stop, ref):
            a["ongoing"] = True
        if reason:
            a["reasons"][reason] = a["reasons"].get(reason, 0) + 1

    lines = [f"MEDICATION HISTORY (last {lookback_days} days, as of {ref}):"]
    if not agg:
        lines.append("  (none on record)")
    for desc in sorted(agg):
        a = agg[desc]
        status = "ongoing" if a["ongoing"] else "discontinued"
        # indication: most frequent reason, tie-broken alphabetically
        if a["reasons"]:
            ind = sorted(a["reasons"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        else:
            ind = "(unspecified)"
        lines.append(
            f"  - {desc} | first started {a['first']} | status: {status} "
            f"| indication: {ind}"
        )
    return "\n".join(lines)


def _ongoing_medications(patient_id, lookback_days=365):
    """Helper: sorted list of drug descriptions ongoing as of discharge."""
    con = _open()
    try:
        ref = _reference_date(con, patient_id)
        if ref is None:
            return []
        cutoff = _cutoff(ref, lookback_days)
        rows = con.execute(
            """SELECT DISTINCT description, stop FROM medications
               WHERE patient_id=? AND substr(start,1,10)>=? AND substr(start,1,10)<=?""",
            (patient_id, cutoff, ref),
        ).fetchall()
    finally:
        con.close()
    return sorted({d for (d, stop) in rows if _active_at(stop, ref)})


# --------------------------------------------------------------------------- #
# Tool 2: lab / vital trend
# --------------------------------------------------------------------------- #
def get_lab_trend(patient_id, lab_name, lookback_days=180, max_points=8):
    """Time-series of values for a lab/vital (verbatim stored values)."""
    con = _open()
    try:
        ref = _reference_date(con, patient_id)
        if ref is None:
            return f"LAB/VITAL TREND - {lab_name}:\n  (no encounters on record)"
        cutoff = _cutoff(ref, lookback_days)
        rows = con.execute(
            """SELECT substr(date,1,10), value, units FROM observations
               WHERE patient_id=? AND description=?
                 AND substr(date,1,10)>=? AND substr(date,1,10)<=?
               ORDER BY date ASC, code ASC, value ASC""",
            (patient_id, lab_name, cutoff, ref),
        ).fetchall()
    finally:
        con.close()

    # Keep the most recent `max_points` readings, then present oldest->newest.
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    if len(rows) > max_points:
        rows = rows[-max_points:]
    lines = [f"LAB/VITAL TREND - {lab_name} (last {lookback_days} days, as of {ref}):"]
    if not rows:
        lines.append("  (no readings in window)")
    for date, value, units in rows:
        u = f" {units}" if units else ""
        lines.append(f"  - {date}: {value}{u}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tool 3: prior diagnoses
# --------------------------------------------------------------------------- #
def get_prior_diagnoses(patient_id):
    """Active and resolved conditions as of the discharge date."""
    con = _open()
    try:
        ref = _reference_date(con, patient_id)
        if ref is None:
            return "DIAGNOSES:\n  (no encounters on record)"
        rows = con.execute(
            """SELECT DISTINCT description, stop FROM conditions
               WHERE patient_id=? AND substr(start,1,10)<=?
               ORDER BY description ASC""",
            (patient_id, ref),
        ).fetchall()
    finally:
        con.close()

    active = sorted({d for (d, stop) in rows if _active_at(stop, ref)})
    resolved = sorted({d for (d, stop) in rows if not _active_at(stop, ref)})
    lines = [f"DIAGNOSES (as of {ref}):", "  Active:"]
    lines.extend(f"    - {c}" for c in active) if active else lines.append("    (none)")
    lines.append("  Resolved:")
    lines.extend(f"    - {c}" for c in resolved) if resolved else lines.append("    (none)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tool 4: prior encounters
# --------------------------------------------------------------------------- #
def get_prior_encounters(patient_id, n=5):
    """Most recent `n` encounter summaries (date, class, reason)."""
    con = _open()
    try:
        ref = _reference_date(con, patient_id)
        rows = con.execute(
            """SELECT substr(start,1,10), encounter_class,
                      COALESCE(NULLIF(reason_description,''), description), id
               FROM encounters WHERE patient_id=?
               ORDER BY start DESC, id ASC
               LIMIT ?""",
            (patient_id, n),
        ).fetchall()
    finally:
        con.close()

    lines = [f"PRIOR ENCOUNTERS (most recent {n}):"]
    if not rows:
        lines.append("  (none on record)")
    for date, eclass, reason, _id in rows:
        lines.append(f"  - {date} [{eclass}] {reason}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tool 5: drug interactions (MOCKED — static table, per build_plan.md 3.2)
# --------------------------------------------------------------------------- #
# Map drug-name substrings (lowercase) -> pharmacologic class.
_DRUG_CLASSES = [
    ("warfarin", "anticoagulant"),
    ("aspirin", "nsaid"),
    ("ibuprofen", "nsaid"),
    ("naproxen", "nsaid"),
    ("ketorolac", "nsaid"),
    ("diclofenac", "nsaid"),
    ("celecoxib", "nsaid"),
    ("meloxicam", "nsaid"),
    ("lisinopril", "ace_arb"),
    ("enalapril", "ace_arb"),
    ("ramipril", "ace_arb"),
    ("captopril", "ace_arb"),
    ("benazepril", "ace_arb"),
    ("olmesartan", "ace_arb"),
    ("losartan", "ace_arb"),
    ("valsartan", "ace_arb"),
    ("irbesartan", "ace_arb"),
    ("candesartan", "ace_arb"),
    ("spironolactone", "k_sparing_diuretic"),
    ("eplerenone", "k_sparing_diuretic"),
    ("amiloride", "k_sparing_diuretic"),
    ("triamterene", "k_sparing_diuretic"),
    ("hydrochlorothiazide", "k_wasting_diuretic"),
    ("chlorthalidone", "k_wasting_diuretic"),
    ("furosemide", "k_wasting_diuretic"),
    ("bumetanide", "k_wasting_diuretic"),
    ("digoxin", "digoxin"),
    ("simvastatin", "statin"),
    ("atorvastatin", "statin"),
    ("lovastatin", "statin"),
    ("rosuvastatin", "statin"),
    ("pravastatin", "statin"),
    ("amlodipine", "ccb"),
    ("diltiazem", "ccb"),
    ("verapamil", "ccb"),
]

# Static 5-entry interaction table: (classA, classB, severity, description).
_INTERACTIONS = [
    ("anticoagulant", "nsaid", "high",
     "Concurrent anticoagulant and NSAID increases GI bleeding risk."),
    ("ace_arb", "k_sparing_diuretic", "high",
     "ACE-inhibitor/ARB with potassium-sparing diuretic risks hyperkalemia."),
    ("ace_arb", "nsaid", "medium",
     "ACE-inhibitor/ARB with NSAID may reduce renal function and blunt BP control."),
    ("digoxin", "k_wasting_diuretic", "high",
     "Diuretic-induced hypokalemia increases digoxin toxicity risk."),
    ("statin", "ccb", "medium",
     "Calcium-channel blocker raises statin exposure; myopathy/rhabdomyolysis risk."),
]

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _classes_for(drug_name):
    low = drug_name.lower()
    return {cls for (kw, cls) in _DRUG_CLASSES if kw in low}


def check_drug_interactions(drug_list):
    """MOCK RxNav: detect interacting pairs via a static 5-entry table.

    drug_list: list of drug-name strings. Output is deterministic (sorted).
    """
    # Map present classes -> sorted example drug names that triggered them.
    present = {}
    for name in sorted(set(drug_list)):
        for cls in _classes_for(name):
            present.setdefault(cls, [])
            if name not in present[cls]:
                present[cls].append(name)

    warnings = []
    for a, b, severity, desc in _INTERACTIONS:
        if a in present and b in present:
            drug_a = sorted(present[a])[0]
            drug_b = sorted(present[b])[0]
            pair = sorted([drug_a, drug_b])
            warnings.append((severity, desc, pair[0], pair[1]))

    warnings.sort(key=lambda w: (_SEVERITY_RANK[w[0]], w[1], w[2], w[3]))
    lines = ["DRUG INTERACTION CHECK (mocked RxNav, static table):"]
    if not warnings:
        lines.append("  (no interactions detected among current medications)")
    for severity, desc, da, db in warnings:
        lines.append(f"  - [{severity.upper()}] {da} + {db}: {desc}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Aggregator: unified tool_results block
# --------------------------------------------------------------------------- #
# Fixed, deterministic set of labs/vitals to trend in the shared context.
_TREND_LABS = [
    "Systolic Blood Pressure",
    "Body Mass Index",
    "Glucose",
    "Creatinine",
    "Potassium",
    "Hemoglobin A1c/Hemoglobin.total in Blood",
]


def build_tool_results(patient_id):
    """Concatenate all five tools into the unified tool_results block.

    This is THE shared block injected identically into every worker's prompt,
    so it must be byte-deterministic for a given patient_id.
    """
    blocks = []
    blocks.append(get_medication_history(patient_id, lookback_days=365))
    blocks.append(get_prior_diagnoses(patient_id))
    blocks.append(get_prior_encounters(patient_id, n=5))
    # Lab trends: wide window so sparse synthetic records still show a trend.
    for lab in _TREND_LABS:
        blocks.append(get_lab_trend(patient_id, lab, lookback_days=3650, max_points=6))
    ongoing = _ongoing_medications(patient_id, lookback_days=365)
    blocks.append(check_drug_interactions(ongoing))
    return "\n\n".join(blocks)


if __name__ == "__main__":
    import json
    import sys
    manifest = json.load(open(os.path.join("data", "notes_manifest.json")))
    pid = sys.argv[1] if len(sys.argv) > 1 else manifest[0]["patient_id"]
    print(build_tool_results(pid))
