"""generate_notes.py — Build templated discharge notes + ground truth.

Per build_plan.md 2.3 (Option A): for each selected patient's most recent
inpatient encounter, fill a discharge-summary template from structured Synthea
data. Also emits fully-observable ground truth (2.4 / 6.1) for evaluation.

Selection: eligible patients (from build_db) that additionally have at least
one inpatient encounter and >=2 prior encounters before it (so the history
tools have something to find). Deterministic (sorted by id) for reproducibility.

Outputs:
    data/notes/note_00.txt ... note_29.txt   (the discharge notes)
    data/notes_manifest.json                  (note_id -> patient_id, path, chars)
    data/ground_truth.json                    (patient_id -> truth dicts)
"""
import json
import os
import sqlite3

DB_PATH = os.path.join("data", "synthea.db")
NOTES_DIR = os.path.join("data", "notes")
MANIFEST_PATH = os.path.join("data", "notes_manifest.json")
GROUND_TRUTH_PATH = os.path.join("data", "ground_truth.json")

N_NOTES = 30

# Vitals/labs to surface in the note narrative (description -> short label).
VITALS = [
    "Systolic Blood Pressure", "Diastolic Blood Pressure", "Heart rate",
    "Respiratory rate", "Body Weight", "Body Mass Index", "Body Height",
]
LABS = [
    "Glucose", "Hemoglobin A1c/Hemoglobin.total in Blood", "Creatinine",
    "Urea Nitrogen", "Sodium", "Potassium", "Chloride",
    "Total Cholesterol", "Triglycerides",
    "Low Density Lipoprotein Cholesterol",
    "High Density Lipoprotein Cholesterol", "Hemoglobin [Mass/volume] in Blood",
]

# condition keyword -> recommended follow-up specialty (eval rule, 6.2).
FOLLOWUP_RULES = [
    ("diabetes", "endocrinology"),
    ("prediabetes", "endocrinology"),
    ("hyperlipidemia", "cardiology"),
    ("hypertension", "cardiology"),
    ("heart failure", "cardiology"),
    ("congestive", "cardiology"),
    ("coronary", "cardiology"),
    ("myocardial", "cardiology"),
    ("atrial fibrillation", "cardiology"),
    ("chronic kidney", "nephrology"),
    ("renal", "nephrology"),
    ("copd", "pulmonology"),
    ("asthma", "pulmonology"),
    ("chronic obstructive", "pulmonology"),
    ("anemia", "hematology"),
    ("obesity", "primary care"),
]


def followup_specialty(desc):
    d = desc.lower()
    for kw, spec in FOLLOWUP_RULES:
        if kw in d:
            return spec
    return None


def active_at(stop, when):
    """A row (with `stop`) is active at `when` if stop is blank or stop >= when."""
    return stop is None or stop == "" or stop >= when


def year(iso):
    return int(iso[:4]) if iso else None


def select_patients(con):
    """Return up to N_NOTES (patient_id, index_encounter) tuples."""
    rows = con.execute(
        """
        SELECT id FROM eligible_patients
        WHERE id IN (SELECT DISTINCT patient_id FROM encounters
                     WHERE encounter_class='inpatient')
        ORDER BY id
        """
    ).fetchall()
    selected = []
    for (pid,) in rows:
        enc = con.execute(
            """
            SELECT id, start, stop, encounter_class, description, reason_description
            FROM encounters
            WHERE patient_id=? AND encounter_class='inpatient'
            ORDER BY start DESC LIMIT 1
            """,
            (pid,),
        ).fetchone()
        if not enc:
            continue
        index_start = enc[1]
        n_prior = con.execute(
            "SELECT COUNT(*) FROM encounters WHERE patient_id=? AND start<?",
            (pid, index_start),
        ).fetchone()[0]
        if n_prior < 2:
            continue
        selected.append((pid, enc))
        if len(selected) >= N_NOTES:
            break
    return selected


def patient_demo(con, pid, as_of):
    p = con.execute(
        "SELECT birthdate, gender FROM patients WHERE id=?", (pid,)
    ).fetchone()
    birth, gender = p
    age = (year(as_of) - year(birth)) if birth else "unknown"
    g = {"M": "male", "F": "female"}.get(gender, gender or "unknown")
    return age, g


def render_note(con, pid, enc):
    enc_id, start, stop, eclass, edesc, ereason = enc
    age, gender = patient_demo(con, pid, start)
    chief = ereason or edesc or "acute illness"

    # Conditions newly recorded at this encounter (the acute presentation).
    new_conditions = [
        r[0] for r in con.execute(
            "SELECT description FROM conditions WHERE patient_id=? AND encounter_id=?",
            (pid, enc_id),
        ).fetchall()
    ]
    # Active conditions as of discharge (chronic problem list).
    active_conditions = sorted({
        r[0] for r in con.execute(
            "SELECT description, stop FROM conditions WHERE patient_id=? AND start<=?",
            (pid, stop or start),
        ).fetchall() if active_at(r[1], stop or start)
    })
    # Procedures performed during the stay.
    procs = [
        r[0] for r in con.execute(
            "SELECT description FROM procedures WHERE patient_id=? AND encounter_id=?",
            (pid, enc_id),
        ).fetchall()
    ]
    # Medications started during the stay.
    started_meds = [
        r[0] for r in con.execute(
            "SELECT description FROM medications WHERE patient_id=? AND encounter_id=?",
            (pid, enc_id),
        ).fetchall()
    ]
    # Active medications at discharge.
    discharge_meds = sorted({
        r[0] for r in con.execute(
            "SELECT description, stop FROM medications WHERE patient_id=? AND start<=?",
            (pid, stop or start),
        ).fetchall() if active_at(r[1], stop or start)
    })
    # Vitals/labs recorded at this encounter for the narrative.
    obs = con.execute(
        """SELECT description, value, units FROM observations
           WHERE patient_id=? AND encounter_id=?""",
        (pid, enc_id),
    ).fetchall()
    obs_map = {d: (v, u) for (d, v, u) in obs}

    # Past medical history: conditions resolved before this admission.
    resolved_conditions = sorted({
        r[0] for r in con.execute(
            "SELECT description, stop FROM conditions WHERE patient_id=? AND start<?",
            (pid, start),
        ).fetchall() if r[1] and r[1] != "" and r[1] < (stop or start)
    })
    # Prior encounters (most recent 8 before this admission) for the recap.
    prior_encs = con.execute(
        """SELECT start, encounter_class, COALESCE(NULLIF(reason_description,''),description)
           FROM encounters WHERE patient_id=? AND start<?
           ORDER BY start DESC LIMIT 8""",
        (pid, start),
    ).fetchall()
    # Medication history over the year before discharge, with indications.
    med_history = con.execute(
        """SELECT description, start, stop, reason_description FROM medications
           WHERE patient_id=? AND start<? ORDER BY start DESC LIMIT 20""",
        (pid, stop or start),
    ).fetchall()

    def obs_lines(names):
        out = []
        for n in names:
            if n in obs_map:
                v, u = obs_map[n]
                out.append(f"  - {n}: {v} {u}".rstrip())
        return out

    vital_lines = obs_lines(VITALS)
    lab_lines = obs_lines(LABS)

    L = []
    L.append("DISCHARGE SUMMARY")
    L.append("=" * 60)
    L.append(f"Patient: {age}-year-old {gender}")
    L.append(f"Admission: {start[:10]} - {(stop or start)[:10]}")
    L.append(f"Encounter type: inpatient ({edesc})")
    L.append(f"Chief Complaint: {chief}")
    L.append("")
    L.append("History of Present Illness:")
    if new_conditions:
        L.append("The patient presented with " +
                 ", ".join(new_conditions) + ".")
    else:
        L.append(f"The patient was admitted for {chief}.")
    if vital_lines:
        L.append("Admission vitals:")
        L.extend(vital_lines)
    if lab_lines:
        L.append("Pertinent laboratory results:")
        L.extend(lab_lines)
    L.append("")
    L.append("Past Medical History:")
    if active_conditions or resolved_conditions:
        for c in active_conditions:
            L.append(f"  - {c} (active)")
        for c in resolved_conditions:
            L.append(f"  - {c} (resolved)")
    else:
        L.append("  - Noncontributory.")
    L.append("")
    L.append("Prior Encounters (most recent):")
    if prior_encs:
        for (s, ec, reason) in prior_encs:
            L.append(f"  - {s[:10]} [{ec}] {reason}")
    else:
        L.append("  - No prior encounters on record.")
    L.append("")
    L.append("Medication History (prior to this admission; some may be discontinued):")
    if med_history:
        for (mdesc, ms, mstop, mreason) in med_history:
            status = "ongoing" if active_at(mstop, stop or start) else "discontinued"
            ind = f" for {mreason}" if mreason else ""
            L.append(f"  - {mdesc} (started {ms[:10]}, {status}{ind})")
    else:
        L.append("  - None on record.")
    L.append("")
    L.append("Hospital Course:")
    if procs:
        L.append("Procedures performed during this admission:")
        L.extend(f"  - {p}" for p in procs)
    else:
        L.append("The patient was managed medically without procedural intervention.")
    if started_meds:
        L.append("Medications initiated or adjusted during this admission:")
        L.extend(f"  - {m}" for m in started_meds)
    L.append("")
    L.append("Discharge Diagnosis:")
    if active_conditions:
        L.extend(f"  - {c}" for c in active_conditions)
    else:
        L.append(f"  - {chief}")
    L.append("")
    L.append("Discharge Medications:")
    if discharge_meds:
        L.extend(f"  - {m}" for m in discharge_meds)
    else:
        L.append("  - None")
    L.append("")
    L.append("Assessment and Plan:")
    if active_conditions:
        for c in active_conditions:
            spec = followup_specialty(c)
            ref = (f" Recommend outpatient {spec} follow-up." if spec
                   else " Continue current management.")
            L.append(f"  # {c}")
            L.append(f"    Stable at discharge. Reviewed associated medications "
                     f"and adherence.{ref}")
    else:
        L.append(f"  # {chief}: resolved at discharge.")
    L.append("")
    L.append("Discharge Plan:")
    L.append("The patient is clinically stable for discharge. Follow-up in 2 weeks "
             "with primary care. Patient counseled on medication adherence and "
             "instructed to return for any worsening symptoms.")
    L.append("")
    note = "\n".join(L)

    truth = {
        "patient_id": pid,
        "index_encounter_id": enc_id,
        "discharge_date": (stop or start)[:10],
        "true_medications": discharge_meds,
        "true_active_conditions": active_conditions,
        "should_have_followup_for": [
            {"condition": c, "specialty": followup_specialty(c)}
            for c in active_conditions if followup_specialty(c)
        ],
    }
    return note, truth


def main():
    if not os.path.exists(DB_PATH):
        raise SystemExit("data/synthea.db not found. Run build_db.py first.")
    os.makedirs(NOTES_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        selected = select_patients(con)
        if len(selected) < N_NOTES:
            print(f"WARNING: only {len(selected)} patients met all criteria.")
        manifest = []
        ground_truth = {}
        lengths = []
        for i, (pid, enc) in enumerate(selected):
            note, truth = render_note(con, pid, enc)
            fname = f"note_{i:02d}.txt"
            path = os.path.join(NOTES_DIR, fname)
            with open(path, "w", encoding="utf-8") as f:
                f.write(note)
            manifest.append({
                "note_id": i,
                "patient_id": pid,
                "path": path,
                "chars": len(note),
                "approx_tokens": len(note) // 4,
            })
            ground_truth[pid] = truth
            lengths.append(len(note))
    finally:
        con.close()

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Wrote {len(manifest)} notes to {NOTES_DIR}/")
    print(f"Wrote {MANIFEST_PATH} and {GROUND_TRUTH_PATH}")
    if lengths:
        lengths.sort()
        print(f"Note length (chars): min={lengths[0]} "
              f"median={lengths[len(lengths)//2]} max={lengths[-1]}")
        print(f"Approx tokens: min={lengths[0]//4} "
              f"median={lengths[len(lengths)//2]//4} max={lengths[-1]//4}")


if __name__ == "__main__":
    main()
