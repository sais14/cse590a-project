"""build_db.py — Load Synthea sample CSVs into a SQLite history database.

Stdlib only (csv + sqlite3) so it runs identically on Colab and locally.

Input : synthea_data/csv/*.csv  (the apr2020 sample data set)
Output: data/synthea.db

Tables (normalized to snake_case so tools.py reads cleanly):
    patients(id, birthdate, deathdate, gender, race, ethnicity, first, last)
    encounters(id, patient_id, start, stop, encounter_class, code, description,
               reason_code, reason_description)
    conditions(patient_id, encounter_id, start, stop, code, description)
    medications(patient_id, encounter_id, start, stop, code, description,
                reason_code, reason_description)
    observations(patient_id, encounter_id, date, code, description, value, units, type)
    procedures(patient_id, encounter_id, date, code, description,
               reason_code, reason_description)
    careplans(id, patient_id, encounter_id, start, stop, code, description,
              reason_code, reason_description)

Cohort filter (per build_plan.md 2.2): patients with >=3 encounters AND
>=5 medication records. Only eligible patients' rows are loaded, which keeps
the DB small and tool queries fast. Eligible ids are also stored in the
`eligible_patients` table for downstream scripts.
"""
import csv
import os
import sqlite3
import sys

CSV_DIR = os.path.join("synthea_data", "csv")
DB_PATH = os.path.join("data", "synthea.db")

MIN_ENCOUNTERS = 3
MIN_MEDICATIONS = 5

# Allow large CSV fields (some Synthea notes/values are long).
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def read_csv(name):
    """Yield rows of <name>.csv as dicts keyed by the (verbatim) CSV header."""
    path = os.path.join(CSV_DIR, name + ".csv")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def compute_eligible():
    """Return the set of patient ids meeting the cohort filter."""
    enc_counts = {}
    for r in read_csv("encounters"):
        pid = r["PATIENT"]
        enc_counts[pid] = enc_counts.get(pid, 0) + 1
    med_counts = {}
    for r in read_csv("medications"):
        pid = r["PATIENT"]
        med_counts[pid] = med_counts.get(pid, 0) + 1
    eligible = {
        pid
        for pid in enc_counts
        if enc_counts.get(pid, 0) >= MIN_ENCOUNTERS
        and med_counts.get(pid, 0) >= MIN_MEDICATIONS
    }
    return eligible


def create_schema(con):
    con.executescript(
        """
        DROP TABLE IF EXISTS patients;
        DROP TABLE IF EXISTS encounters;
        DROP TABLE IF EXISTS conditions;
        DROP TABLE IF EXISTS medications;
        DROP TABLE IF EXISTS observations;
        DROP TABLE IF EXISTS procedures;
        DROP TABLE IF EXISTS careplans;
        DROP TABLE IF EXISTS eligible_patients;

        CREATE TABLE patients(
            id TEXT PRIMARY KEY, birthdate TEXT, deathdate TEXT, gender TEXT,
            race TEXT, ethnicity TEXT, first TEXT, last TEXT);
        CREATE TABLE encounters(
            id TEXT, patient_id TEXT, start TEXT, stop TEXT, encounter_class TEXT,
            code TEXT, description TEXT, reason_code TEXT, reason_description TEXT);
        CREATE TABLE conditions(
            patient_id TEXT, encounter_id TEXT, start TEXT, stop TEXT,
            code TEXT, description TEXT);
        CREATE TABLE medications(
            patient_id TEXT, encounter_id TEXT, start TEXT, stop TEXT,
            code TEXT, description TEXT, reason_code TEXT, reason_description TEXT);
        CREATE TABLE observations(
            patient_id TEXT, encounter_id TEXT, date TEXT, code TEXT,
            description TEXT, value TEXT, units TEXT, type TEXT);
        CREATE TABLE procedures(
            patient_id TEXT, encounter_id TEXT, date TEXT, code TEXT,
            description TEXT, reason_code TEXT, reason_description TEXT);
        CREATE TABLE careplans(
            id TEXT, patient_id TEXT, encounter_id TEXT, start TEXT, stop TEXT,
            code TEXT, description TEXT, reason_code TEXT, reason_description TEXT);
        CREATE TABLE eligible_patients(id TEXT PRIMARY KEY);
        """
    )


def load(con, eligible):
    def keep(r):
        return r["PATIENT"] in eligible

    # patients
    con.executemany(
        "INSERT OR IGNORE INTO patients VALUES (?,?,?,?,?,?,?,?)",
        (
            (r["Id"], r["BIRTHDATE"], r["DEATHDATE"], r["GENDER"],
             r["RACE"], r["ETHNICITY"], r["FIRST"], r["LAST"])
            for r in read_csv("patients")
            if r["Id"] in eligible
        ),
    )
    con.executemany(
        "INSERT INTO eligible_patients VALUES (?)", ((pid,) for pid in eligible)
    )
    # encounters
    con.executemany(
        "INSERT INTO encounters VALUES (?,?,?,?,?,?,?,?,?)",
        (
            (r["Id"], r["PATIENT"], r["START"], r["STOP"], r["ENCOUNTERCLASS"],
             r["CODE"], r["DESCRIPTION"], r["REASONCODE"], r["REASONDESCRIPTION"])
            for r in read_csv("encounters") if keep(r)
        ),
    )
    # conditions
    con.executemany(
        "INSERT INTO conditions VALUES (?,?,?,?,?,?)",
        (
            (r["PATIENT"], r["ENCOUNTER"], r["START"], r["STOP"],
             r["CODE"], r["DESCRIPTION"])
            for r in read_csv("conditions") if keep(r)
        ),
    )
    # medications
    con.executemany(
        "INSERT INTO medications VALUES (?,?,?,?,?,?,?,?)",
        (
            (r["PATIENT"], r["ENCOUNTER"], r["START"], r["STOP"], r["CODE"],
             r["DESCRIPTION"], r["REASONCODE"], r["REASONDESCRIPTION"])
            for r in read_csv("medications") if keep(r)
        ),
    )
    # observations
    con.executemany(
        "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?)",
        (
            (r["PATIENT"], r["ENCOUNTER"], r["DATE"], r["CODE"],
             r["DESCRIPTION"], r["VALUE"], r["UNITS"], r["TYPE"])
            for r in read_csv("observations") if keep(r)
        ),
    )
    # procedures
    con.executemany(
        "INSERT INTO procedures VALUES (?,?,?,?,?,?,?)",
        (
            (r["PATIENT"], r["ENCOUNTER"], r["DATE"], r["CODE"],
             r["DESCRIPTION"], r["REASONCODE"], r["REASONDESCRIPTION"])
            for r in read_csv("procedures") if keep(r)
        ),
    )
    # careplans
    con.executemany(
        "INSERT INTO careplans VALUES (?,?,?,?,?,?,?,?,?)",
        (
            (r["Id"], r["PATIENT"], r["ENCOUNTER"], r["START"], r["STOP"],
             r["CODE"], r["DESCRIPTION"], r["REASONCODE"], r["REASONDESCRIPTION"])
            for r in read_csv("careplans") if keep(r)
        ),
    )


def create_indexes(con):
    for tbl in ["encounters", "conditions", "medications",
                "observations", "procedures", "careplans"]:
        con.execute(
            f"CREATE INDEX idx_{tbl}_patient ON {tbl}(patient_id)"
        )
    con.execute("CREATE INDEX idx_obs_patient_desc ON observations(patient_id, description)")


def summarize(con):
    print("\n=== synthea.db summary ===")
    tables = ["patients", "encounters", "conditions", "medications",
              "observations", "procedures", "careplans", "eligible_patients"]
    for t in tables:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:20s} {n:>8d} rows")
    n_elig = con.execute("SELECT COUNT(*) FROM eligible_patients").fetchone()[0]
    print(f"\n  Eligible patients (>= {MIN_ENCOUNTERS} enc & "
          f">= {MIN_MEDICATIONS} meds): {n_elig}")


def main():
    if not os.path.isdir(CSV_DIR):
        sys.exit(f"ERROR: {CSV_DIR} not found. Unzip the Synthea sample first.")
    os.makedirs("data", exist_ok=True)
    print("Computing eligible cohort ...")
    eligible = compute_eligible()
    print(f"  {len(eligible)} patients pass the cohort filter.")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    try:
        create_schema(con)
        print("Loading filtered rows into SQLite ...")
        load(con, eligible)
        create_indexes(con)
        con.commit()
        summarize(con)
    finally:
        con.close()
    print(f"\nWrote {DB_PATH}")


if __name__ == "__main__":
    main()
