import argparse
import json
import os

_DRUG_STOP = {
    "oral", "tablet", "tablets", "capsule", "injection", "injectable",
    "suspension", "solution", "extended", "release", "hour", "hr", "pack",
    "day", "drug", "implant", "metered", "dose", "inhaler", "actuat", "film",
    "coated", "delayed", "sodium", "potassium", "hydrochloride", "bitartrate",
    "succinate", "sulfate", "phosphate", "calcium", "micrograms", "unit",
    "units", "ml", "mg", "mcg", "puff", "spray", "intravenous", "syringe",
    "prefilled", "auto", "injector", "patch", "transdermal", "system",
}
_COND_STOP = {"disorder", "finding", "disease", "of", "the", "with", "and",
              "in", "due", "to", "no", "first"}

def _tokens(s):
    return "".join(c if c.isalnum() else " " for c in s.lower()).split()

def drug_tokens(s):
    return {t for t in _tokens(s) if len(t) >= 4 and not t.isdigit()
            and t not in _DRUG_STOP}

def cond_tokens(s):
    return {t for t in _tokens(s) if len(t) >= 4 and not t.isdigit()
            and t not in _COND_STOP}

def _match(a_tokens, b_tokens):
    return bool(a_tokens & b_tokens)

def prf(true_items, pred_items, tokenizer):
    """Token-overlap precision/recall/F1 between two string lists. None if
    ground truth is empty (N/A)."""
    if not true_items:
        return None
    true_tok = [tokenizer(t) for t in true_items]
    pred_tok = [tokenizer(p) for p in pred_items]
    true_tok = [t for t in true_tok if t]
    pred_tok = [p for p in pred_tok if p]
    if not true_items:
        return None
    tp_recall = sum(1 for t in true_tok if any(_match(t, p) for p in pred_tok))
    recall = tp_recall / len(true_tok) if true_tok else 0.0
    if pred_tok:
        tp_prec = sum(1 for p in pred_tok if any(_match(p, t) for t in true_tok))
        precision = tp_prec / len(pred_tok)
    else:
        precision = 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "n_true": len(true_tok), "n_pred": len(pred_tok)}

_SPECIALTY_CANON = {
    "cardiology": "cardiology", "cardiac": "cardiology", "cardiologist": "cardiology",
    "endocrinology": "endocrinology", "endocrine": "endocrinology",
    "endocrinologist": "endocrinology",
    "nephrology": "nephrology", "renal": "nephrology", "nephrologist": "nephrology",
    "hematology": "hematology", "haematology": "hematology",
    "hematologist": "hematology",
    "pulmonology": "pulmonology", "pulmonary": "pulmonology",
    "pulmonologist": "pulmonology", "respiratory": "pulmonology",
    "primary care": "primary care", "pcp": "primary care",
    "primary": "primary care", "family medicine": "primary care",
    "internal medicine": "primary care", "general practice": "primary care",
}

def canon_specialty(s):
    low = s.strip().lower()
    if low in _SPECIALTY_CANON:
        return _SPECIALTY_CANON[low]
    for key, val in _SPECIALTY_CANON.items():
        if key in low:
            return val
    return low

def followup_score(required_specialties, recommended_specialties):
    """correct_specialties / required_specialties. None if no required."""
    req = {canon_specialty(s) for s in required_specialties}
    if not req:
        return None
    rec = {canon_specialty(s) for s in recommended_specialties}
    correct = len(req & rec)
    return {"score": correct / len(req), "n_required": len(req),
            "n_correct": correct, "required": sorted(req), "recommended": sorted(rec)}

def evaluate_row(row, gt):
    """Compute the four metric families for one patient row."""
    plan = row["plan"]
    truth = gt[row["patient_id"]]

    med_pred = [m.get("name", "") for m in
                plan.get("medication_reconciliation", {}).get("discharge_medications", [])]
    prob_pred = [p.get("name", "") for p in
                 plan.get("problem_list", {}).get("active_problems", [])]
    fu_specialties = [a.get("specialty", "") for a in
                      plan.get("follow_up", {}).get("appointments", [])]

    med = prf(truth["true_medications"], med_pred, drug_tokens)
    prob = prf(truth["true_active_conditions"], prob_pred, cond_tokens)
    required = [f["specialty"] for f in truth["should_have_followup_for"]]
    fu = followup_score(required, fu_specialties)

    v = row.get("validity", {})
    return {
        "medication": med, "problem": prob, "followup": fu,
        "json_valid_rate": (v.get("n_json_valid", 0) / v["n_calls"]
                            if v.get("n_calls") else None),
        "schema_valid_rate": (v.get("n_schema_valid", 0) / v["n_calls"]
                             if v.get("n_calls") else None),
    }

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def aggregate_eval(per_patient):
    def field(metric, key):
        return _mean([p[metric][key] for p in per_patient
                      if p[metric] is not None])
    med_f1 = field("medication", "f1")
    prob_f1 = field("problem", "f1")
    fu = _mean([p["followup"]["score"] for p in per_patient
                if p["followup"] is not None])
    worker_f1s = [x for x in [med_f1, prob_f1, fu] if x is not None]
    return {
        "medication": {"precision": field("medication", "precision"),
                       "recall": field("medication", "recall"),
                       "f1": med_f1,
                       "n_evaluated": sum(1 for p in per_patient
                                          if p["medication"] is not None)},
        "problem": {"precision": field("problem", "precision"),
                    "recall": field("problem", "recall"),
                    "f1": prob_f1,
                    "n_evaluated": sum(1 for p in per_patient
                                       if p["problem"] is not None)},
        "followup": {"appropriateness": fu,
                     "n_evaluated": sum(1 for p in per_patient
                                        if p["followup"] is not None)},
        "json_validity_rate": _mean([p["json_valid_rate"] for p in per_patient]),
        "schema_validity_rate": _mean([p["schema_valid_rate"] for p in per_patient]),
        "mean_worker_f1": sum(worker_f1s) / len(worker_f1s) if worker_f1s else None,
    }

def load_rows(results_dir, config):
    path = os.path.join(results_dir, f"raw_{config}.jsonl")
    if not os.path.exists(path):
        return None
    return [json.loads(line) for line in open(path, encoding="utf-8")]

def plot_latency_vs_length(rows_by_config, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import numpy as np
    except Exception:
        np = None

    colors = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}
    labels = {"A": "A: single-shot", "B": "B: agent, cache OFF",
              "C": "C: agent, cache ON"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for config, rows in rows_by_config.items():
        if not rows:
            continue
        x = [r["note_tokens"] for r in rows]
        y = [r["wall_clock_s"] for r in rows]
        ax.scatter(x, y, s=28, alpha=0.6, color=colors.get(config),
                   label=labels.get(config, config))
        if np is not None and len(x) >= 2:
            xs = np.array(x, dtype=float)
            ys = np.array(y, dtype=float)
            order = xs.argsort()
            coef = np.polyfit(xs, ys, 1)
            ax.plot(xs[order], np.poly1d(coef)(xs[order]),
                    color=colors.get(config), lw=1.5, alpha=0.9)
    ax.set_xlabel("Input note length (tokens)")
    ax.set_ylabel("End-to-end latency (s)")
    ax.set_title("End-to-end latency vs note length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")

def plot_pareto(eval_by_config, latency_by_config, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for config, ev in eval_by_config.items():
        f1 = ev.get("mean_worker_f1")
        lat = latency_by_config.get(config)
        if f1 is None or lat is None:
            continue
        ax.scatter(lat, f1, s=90, color=colors.get(config), zorder=3)
        ax.annotate(f"  {config}", (lat, f1), fontsize=11, va="center")
    ax.set_xlabel("Median end-to-end latency (s)")
    ax.set_ylabel("Mean worker F1 (med / problem / follow-up)")
    ax.set_title("Quality vs latency (Pareto)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ground-truth", default=os.path.join("data", "ground_truth.json"))
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--plots-dir", default=None,
                    help="defaults to <results-dir>/plots")
    args = ap.parse_args()

    gt = json.load(open(args.ground_truth, encoding="utf-8"))
    plots_dir = args.plots_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    evaluation = {}
    rows_by_config = {}
    median_latency = {}
    for config in ["A", "B", "C"]:
        rows = load_rows(args.results_dir, config)
        if rows is None:
            continue
        rows_by_config[config] = rows
        per_patient = [evaluate_row(r, gt) for r in rows]
        evaluation[config] = aggregate_eval(per_patient)
        lats = sorted(r["wall_clock_s"] for r in rows)
        median_latency[config] = lats[len(lats) // 2] if lats else None

    out_eval = os.path.join(args.results_dir, "evaluation.json")
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)
    print(f"Wrote {out_eval} (configs: {list(evaluation.keys())})")

    print("\n=== Worker x Config table ===")
    header = f"{'metric':22s}" + "".join(f"{c:>10s}" for c in evaluation)
    print(header)
    for label, path in [("medication F1", ("medication", "f1")),
                        ("problem F1", ("problem", "f1")),
                        ("followup approp.", ("followup", "appropriateness")),
                        ("mean worker F1", ("mean_worker_f1",)),
                        ("schema validity", ("schema_validity_rate",))]:
        line = f"{label:22s}"
        for c in evaluation:
            v = evaluation[c]
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            line += f"{(v if v is not None else float('nan')):10.3f}"
        print(line)

    if rows_by_config:
        plot_latency_vs_length(
            rows_by_config, os.path.join(plots_dir, "latency_vs_length.pdf"))
        plot_pareto(evaluation, median_latency,
                    os.path.join(plots_dir, "pareto.pdf"))

if __name__ == "__main__":
    main()
