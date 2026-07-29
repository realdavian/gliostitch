"""Back-trace LUMIERE vital status from survival time vs. last imaging contact.

OUTCOME: the manifest stores os_event = None for LUMIERE. This analysis is
evidence, not a rule the adapter applies. Every signal below points at "the
patients with a recorded survival time died", but that is an inference, and
inferring a vital status is a modelling decision — it belongs in gbm_os at
selection time, next to the rest of the censoring policy, not baked into the
manifest where the alternative becomes unreachable. Kept so that decision can
be made on the evidence rather than made again from scratch.

LUMIERE ships an overall-survival time but no event indicator. The release is
anonymised, so there are no calendar dates — but every study date is recorded
as a week offset from the first scan, for all 91 patients. That gives a usable
comparison:

    os_weeks   — survival from first resection (demographics)
    last_scan  — week of the patient's final imaging study (datacompleteness)

If survival is recorded well BEYOND the last scan, the patient was followed
past the end of imaging and a death date was known -> event observed.
If survival lands AT the last scan, the recorded time is just last-known-contact
-> indistinguishable from censoring.
If survival is BEFORE the last scan, something is inconsistent -> flag it.

Cohort: pre-op MRI Aug 2008-Dec 2013, follow-up to 2017 (Suter et al. 2022),
so the shortest possible administrative horizon is ~4 years / 208 weeks.
"""
from __future__ import annotations

import csv
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path("/mnt/disk1/datasets/LUMIERE")
WEEK_RE = re.compile(r"week-(\d+)")

# Shortest administrative horizon: enrolled Dec 2013, followed to end of 2017.
MIN_HORIZON_WEEKS = 4 * 52


def week_of(timepoint: str) -> int | None:
    m = WEEK_RE.search(timepoint)
    return int(m.group(1)) if m else None


def load() -> dict[str, dict]:
    last_scan: dict[str, int] = {}
    for r in csv.DictReader(open(ROOT / "LUMIERE-datacompleteness.csv")):
        w = week_of(r["Timepoint"])
        if w is not None:
            last_scan[r["Patient"]] = max(last_scan.get(r["Patient"], 0), w)

    rating_key = None
    last_rating: dict[str, tuple[int, str]] = {}
    for r in csv.DictReader(open(ROOT / "LUMIERE-ExpertRating-v202211.csv")):
        rating_key = rating_key or next(k for k in r if k.startswith("Rating ("))
        w = week_of(r["Date"])
        if w is None:
            continue
        prev = last_rating.get(r["Patient"])
        if prev is None or w >= prev[0]:
            last_rating[r["Patient"]] = (w, r[rating_key].strip())

    out: dict[str, dict] = {}
    for r in csv.DictReader(open(ROOT / "LUMIERE-Demographics_Pathology.csv")):
        p = r["Patient"]
        raw = r["Survival time (weeks)"].strip()
        os_weeks = float(raw) if raw.replace(".", "", 1).isdigit() else None
        out[p] = {
            "os_weeks": os_weeks,
            "last_scan": last_scan.get(p),
            "n_studies": sum(1 for _ in ()) or None,
            "last_rating": last_rating.get(p, (None, "n/a"))[1],
        }
    counts = Counter(
        r["Patient"] for r in csv.DictReader(open(ROOT / "LUMIERE-datacompleteness.csv"))
    )
    for p, rec in out.items():
        rec["n_studies"] = counts.get(p, 0)
    return out


def classify(rec: dict, margin: int) -> str:
    """Assign a vital-status basket.

    The margin is deliberately swept rather than fixed: the observed/ambiguous
    split moves from 81/1 at one week to 41/41 at thirteen, so any single
    threshold would be arbitrary. The conclusion rests instead on the two
    margin-independent signals — the absent censoring ceiling and the RANO
    trajectory — reported by main().
    """
    osw, last = rec["os_weeks"], rec["last_scan"]
    if osw is None:
        return "no_survival_recorded"
    if last is None:
        return "no_imaging_recorded"
    gap = osw - last
    if gap < -1:
        return "inconsistent(os_before_last_scan)"
    if gap >= margin:
        return "observed_death"
    return "ambiguous(os_at_last_contact)"


def main() -> None:
    data = load()
    print(f"patients: {len(data)}\n")

    gaps = [
        d["os_weeks"] - d["last_scan"]
        for d in data.values()
        if d["os_weeks"] is not None and d["last_scan"] is not None
    ]
    print("gap = os_weeks - last_scan_week")
    print(f"  n={len(gaps)}  median={statistics.median(gaps):.0f}  "
          f"mean={statistics.mean(gaps):.0f}  min={min(gaps):.0f}  max={max(gaps):.0f}")
    qs = sorted(gaps)
    print("  deciles:", [f"{qs[int(len(qs)*p/10)]:.0f}" for p in range(10)])
    print(f"  gap <= 0 : {sum(1 for g in gaps if g <= 0)}")
    print(f"  gap 1-4  : {sum(1 for g in gaps if 0 < g <= 4)}")
    print(f"  gap > 4  : {sum(1 for g in gaps if g > 4)}\n")

    print("sensitivity of the basket split to the margin:")
    for margin in (1, 2, 4, 8, 13):
        c = Counter(classify(d, margin) for d in data.values())
        print(f"  margin={margin:>2}wk  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print()

    MARGIN = 4
    print(f"--- adopting margin = {MARGIN} weeks ---")
    for p, d in data.items():
        d["basket"] = classify(d, MARGIN)
    c = Counter(d["basket"] for d in data.values())
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {k:34} {v:3}  ({v / len(data) * 100:.1f}%)")
    print()

    print("administrative-censoring check (shortest horizon ~208 wk):")
    surv = sorted(d["os_weeks"] for d in data.values() if d["os_weeks"] is not None)
    print(f"  max os_weeks = {surv[-1]:.0f}  ({surv[-1] * 7:.0f} d)")
    print(f"  os_weeks >= {MIN_HORIZON_WEEKS}: {sum(1 for s in surv if s >= MIN_HORIZON_WEEKS)}")
    print(f"  top 10 survival weeks: {[int(s) for s in surv[-10:]]}")
    print("  (a pile-up at one value would indicate a common study-close date)\n")

    print("last RANO rating by basket:")
    tab: dict[tuple[str, str], int] = Counter(
        (d["basket"], d["last_rating"]) for d in data.values()
    )
    baskets = sorted({b for b, _ in tab})
    ratings = sorted({r for _, r in tab})
    print(f"  {'rating':12}" + "".join(f"{b[:22]:>24}" for b in baskets))
    for r in ratings:
        print(f"  {r:12}" + "".join(f"{tab.get((b, r), 0):>24}" for b in baskets))
    print()

    # Restrict to the patients that actually reach the cohort.
    need = {"t1_skull_strip.nii", "ct1_skull_strip.nii",
            "t2_skull_strip.nii", "flair_skull_strip.nii"}
    usable = []
    rating_key = None
    for r in csv.DictReader(open(ROOT / "LUMIERE-ExpertRating-v202211.csv")):
        rating_key = rating_key or next(k for k in r if k.startswith("Rating ("))
        if r[rating_key].strip() != "Pre-Op":
            continue
        d = ROOT / "train" / r["Patient"] / r["Date"].strip()
        if d.is_dir() and need <= set(p.name for p in d.iterdir()):
            usable.append(r["Patient"])
    usable = sorted(set(usable))
    print(f"--- the {len(usable)} patients with a usable pre-op session ---")
    c = Counter(data[p]["basket"] for p in usable if p in data)
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {k:34} {v:3}")


if __name__ == "__main__":
    main()
