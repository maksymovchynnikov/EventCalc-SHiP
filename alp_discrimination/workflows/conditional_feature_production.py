from __future__ import annotations

import argparse, json, shutil, subprocess, sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alp_discrimination.templates.conditional_features import FEATURE_LABELS, load_conditional_feature_moments, validate_conditional_feature_moments
from alp_discrimination.templates.lifetime_banks import load_template_bank
from alp_discrimination.workflows import float_token
from alp_discrimination.workflows.conditional_feature_scan import persistent_threshold, run_conditional_feature_point
from alp_discrimination.progress import CheckpointMonitor
from alp_discrimination.statistics.basic import MINIMUM_OBSERVED_EVENTS

OBSERVABLES = ("energy","energy_mean_z","energy_mean_r_perp","energy_mean_z_r_perp")
SCREEN_SEEDS = (73241, 83244)
PROD_SEEDS = (73241, 83244, 93247, 103250, 113253)
SPARSE_GRIDS = (
    (1,2,3,4,5,6,8,10,15,20,30),
    (20,30,45,65,90,125,175,250,350),
    (250,350,500,700,1000,1400),
)

@dataclass(frozen=True)
class RangeResult:
    observable: str
    candidate_threshold: int
    lower: int
    upper: int
    full_grid: tuple[int,...]
    selection_grid: tuple[int,...]
    def as_dict(self):
        return {
            "observable": self.observable,
            "candidate_threshold": self.candidate_threshold,
            "lower": self.lower,
            "upper": self.upper,
            "full_grid": list(self.full_grid),
            "selection_grid": list(self.selection_grid),
        }

def resolve(repo: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (repo/path).resolve()

def moment_paths(root: Path, mass: float):
    token = float_token(mass)
    return (
        root/f"conditional_feature_moments_ma_{token}.npz",
        root/f"conditional_feature_moments_ma_{token}_quality.csv",
    )

def ensure_moments(bank_path, domain_path, root, workers, chunk):
    bank = load_template_bank(bank_path)
    root.mkdir(parents=True, exist_ok=True)
    mp, qp = moment_paths(root, float(bank.mass_gev))
    if mp.is_file() and qp.is_file():
        validate_conditional_feature_moments(load_conditional_feature_moments(mp), bank)
        return mp, qp
    if mp.exists() or qp.exists():
        raise FileExistsError("Partial permanent moment product exists; inspect it manually.")
    run_conditional_feature_point(
        bank_path=bank_path, output_dir=root, domain_path=domain_path,
        pseudoexperiments=1, seeds=(SCREEN_SEEDS[0],), workers=workers,
        chunk_size=chunk, event_counts=(MINIMUM_OBSERVED_EVENTS,), observables=("energy_mean_z_r_perp",),
        truth_grid="screening", moments_only=True,
    )
    validate_conditional_feature_moments(load_conditional_feature_moments(mp), bank)
    return mp, qp

def copy_moments(mp, qp, stage):
    stage.mkdir(parents=True, exist_ok=True)
    for src in (mp, qp):
        dst = stage/src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        elif dst.stat().st_size != src.stat().st_size:
            raise FileExistsError(f"Incompatible stage moment file: {dst}")

# Legacy checkpoint filenames retain 'pilot' so completed runs remain resumable.
def stage_summary(stage, mass):
    return stage/f"conditional_feature_pilot_summary_ma_{float_token(mass)}.json"

def stage_curve(stage, mass):
    return stage/f"conditional_feature_pilot_curves_ma_{float_token(mass)}.csv"

def run_scan(bank_path, domain_path, stage, mp, qp, observable, counts, pes, seeds, truth_grid, workers, chunk, resume):
    bank = load_template_bank(bank_path)
    summary = stage_summary(stage, float(bank.mass_gev))
    if summary.is_file():
        if not resume:
            raise FileExistsError(f"Completed stage exists; use --resume: {summary}")
        return json.loads(summary.read_text())
    copy_moments(mp, qp, stage)
    token = float_token(float(bank.mass_gev))
    monitor = CheckpointMonitor(
        checkpoint_dir=stage / "truth_parts",
        truth_table=(
            stage / f"conditional_feature_screening_truths_ma_{token}.csv"
        ),
        seeds_per_truth=len(tuple(seeds)),
        label=(
            f"m={float(bank.mass_gev):g} {bank.selection_name} "
            f"{observable} {truth_grid}"
        ),
    )
    monitor.start()
    try:
        return run_conditional_feature_point(
            bank_path=bank_path, output_dir=stage, domain_path=domain_path,
            pseudoexperiments=pes, seeds=tuple(seeds), workers=workers,
            chunk_size=chunk, event_counts=tuple(counts), observables=(observable,),
            truth_grid=truth_grid, reuse_moments=True,
        )
    finally:
        monitor.stop()

def read_curve(stage, mass, observable):
    frame = pd.read_csv(stage_curve(stage, mass))
    frame = frame[frame["observable"].astype(str)==observable].copy()
    if frame.empty:
        raise ValueError(f"No curve for {observable} in {stage}")
    return frame.sort_values("number_of_events", ignore_index=True)

def bracket(curve):
    ordered = curve.sort_values("number_of_events", ignore_index=True)
    threshold = persistent_threshold(ordered)
    if threshold is None:
        return None, int(ordered["number_of_events"].max()), None
    counts = ordered["number_of_events"].to_numpy(dtype=int)
    lower = counts[counts < threshold]
    return int(threshold), int(lower[-1]) if len(lower) else MINIMUM_OBSERVED_EVENTS - 1, int(threshold)

def refinement_grid(lower, upper):
    width = int(upper)-int(lower)
    if width <= 0:
        raise ValueError("Invalid bracket")
    if width <= 24:
        core = list(range(max(MINIMUM_OBSERVED_EVENTS,int(lower)), int(upper)+1))
    else:
        step = max(2, int(ceil(width/12)))
        core = list(range(max(MINIMUM_OBSERVED_EVENTS,int(lower)), int(upper)+1, step))
        if int(upper) not in core:
            core.append(int(upper))
    tail = [int(upper)+max(3,ceil(.05*upper)), int(upper)+max(6,ceil(.10*upper))]
    return tuple(sorted(set(int(x) for x in core+tail if x>=MINIMUM_OBSERVED_EVENTS)))

def full_grid(n):
    n = int(n)
    if n < MINIMUM_OBSERVED_EVENTS:
        raise ValueError("Observed event counts must be positive")
    # Preserve the validated N>=2 grid away from the low-count edge. When the
    # screening crossing is close to the edge, include N=1 explicitly.
    floor_count = MINIMUM_OBSERVED_EVENTS if n <= 3 else 2
    values = list(range(max(floor_count,n-8), n+16))
    values += [max(floor_count,n-20), max(floor_count,n-12), n+20, n+30, ceil(1.25*n), ceil(1.50*n)]
    return tuple(sorted(set(int(x) for x in values)))

def selection_grid(n, available):
    available = set(int(x) for x in available)
    result = tuple(
        x for x in (n-1,n,n+1)
        if x>=MINIMUM_OBSERVED_EVENTS and x in available
    )
    if n not in result:
        raise ValueError("Candidate threshold missing from final grid")
    return result

def rangefinder(bank_path, domain_path, root, mp, qp, observable, pes, workers, chunk, resume):
    bank = load_template_bank(bank_path)
    mass = float(bank.mass_gev)
    base = root/"rangefinder"/observable
    result_path = base/"rangefinder_result.json"
    if result_path.is_file() and resume:
        saved = json.loads(result_path.read_text())
        return RangeResult(
            observable=str(saved["observable"]),
            candidate_threshold=int(saved["candidate_threshold"]),
            lower=int(saved["lower"]),
            upper=int(saved["upper"]),
            full_grid=tuple(int(x) for x in saved["full_grid"]),
            selection_grid=tuple(int(x) for x in saved["selection_grid"]),
        )
    lower = upper = None
    for i, grid in enumerate(SPARSE_GRIDS):
        stage = base/f"sparse_{i:02d}"
        run_scan(bank_path,domain_path,stage,mp,qp,observable,grid,pes,SCREEN_SEEDS,"screening",workers,chunk,resume)
        threshold, lo, hi = bracket(read_curve(stage,mass,observable))
        lower = lo
        if threshold is not None:
            upper = hi
            break
    if lower is None or upper is None:
        raise RuntimeError(f"Could not bracket {observable} through N={SPARSE_GRIDS[-1][-1]}")
    i = 0
    while upper-lower > 1:
        stage = base/f"refine_{i:02d}"
        grid = refinement_grid(lower,upper)
        run_scan(bank_path,domain_path,stage,mp,qp,observable,grid,pes,SCREEN_SEEDS,"screening",workers,chunk,resume)
        threshold, lo, hi = bracket(read_curve(stage,mass,observable))
        if threshold is None:
            raise RuntimeError(f"Refinement lost crossing for {observable}; inspect {stage}")
        if (lo,hi)==(lower,upper):
            raise RuntimeError(f"Refinement stalled for {observable}: {lower}-{upper}")
        lower,upper = lo,hi
        i += 1
        if i > 8:
            raise RuntimeError("Too many refinement rounds")
    result = RangeResult(observable,int(upper),int(lower),int(upper),full_grid(upper),selection_grid(upper,full_grid(upper)))
    base.mkdir(parents=True,exist_ok=True)
    result_path.write_text(json.dumps(result.as_dict(),indent=2)+"\n")
    return result

def full_domain(bank_path,domain_path,root,mp,qp,rr,pes,workers,chunk,resume):
    bank = load_template_bank(bank_path)

    def evaluate(stage, counts):
        run_scan(
            bank_path,domain_path,stage,mp,qp,rr.observable,counts,pes,
            PROD_SEEDS,"all",workers,chunk,resume,
        )
        curve = read_curve(stage,float(bank.mass_gev),rr.observable)
        threshold = persistent_threshold(curve)
        if threshold is None:
            raise RuntimeError(f"Full-domain curve does not cross for {rr.observable}")
        tested = set(curve["number_of_events"].astype(int))
        if int(threshold) == max(tested):
            raise RuntimeError(
                f"Full-domain N90 at upper grid edge for {rr.observable}: {threshold}"
            )
        return curve,int(threshold),tested

    stage = root/"full_domain"/rr.observable
    curve,threshold,counts = evaluate(stage,rr.full_grid)
    if threshold == min(counts) and threshold > MINIMUM_OBSERVED_EVENTS:
        # Existing validated checkpoints commonly start at N=2. Preserve them
        # and run one versioned all-truth extension including N=1 instead of
        # overwriting a completed production result.
        extended = tuple(sorted(set(rr.full_grid) | {MINIMUM_OBSERVED_EVENTS}))
        stage = root/"full_domain_n1"/rr.observable
        curve,threshold,counts = evaluate(stage,extended)
    if threshold == min(counts) and threshold > MINIMUM_OBSERVED_EVENTS:
        raise RuntimeError(
            f"Full-domain N90 remains at an unresolved lower grid edge for {rr.observable}: {threshold}"
        )
    return stage,int(threshold)

def selected_audit(bank_path,mp,full_dir,root,observable,threshold,counts,pes,workers,chunk,resume):
    bank = load_template_bank(bank_path)
    token = float_token(float(bank.mass_gev))
    out = root/"selected"/observable
    out.mkdir(parents=True,exist_ok=True)
    summary = out/f"selected_5k_audit_summary_ma_{token}.json"
    if summary.is_file():
        if not resume:
            raise FileExistsError(f"Completed selected audit exists; use --resume: {summary}")
        return out,json.loads(summary.read_text())
    sel = selection_grid(int(threshold),counts)
    cmd = [
        sys.executable,"-m","alp_discrimination.workflows.conditional_feature_selected",
        "--full-domain-dir",str(full_dir),"--bank-path",str(bank_path),
        "--moments-path",str(mp),"--output-dir",str(out),
        "--observable",observable,"--pseudoexperiments",str(int(pes)),
        "--seeds",*[str(x) for x in PROD_SEEDS],
        "--event-counts",*[str(int(x)) for x in counts],
        "--selection-event-counts",*[str(int(x)) for x in sel],
        "--workers",str(int(workers)),"--chunk-size",str(int(chunk)),
    ]
    print(" ".join(cmd),flush=True)
    subprocess.run(cmd,check=True)
    return out,json.loads(summary.read_text())

def write_products(root,bank,rows):
    tables,plots = root/"tables",root/"plots"
    tables.mkdir(parents=True,exist_ok=True); plots.mkdir(parents=True,exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("observable",ignore_index=True)
    frame.to_csv(tables/"n90_by_observable.csv",index=False)
    (tables/"n90_by_observable.json").write_text(json.dumps(rows,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(7.6,4.8))
    x=np.arange(len(frame))
    ax.bar(x,frame["N90"].to_numpy(float))
    ax.set_xticks(x,[FEATURE_LABELS.get(o,o) for o in frame["observable"]],rotation=20,ha="right")
    ax.set_ylabel(r"Minimum observed events, $N_{90}$")
    ax.set_title(f"$m_a={float(bank.mass_gev):g}$ GeV, {bank.selection_name}")
    ax.grid(axis="y",alpha=.25); fig.tight_layout()
    fig.savefig(plots/"n90_observable_ablation.pdf")
    fig.savefig(plots/"n90_observable_ablation.png",dpi=180)
    plt.close(fig)

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-path",type=Path,required=True)
    p.add_argument("--domain-path",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--observables",nargs="+",choices=OBSERVABLES,default=list(OBSERVABLES))
    p.add_argument("--workers",choices=(1,2),type=int,default=2)
    p.add_argument("--chunk-size",type=int,default=30)
    p.add_argument("--screen-pseudoexperiments",type=int,default=500)
    p.add_argument("--full-domain-pseudoexperiments",type=int,default=2000)
    p.add_argument("--selected-pseudoexperiments",type=int,default=5000)
    p.add_argument("--stop-after",choices=("moments","rangefinder","full_domain","selected"),default="selected")
    p.add_argument("--resume",action="store_true")
    return p.parse_args(argv)

def main(argv=None):
    args=parse_args(argv)
    repo=Path.cwd().resolve()
    if not (repo/"alp_discrimination").is_dir():
        raise SystemExit("Run from the EventCalc-SHiP repository root")
    bank_path,domain_path,root=(resolve(repo,args.bank_path),resolve(repo,args.domain_path),resolve(repo,args.output_dir))
    if not bank_path.is_file() or not domain_path.is_file():
        raise FileNotFoundError("Bank or domain file missing")
    bank=load_template_bank(bank_path)
    root.mkdir(parents=True,exist_ok=True)
    mp,qp=ensure_moments(bank_path,domain_path,root/"moments",args.workers,args.chunk_size)
    if args.stop_after=="moments":
        print(mp); return
    ranges={}
    for obs in dict.fromkeys(args.observables):
        print(f"\n=== THRESHOLD SCAN {obs} ===", flush=True)
        ranges[obs]=rangefinder(bank_path,domain_path,root,mp,qp,obs,args.screen_pseudoexperiments,args.workers,args.chunk_size,args.resume)
    if args.stop_after=="rangefinder":
        return
    full={}
    for obs,rr in ranges.items():
        print(f"\n=== LIFETIME SCAN {obs} ===", flush=True)
        full[obs]=full_domain(bank_path,domain_path,root,mp,qp,rr,args.full_domain_pseudoexperiments,args.workers,args.chunk_size,args.resume)
    if args.stop_after=="full_domain":
        return
    rows=[]
    for obs,(fd,thr2k) in full.items():
        print(f"\n=== HIGH-STATISTICS VALIDATION {obs} ===", flush=True)
        active_curve = read_curve(fd,float(bank.mass_gev),obs)
        active_counts = tuple(sorted(set(active_curve["number_of_events"].astype(int))))
        out,s=selected_audit(bank_path,mp,fd,root,obs,thr2k,active_counts,args.selected_pseudoexperiments,args.workers,args.chunk_size,args.resume)
        thr5=s["persistent_thresholds"]["selected_5k"]
        rows.append({
            "mass_GeV":float(bank.mass_gev),"selection_name":str(bank.selection_name),
            "observable":obs,"N90":int(thr5),"full_domain_2k_N90":int(thr2k),
            "validation_level":"selected_5k_decision_audited",
            "omitted_truth_audit_passed":bool(s["final_omitted_truth_audit_passed"]),
            "recommend_selected_10k":bool(s["recommend_selected_10k"]),
            "selected_summary_path":str(out/f"selected_5k_audit_summary_ma_{float_token(bank.mass_gev)}.json"),
        })
    write_products(root,bank,rows)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":
    main()
