#!/usr/bin/env python3
"""Step 46: read-only evidence-to-manuscript audit for CAJO.
Python 3.10+, standard library. No training, inference, checkpoint loading,
sealed-test reads, or modification of the experimental protocol.
"""
import argparse, collections, csv, datetime as dt, hashlib, json, math, re, statistics, sys, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

TASKS=("CIFAR10","CIFAR100","SpeechCommands_v0.02")
SEEDS=(42,43,44)
EXPECTED_CONFIG="56361ba3ef2cb530d54a923cf4f5b3758a22bee82a9119822395cef0be9bbdc5"
CONF="Optimization_Protocol/Final_Experiment_Config/cajo_experiment_config.json"
STEP43="CAJO_Step43_Experimental_Execution"
STEP44="CAJO_Step44_Controlled_Training"
STEP38="CAJO_Step38_Integrated_Closure_v1"
MAX_BYTES=60_000_000
REQUIREMENTS=[
 ("R01","§2.1 Eq.1–2","Validation accuracy by task/seed","candidate_accuracy","required"),
 ("R02","§2.1 Eq.3–5","Native inference peak memory including packed weights, live activations and workspaces","native_memory","blocking"),
 ("R03","§2.1 Eq.6–7","Target-CPU latency measurements and protocol","latency","required"),
 ("R04","§2.1 Eq.8–9","Qualified per-inference energy measurement consistent with claimed instrumentation","energy","blocking"),
 ("R05","§2.1 Eq.10–11; §2.2 Eq.16–17","Hardware, data, retraining and teacher admissibility evidence","admissibility","required"),
 ("R06","§2.2 Eq.12–14","Architecture, layerwise pruning and bit-width decisions with construction provenance","configuration","required"),
 ("R07","§2.2 Eq.15","KD versus CE matched controlled training","kd_ablation","required"),
 ("R08","§2.2 Eq.12–17","Pruning versus no-pruning controlled training","pruning_ablation","required"),
 ("R09","§2.2 Eq.12–17","FP32 versus INT8 matched evaluation","quant_ablation","required"),
 ("R10","§2.2 Eq.12–17","Architecture/NAS isolated ablation with matched budget","nas_ablation","required"),
 ("R11","§2.2; §2.3 Eq.19","Distinct joint versus sequential training/search with matched budgets","joint_sequential","blocking"),
 ("R12","§2.3 Eq.19","Prespecified candidate enumeration/sampling policy, seeds and maximum search budget","search_budget","required"),
 ("R13","§2.3 Eq.18–22","Complete verified five-objective vectors and feasibility filtering","five_objective","blocking"),
 ("R14","§2.3 Eq.21–22","Feasible five-objective Pareto front per task","pareto","blocking"),
 ("R15","§2.3 Eq.23","Selection preference agrees with locked protocol and is frozen before testing","selection","blocking"),
 ("R16","§3–4 described in Introduction","Baseline comparisons on identical splits, hardware and inference protocol","baselines","required"),
 ("R17","§3–4 described in Introduction","Paired seed-level comparisons and uncertainty statistics","statistics","required"),
 ("R18","§3–4 described in Introduction","Final held-out evaluation after frozen selection","sealed_test","blocking"),
 ("R19","§2.1 Eq.1 vs evaluation protocol","Validation/test terminology consistency","test_wording","text"),
 ("R20","§2.3 Eq.23 vs locked selection rule","Selection-priority wording consistency","selection_wording","text"),
 ("R21","§2.1 Eq.8–9 vs power telemetry","Energy-measurement wording consistency","energy_wording","text"),
 ("R22","Declarations: Data/Code/Reporting","Experimental-study declarations consistent with performed work","declarations","text"),
]
def safe_read(path,kind):
 if not path.is_file():return None
 if path.stat().st_size>MAX_BYTES:raise ValueError("File exceeds safe read limit: "+str(path))
 if kind=="json":return json.loads(path.read_text(encoding="utf-8-sig"))
 with path.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def value(d,*names):
 for k in names:
  x=d.get(k)
  if x is not None and str(x).strip() not in ("","NOT_MEASURED","None","null"):return x
 return None
def number(v):
 try:
  x=float(v)
  return x if math.isfinite(x) else None
 except (ValueError,TypeError):return None
def docx_text(path):
 if not path or not path.is_file():return ""
 ns="{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
 with zipfile.ZipFile(path) as z:
  root=ET.fromstring(z.read("word/document.xml"))
 lines=[]
 for para in root.iter(ns+"p"):
  lines.append("".join(t.text or "" for t in para.iter(ns+"t")))
 return "\n".join(lines)
def stats(vals):
 if not vals:return None
 return {"n":len(vals),"mean":round(statistics.mean(vals),6),
         "sd":round(statistics.stdev(vals),6) if len(vals)>1 else None,
         "min":round(min(vals),6),"max":round(max(vals),6)}
def write_csv(p,rows,cols):
 with p.open("w",newline="",encoding="utf-8-sig") as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore");w.writeheader();w.writerows(rows)
def classify(ok,partial=False):
 return "EVIDENCED" if ok else "PARTIAL" if partial else "MISSING_OR_UNVERIFIED"
def main():
 ap=argparse.ArgumentParser(description="Audit CAJO project evidence against current manuscript methods, without running experiments")
 ap.add_argument("--root",type=Path,default=Path(".."),help="New_Experiments project directory")
 ap.add_argument("--paper",type=Path,help="Manuscript .docx (optional; for text consistency checks)")
 ap.add_argument("--out",type=Path,help="Audit output directory (default: Reports/CAJO_Manuscript_Evidence_Audit)")
 args=ap.parse_args()
 root=args.root.resolve();reports=root/"Reports";codes=root/"Codes"
 if not reports.is_dir():ap.error("Missing Reports directory: "+str(reports))
 out=(args.out or reports/"CAJO_Manuscript_Evidence_Audit").resolve()
 if out==reports:ap.error("--out must be a separate directory")
 out.mkdir(parents=True,exist_ok=True)
 problems=[];sources={}
 def get(rel,kind="csv"):
  p=reports/rel
  try:d=safe_read(p,kind)
  except (OSError,ValueError,UnicodeError,json.JSONDecodeError,csv.Error) as e:
   problems.append({"source":rel,"error":str(e)});d=None
  sources[rel]={"exists":p.is_file(),"rows":len(d) if isinstance(d,list) else None,
                "sha256":digest(p) if p.is_file() and p.stat().st_size<=MAX_BYTES else None}
  return d
 cfg=get(CONF,"json")
 cfg_ok=sources[CONF]["sha256"]==EXPECTED_CONFIG
 candidates=get(STEP43+"/measured_validation_latency_storage.csv") or []
 precision_pairs=get(STEP43+"/paired_precision_ablation.csv") or []
 provisional=get(STEP43+"/provisional_three_objective_pareto.csv") or []
 baseline=get("Baseline_Resource_Profiling/baseline_resource_comparison.csv") or []
 energy=get("CAJO_Step34_Audit/energy_models.csv") or []
 gate=get(STEP38+"/evidence_gate.json","json") or {}
 valid5=get(STEP38+"/validated_objectives_input.csv") or []
 frozen=get("CAJO_Step35_Closure_v1/selection_gate.json","json") or {}
 # Read individual Step44 results directly, without relying on missing aggregate CSVs.
 raw=[]
 step44dir=reports/STEP44
 if step44dir.is_dir():
  for p in sorted(step44dir.glob("*/result.json")):
   try:
    d=safe_read(p,"json")
    if not isinstance(d,dict):continue
    label=p.parent.name
    parts=label.split("__")
    task=value(d,"task") or (parts[0] if parts else None)
    seed=value(d,"seed")
    if seed is None:
     m=re.search(r"__s(\d+)(?:__|$)",label);seed=int(m.group(1)) if m else None
    pruning=re.search(r"__P(\d+)(?:__|$)",label)
    kd=re.search(r"__(KD|CE)(?:__|$)",label)
    order=re.search(r"__(joint|sequential)(?:__|$)",label)
    raw.append({"task":task,"seed":int(seed) if seed is not None else None,
      "pruning_pct":int(pruning.group(1)) if pruning else None,
      "loss":kd.group(1) if kd else None,"order":order.group(1) if order else None,
      "accuracy_pct":number(value(d,"best_validation_accuracy_pct","validation_accuracy_pct")),
      "source":p.relative_to(root).as_posix(),"experiment_id":label})
   except (OSError,ValueError,TypeError,json.JSONDecodeError) as e:
    problems.append({"source":str(p),"error":str(e)})
 # All six arms x three tasks x three seeds, inferred from labels and result values.
 bykey=collections.defaultdict(list)
 for r in raw:
  key=(r["task"],r["seed"],r["pruning_pct"],r["loss"],r["order"])
  bykey[key].append(r)
 dupes=[str(k) for k,v in bykey.items() if len(v)>1]
 def matched(p1,p2,contrast):
  rows=[]
  for task in TASKS:
   for seed in SEEDS:
    left=bykey.get((task,seed,*p1),[])
    right=bykey.get((task,seed,*p2),[])
    if len(left)!=1 or len(right)!=1:continue
    a,b=left[0],right[0]
    if a["accuracy_pct"] is None or b["accuracy_pct"] is None:continue
    rows.append({"contrast":contrast,"task":task,"seed":seed,
      "arm_a":a["experiment_id"],"arm_b":b["experiment_id"],
      "accuracy_a_pct":a["accuracy_pct"],"accuracy_b_pct":b["accuracy_pct"],
      "delta_a_minus_b_pp":round(a["accuracy_pct"]-b["accuracy_pct"],6),
      "source_a":a["source"],"source_b":b["source"]})
  return rows
 comparisons=[]
 comparisons+=matched((0,"KD","joint"),(0,"CE","joint"),"KD minus CE; no pruning")
 comparisons+=matched((25,"KD","joint"),(25,"CE","joint"),"KD minus CE; joint-order P25")
 comparisons+=matched((25,"KD","sequential"),(25,"CE","sequential"),"KD minus CE; sequential-order P25")
 comparisons+=matched((25,"CE","joint"),(0,"CE","joint"),"P25 minus P0; CE")
 comparisons+=matched((25,"KD","joint"),(0,"KD","joint"),"P25 minus P0; KD")
 comparisons+=matched((25,"CE","joint"),(25,"CE","sequential"),"joint minus sequential; CE")
 comparisons+=matched((25,"KD","joint"),(25,"KD","sequential"),"joint minus sequential; KD")
 comparison_summary=[]
 for (contrast,task),rs in sorted(collections.defaultdict(list, {k:[r for r in comparisons if (r["contrast"],r["task"])==k] for k in {(r["contrast"],r["task"]) for r in comparisons}}).items()):
  ss=stats([r["delta_a_minus_b_pp"] for r in rs])
  comparison_summary.append({"contrast":contrast,"task":task,**(ss or {})})
 # Do not confuse three matched seeds with significance or full 4-component joint algorithm.
 candidate_keys={(value(r,"task","dataset"),value(r,"candidate_id"),value(r,"precision")) for r in candidates}
 threeD=sum(all(number(value(r,*ks)) is not None for ks in (
   ("validation_accuracy_pct","accuracy_pct"),("storage_bytes",),("latency_mean_ms",))) for r in candidates)
 energy_status=collections.Counter(str(value(r,"energy_status") or "unknown") for r in energy)
 baseline_missing=sum(any(value(r,k) is None for k in ("target_latency_ms","target_energy_mj","target_peak_memory_bytes")) for r in baseline)
 check=gate.get("checks",{}) if isinstance(gate.get("checks"),dict) else {}
 # Project-level source provenance: existence and hashes only, never execution.
 source_code={}
 for label,globpat in (
  ("architecture_search","*architecture_search_space*.py"),
  ("joint_search","*matched_joint_search*.py"),
  ("final_training","*final_joint_training*.py"),
  ("closure_algorithm","*final_closure*.py"),
  ("quantization","*int8_experiments*.py"),
  ("controlled_training","44_cajo_experimental_core*.py"),
  ("memory_meter","*targeted_memory_energy_acquisition*.py"),
  ("energy_meter","*hwinfo_energy*.py")):
  paths=sorted(codes.glob(globpat)) if codes.is_dir() else []
  source_code[label]=[{"path":x.relative_to(root).as_posix(),"sha256":digest(x)}
                      for x in paths if x.is_file() and x.stat().st_size<=MAX_BYTES]
 paper=docx_text(args.paper.resolve()) if args.paper else ""
 lower=paper.lower()
 # Text-only checks are source-grounded and require a manuscript; no guesses from file absence.
 flags={
  "test_wording":bool(re.search(r"number of correctly classified test samples|n_test",lower)),
  "selection_wording":bool("energy-critical deployment selects" in lower or "ties are resolved using latency, memory, and prediction error" in lower),
  "energy_wording":bool("measured supply voltage and current" in lower),
  "declarations":bool("no original software implementation" in lower or "no new datasets were generated or analyzed" in lower or "conceptual synthesis" in lower),
 }
 # Evidence statuses: every status carries an explicit reason and provenance.
 def item(key):
  if key=="candidate_accuracy":
   return classify(threeD==36,threeD>0),f"{threeD}/{len(candidates)} candidate rows have validation accuracy, storage and latency",STEP43
  if key=="native_memory":
   return classify(check.get("native_memory") is True,False),"Independent native-memory verification gate: "+str(check.get("native_memory")),STEP38
  if key=="latency":
   return classify(threeD==36,threeD>0),f"{threeD} candidate rows have latency; benchmarking protocol still requires manuscript documentation",STEP43
  if key=="energy":
   return classify(check.get("energy") is True,len(energy)>0),f"{len(energy)} energy summary rows; status counts={dict(energy_status)}; qualified gate={check.get('energy')}",STEP38
  if key=="admissibility":
   return "PARTIAL" if cfg_ok else "MISSING_OR_UNVERIFIED","Locked configuration present; no verified contextual eligibility decision table is established",CONF
  if key=="configuration":
   return "PARTIAL" if candidates else "MISSING_OR_UNVERIFIED","Candidate exports exist; complete layerwise decisions and four-component construction not verified",STEP43
  if key=="kd_ablation":
   n=sum(r["contrast"]=="KD minus CE; no pruning" for r in comparisons)
   return classify(n==9,n>0),f"{n}/9 seed-matched unpruned KD-versus-CE comparisons",STEP44
  if key=="pruning_ablation":
   n=sum(r["contrast"]=="P25 minus P0; CE" for r in comparisons)
   return classify(n==9,n>0),f"{n}/9 seed-matched CE pruning contrasts",STEP44
  if key=="quant_ablation":
   return classify(len(precision_pairs)==18,len(precision_pairs)>0),f"{len(precision_pairs)}/18 FP32–INT8 comparison rows; same-export PTQ contrast only",STEP43
  if key=="nas_ablation":
   return "MISSING_OR_UNVERIFIED","No independently matched architecture-only ablation has been established by this audit",STEP44
  if key=="joint_sequential":
   n=sum(r["contrast"]=="joint minus sequential; KD" for r in comparisons)
   return "PARTIAL" if n else "MISSING_OR_UNVERIFIED",f"{n}/9 matched operation-order KD contrasts; not proof of full joint pruning+quantization+KD+NAS algorithm",STEP44
  if key=="search_budget":
   return "PARTIAL" if cfg_ok else "MISSING_OR_UNVERIFIED","Locked config exists; independent proof of exact search-policy/budget execution is not established",CONF
  if key=="five_objective":
   return classify(check.get("five_objective_complete") is True and len(valid5)>0,len(valid5)>0),f"{len(valid5)} validated objective rows; Step38 gate={check.get('five_objective_complete')}",STEP38
  if key=="pareto":
   return classify(check.get("five_objective_complete") is True and check.get("three_tasks_feasible") is True,len(provisional)>0),f"{len(provisional)} provisional three-objective frontier rows; no certified five-objective frontier",STEP38
  if key=="selection":
   return classify(bool(frozen.get("selection_frozen")) and check.get("five_objective_complete") is True,False),"Selection must follow locked preference and verified feasible Pareto; freeze not demonstrated",STEP38
  if key=="baselines":
   return "PARTIAL" if baseline else "MISSING_OR_UNVERIFIED",f"{len(baseline)} baseline profiles; {baseline_missing} lack one or more target latency/energy/memory measurements","Baseline_Resource_Profiling"
  if key=="statistics":
   return "PARTIAL" if comparisons else "MISSING_OR_UNVERIFIED",f"{len(comparisons)} reconstructed paired seed contrasts; descriptive SD computed, inference/effect uncertainty not yet established",STEP44
  if key=="sealed_test":
   return "EVIDENCED" if gate.get("sealed_test_access")=="COMPLETED_AFTER_FREEZE" else "MISSING_OR_UNVERIFIED",f"Sealed-test state={gate.get('sealed_test_access','unknown')}; do not open until selection frozen",STEP38
  if key in flags:
   if not args.paper:return "NOT_CHECKED","Provide --paper path to inspect manuscript wording","manuscript"
   return ("TEXT_CONFLICT" if flags[key] else "NOT_DETECTED"),("Potential wording conflict located; reconcile to locked protocol" if flags[key] else "No predefined conflict phrase detected; manual review still needed"),"manuscript"
  return "NOT_CHECKED","No evaluator defined",""
 audit=[]
 for rid,section,expected,key,critical in REQUIREMENTS:
  status,why,src=item(key)
  audit.append({"id":rid,"manuscript_location":section,"required_evidence":expected,
                "status":status,"criticality":critical,"finding":why,"source":src})
 # Prioritize by manuscript dependencies, not arbitrary completion percentages.
 gaps=[
  {"priority":1,"task":"Resolve exact native inference memory, including packed INT8 weights and workspaces","requirements":["R02"],"type":"measurement"},
  {"priority":2,"task":"Resolve CPU energy uncertainty and reconcile measurement scope with Eqs. 8–9","requirements":["R04","R21"],"type":"measurement+manuscript"},
  {"priority":3,"task":"Establish complete joint-versus-sequential four-technique comparison and matched NAS ablations","requirements":["R10","R11"],"type":"experiment"},
  {"priority":4,"task":"Complete matched baseline resource measurements and contextual admissibility evidence","requirements":["R05","R06","R12","R16"],"type":"evidence_or_experiment"},
  {"priority":5,"task":"Build verified five-objective vectors, feasibility filters and Pareto front","requirements":["R13","R14"],"type":"analysis_after_measurements"},
  {"priority":6,"task":"Reconcile selection rule; freeze selection, then run sealed test once","requirements":["R15","R18","R19","R20"],"type":"analysis_then_test"},
  {"priority":7,"task":"Complete seed-level statistical comparisons and update experimental declarations","requirements":["R17","R22"],"type":"analysis+manuscript"}
 ]
 missing={r["id"] for r in audit if r["status"] not in ("EVIDENCED",)}
 gaps=[g for g in gaps if any(r in missing for r in g["requirements"])]
 report={"schema":"cajo_manuscript_evidence_audit_v1","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
  "root":str(root),"paper":str(args.paper.resolve()) if args.paper else None,
  "locked_config_sha256_expected":EXPECTED_CONFIG,"locked_config_sha256_observed":sources[CONF]["sha256"],
  "locked_config_matches":cfg_ok,
  "counts":{"candidate_exports":len(candidates),"three_objective_rows":threeD,"precision_pairs":len(precision_pairs),
    "step44_result_files":len(raw),"step44_unique_arms":len(bykey),"step44_duplicate_arm_keys":len(dupes),
    "step44_matched_contrasts":len(comparisons),"baseline_profiles":len(baseline),
    "energy_rows":len(energy),"provisional_3d_pareto_rows":len(provisional),
    "verified_5d_input_rows":len(valid5)},
  "step38_gate_checks":check,"sealed_test_access":gate.get("sealed_test_access"),"source_code_inventory":source_code,
  "requirements":audit,"next_evidence_tasks":gaps,"source_manifest":sources,
  "read_errors":problems,"duplicate_training_keys":dupes,
  "limitations":["File content and explicit verification gates take precedence over filenames.",
  "Matched pruning/recovery order is not equivalent to independently optimized four-component joint CAJO.",
  "Sensor energy and process RSS do not establish qualified energy or native model-only peak memory.",
  "Code source files are inventoried by hash only; source presence does not prove algorithm independence.",
  "No sealed test data, checkpoints, or training/inference functions were accessed.",
  "Absence of evidence is not proof the experiment never occurred; unrecognized schemas must be reviewed manually."]}
 (out/"manuscript_evidence_audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
 write_csv(out/"requirement_matrix.csv",audit,["id","manuscript_location","required_evidence","status","criticality","finding","source"])
 write_csv(out/"step44_matched_comparisons.csv",comparisons,["contrast","task","seed","arm_a","arm_b","accuracy_a_pct","accuracy_b_pct","delta_a_minus_b_pp","source_a","source_b"])
 write_csv(out/"step44_contrast_summary.csv",comparison_summary,["contrast","task","n","mean","sd","min","max"])
 write_csv(out/"step44_training_arms.csv",raw,["task","seed","pruning_pct","loss","order","accuracy_pct","experiment_id","source"])
 lines=["# CAJO manuscript-to-results evidence audit","",
   f"Project: `{root}`","",
   f"Locked configuration matches: **{cfg_ok}**","",
   "## Observed experimental coverage","",
   f"- Existing exports: {len(candidates)}; three-objective rows: {threeD}",
   f"- FP32/INT8 matched rows: {len(precision_pairs)}",
   f"- Step44 individual training results: {len(raw)}; distinct arm keys: {len(bykey)}",
   f"- Reconstructed paired training contrasts: {len(comparisons)}",
   f"- Baseline profiles: {len(baseline)}; energy summary rows: {len(energy)}",
   f"- Provisional three-objective frontier rows: {len(provisional)}",
   f"- Verified five-objective input rows: {len(valid5)}",
   f"- Sealed-test status: {gate.get('sealed_test_access','unknown')}","",
   "## Manuscript requirements versus available evidence","",
   "| ID | Paper section | Status | Finding |","|---|---|---|---|"]
 for r in audit:
  lines.append("| "+ " | ".join(str(r[k]).replace("|","/").replace("\n"," ") for k in ("id","manuscript_location","status","finding"))+" |")
 lines+=["","## Missing evidence in dependency order",""]
 for g in gaps:lines.append(f"{g['priority']}. **{g['task']}** ({', '.join(g['requirements'])})")
 lines+=["","## Matched training contrasts","",
   "| Comparison | Task | Seeds | Mean delta (pp) | SD (pp) |","|---|---|---:|---:|---:|"]
 for r in comparison_summary:
  lines.append(f"| {r['contrast']} | {r['task']} | {r['n']} | {r['mean']} | {r['sd']} |")
 lines+=["","## Safeguards","","No models were trained or evaluated; no test sets were opened; no input reports were modified.",
 "The audit does not turn partial evidence into a certified five-objective result."]
 (out/"audit_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 print(json.dumps({"audit_report":str(out/"audit_report.md"),"requirement_matrix":str(out/"requirement_matrix.csv"),
   "json":str(out/"manuscript_evidence_audit.json"),"counts":report["counts"],
   "status_counts":dict(collections.Counter(r["status"] for r in audit)),
   "config_matches":cfg_ok,"read_errors":len(problems)},indent=2))
if __name__=="__main__":sys.exit(main())
