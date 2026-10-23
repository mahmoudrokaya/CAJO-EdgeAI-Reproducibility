#!/usr/bin/env python3
"""CAJO read-only results explorer. Python 3.10+, standard library only.
Never loads checkpoints, runs inference, trains, or opens sealed-test datasets.
Outputs a navigable HTML dashboard plus machine-readable tables.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics
import sys

TASKS=("CIFAR10","CIFAR100","SpeechCommands_v0.02")
CONFIG="Optimization_Protocol/Final_Experiment_Config/cajo_experiment_config.json"
CONFIG_SHA="56361ba3ef2cb530d54a923cf4f5b3758a22bee82a9119822395cef0be9bbdc5"
CANDIDATE_SOURCES=[
 "CAJO_Step43_Experimental_Execution/measured_validation_latency_storage.csv",
 "CAJO_Independent_Closure_v1/finalist_unique_seed_precision.csv",
 "CAJO_Final_Pipeline/final_cpu_per_seed.csv",
]
SPECIAL=[
 ("Step43 precision pairs","CAJO_Step43_Experimental_Execution/paired_precision_ablation.csv"),
 ("Step43 provisional frontier","CAJO_Step43_Experimental_Execution/provisional_three_objective_pareto.csv"),
 ("Step44 training arms","CAJO_Step44_Controlled_Training/completed_training.csv"),
 ("Step44 matched order","CAJO_Step44_Controlled_Training/matched_order_comparison.csv"),
 ("Baseline resources","Baseline_Resource_Profiling/baseline_resource_comparison.csv"),
 ("Energy models","CAJO_Step34_Audit/energy_models.csv"),
 ("Step38 objective inputs","CAJO_Step38_Integrated_Closure_v1/validated_objectives_input.csv"),
]
GATES=[
 "CAJO_Step38_Integrated_Closure_v1/evidence_gate.json",
 "CAJO_Step35_Closure_v1/selection_gate.json",
 "CAJO_Step43_Experimental_Execution/closure.json",
]
MAX_CSV_MB=50
MAX_JSON_MB=8

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def load_csv(p):
 if p.stat().st_size>MAX_CSV_MB*1024*1024:raise ValueError("CSV too large")
 with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def load_json(p):
 if p.stat().st_size>MAX_JSON_MB*1024*1024:raise ValueError("JSON too large")
 return json.loads(p.read_text(encoding="utf-8-sig"))
def num(x):
 try:
  if x is None or str(x).strip()=="" :return None
  v=float(x)
  return v if math.isfinite(v) else None
 except (ValueError,TypeError):return None
def val(row,*keys):
 for k in keys:
  if row.get(k) not in ("",None):return row[k]
 return None
def normalize(row,source):
 task=val(row,"task","dataset")
 if not task:
  cid=val(row,"candidate_id","experiment") or ""
  task=next((t for t in TASKS if cid.startswith(t)),None)
 cid=val(row,"candidate_id","experiment") or ""
 precision=val(row,"precision")
 if not precision:
  precision=next((x for x in ("FP32","INT8") if "__"+x in cid),None)
 acc=num(val(row,"validation_accuracy_pct","accuracy_pct","validation_accuracy"))
 error=num(val(row,"error_pct","validation_error_pct"))
 if error is None and acc is not None:error=100-acc
 storage=num(val(row,"storage_bytes","serialized_storage_bytes"))
 latency=num(val(row,"latency_mean_ms","cpu_latency_ms"))
 mem=num(val(row,"inference_memory_bytes","native_inference_memory_bytes"))
 energy=num(val(row,"energy_incremental_mj","energy_mj_per_inference"))
 return {"task":task,"candidate_id":cid,"precision":precision,"validation_accuracy_pct":acc,
         "validation_error_pct":error,"storage_bytes":storage,"latency_mean_ms":latency,
         "native_memory_bytes_unverified":mem,"energy_mj_unverified":energy,
         "source":source,"export_sha256":val(row,"export_sha256","source_sha256"),
         "seed":val(row,"seed")}
def read_if(path,reader,errors):
 if not path.is_file():return None
 try:return reader(path)
 except Exception as e:
  errors.append({"file":str(path),"error":str(e)});return None
def discover(reports,errors):
 """Bounded metadata inventory; no recursive JSON parsing of huge datasets."""
 inventory=[]
 for base,dirs,files in os.walk(reports):
  dirs[:]=[d for d in dirs if d not in (".git","__pycache__")]
  for name in files:
   p=Path(base)/name
   if p.suffix.lower() not in (".csv",".json",".png",".html",".md",".txt",".pt",".pth",".onnx",".tsv",".parquet"):continue
   try:
    st=p.stat()
    inventory.append({"path":p.relative_to(reports).as_posix(),"bytes":st.st_size,
                      "extension":p.suffix.lower(),"modified":dt.datetime.fromtimestamp(st.st_mtime,dt.timezone.utc).isoformat()})
   except OSError as e:errors.append({"file":str(p),"error":str(e)})
 return inventory
def table_file(out,name,records,fields):
 p=out/name
 with p.open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader()
  for row in records:w.writerow({k:row.get(k) for k in fields})
def jwrite(out,name,data):
 (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
def escape(x):return html.escape(str(x if x is not None else ""),quote=True)
def html_table(records,fields,limit=None):
 if limit is not None:records=records[:limit]
 head="".join("<th>"+escape(k)+"</th>" for k in fields)
 body="".join("<tr>"+"".join("<td>"+escape(r.get(k,""))+"</td>" for k in fields)+"</tr>" for r in records)
 return "<div class='scroll'><table><thead><tr>"+head+"</tr></thead><tbody>"+body+"</tbody></table></div>"
def main():
 ap=argparse.ArgumentParser(description="Explore existing CAJO Reports without changing experiments or opening test data")
 ap.add_argument("--root",type=Path,default=Path(".."),help="Project root containing Reports")
 ap.add_argument("--reports",type=Path,help="Direct Reports directory (overrides --root)")
 ap.add_argument("--out",type=Path,help="Output directory (default: <Reports>/CAJO_Results_Explorer)")
 a=ap.parse_args()
 reports=(a.reports or a.root/"Reports").resolve()
 if not reports.is_dir():ap.error("Reports directory does not exist: "+str(reports))
 out=(a.out or reports/"CAJO_Results_Explorer").resolve()
 if out==reports:ap.error("Output must not overwrite Reports")
 out.mkdir(parents=True,exist_ok=True)
 errors=[]
 inventory=discover(reports,errors)
 cfg=reports/CONFIG
 config_hash=sha(cfg) if cfg.is_file() else None
 config_match=(config_hash==CONFIG_SHA)
 source_info={}
 for name in CANDIDATE_SOURCES:
  p=reports/name
  rows=read_if(p,load_csv,errors)
  source_info[name]={"present":p.is_file(),"rows":len(rows) if rows is not None else None,
                     "sha256":sha(p) if p.is_file() else None,
                     "columns":list(rows[0]) if rows else []}
 primary_name=next((n for n in CANDIDATE_SOURCES if source_info[n]["rows"]),None)
 primary=load_csv(reports/primary_name) if primary_name else []
 candidates=[normalize(r,primary_name) for r in primary]
 task_counts={t:sum(x["task"]==t for x in candidates) for t in TASKS}
 precision_counts=dict(collections.Counter(x["precision"] or "unknown" for x in candidates))
 complete3=sum(all(x[k] is not None for k in ("validation_error_pct","storage_bytes","latency_mean_ms")) for x in candidates)
 # Do not infer certification from a numeric memory field; proxy/native attribution matters.
 grouped=[]
 for task in TASKS:
  for precision in ("FP32","INT8"):
   sub=[r for r in candidates if r["task"]==task and r["precision"]==precision]
   summary={"task":task,"precision":precision,"n":len(sub)}
   for key in ("validation_accuracy_pct","storage_bytes","latency_mean_ms"):
    vs=[r[key] for r in sub if r[key] is not None]
    summary[key+"_mean"]=round(statistics.mean(vs),6) if vs else None
    summary[key+"_min"]=min(vs) if vs else None
    summary[key+"_max"]=max(vs) if vs else None
   grouped.append(summary)
 # Capture unsummarized Step44 experiments as well, without opening model checkpoints.
 step44_raw=[]
 step44_root=reports/"CAJO_Step44_Controlled_Training"
 if step44_root.is_dir():
  for rp in sorted(step44_root.glob("*/result.json")):
   d=read_if(rp,load_json,errors)
   if isinstance(d,dict):
    step44_raw.append({"experiment_dir":rp.parent.name,
                       "task":d.get("task"),"seed":d.get("seed"),
                       "best_validation_accuracy_pct":d.get("best_validation_accuracy_pct"),
                       "operation_order":d.get("operation_order"),
                       "result_path":rp.relative_to(reports).as_posix()})
 tables={}
 for label,name in SPECIAL:
  rows=read_if(reports/name,load_csv,errors)
  tables[label]={"path":name,"present":rows is not None,"rows":len(rows) if rows is not None else None,
                 "columns":list(rows[0]) if rows else [],"data":rows or []}
 gates={}
 for name in GATES:
  p=reports/name
  data=read_if(p,load_json,errors)
  if isinstance(data,dict):
   gates[name]={k:data.get(k) for k in ("schema","config_sha256","locked_config_sha256","ready_to_freeze","selection_ready","selection_frozen","sealed_test_access","checks","verified_objective_vectors","feasible_vectors","issues") if k in data}
 # Independently summarize any Step44 training results without claiming algorithm certification.
 training=tables["Step44 training arms"]["data"]
 order=tables["Step44 matched order"]["data"]
 order_deltas=[num(r.get("joint_minus_sequential_pp")) for r in order]
 order_deltas=[v for v in order_deltas if v is not None]
 issues=[]
 if not config_match:issues.append("Locked config missing or SHA256 differs from expected; do not combine results as one protocol.")
 if not primary_name:issues.append("No candidate evaluation table found in known canonical locations.")
 if not training and not step44_raw:issues.append("No Step44 training results found (neither summary CSV nor individual result.json).")
 if not order:issues.append("No matched Step44 joint/sequential training results found.")
 if not tables["Step43 precision pairs"]["data"]:issues.append("No Step43 precision-ablation pairs found.")
 if not tables["Step38 objective inputs"]["data"]:issues.append("No validated Step38 five-objective input table found.")
 gate38=gates.get(GATES[0],{})
 checks=gate38.get("checks",{})
 if isinstance(checks,dict):
  for k,v in checks.items():
   if v is False:issues.append("Step38 evidence gate: "+k+" = false.")
 # Never scan or read test-related file content.
 result={"schema":"cajo_results_explorer_v1","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
  "reports_root":str(reports),"locked_config_sha256_expected":CONFIG_SHA,
  "locked_config_sha256_observed":config_hash,"locked_config_matches":config_match,
  "inventory_files":len(inventory),"candidate_source":primary_name,
  "candidate_rows":len(candidates),"candidate_rows_with_3_objectives":complete3,
  "task_counts":task_counts,"precision_counts":precision_counts,
  "step43_matched_precision_pairs":tables["Step43 precision pairs"]["rows"],
  "step43_provisional_frontier_rows":tables["Step43 provisional frontier"]["rows"],
  "step44_training_arms":len(training),"step44_raw_result_files":len(step44_raw),"step44_matched_order_pairs":len(order),
  "step44_mean_joint_minus_sequential_pp":statistics.mean(order_deltas) if order_deltas else None,
  "source_tables":source_info,"other_tables":{k:{q:v[q] for q in ("path","present","rows","columns")} for k,v in tables.items()},
  "evidence_gates":gates,"issues":issues,"read_errors":errors,
  "safety":"Read-only input; no checkpoint loads, test access, inference, or protocol edits. HTML is descriptive, not certification."}
 jwrite(out,"exploration_summary.json",result)
 fields=["task","candidate_id","precision","seed","validation_accuracy_pct","validation_error_pct","storage_bytes","latency_mean_ms","native_memory_bytes_unverified","energy_mj_unverified","export_sha256","source"]
 table_file(out,"candidate_evidence.csv",candidates,fields)
 table_file(out,"task_precision_summary.csv",grouped,["task","precision","n"]+[k+"_"+s for k in ("validation_accuracy_pct","storage_bytes","latency_mean_ms") for s in ("mean","min","max")])
 table_file(out,"file_inventory.csv",inventory,["path","bytes","extension","modified"])
 table_file(out,"step44_individual_results.csv",step44_raw,["experiment_dir","task","seed","best_validation_accuracy_pct","operation_order","result_path"])
 def section(title,body):return "<section><h2>"+escape(title)+"</h2>"+body+"</section>"
 stat_items=[("Files indexed",len(inventory)),("Candidate rows",len(candidates)),("Rows with 3 objectives",complete3),
             ("Step43 precision pairs",tables["Step43 precision pairs"]["rows"] or 0),
             ("Step44 completed arms",len(training)),("Step44 matched order pairs",len(order))]
 cards="<div class='cards'>"+"".join("<div class='card'><b>"+escape(v)+"</b><span>"+escape(k)+"</span></div>" for k,v in stat_items)+"</div>"
 sections=[section("Scope and experiment counts",cards+"<p>Configuration SHA256 match: <strong>"+escape(config_match)+"</strong>. Source: "+escape(primary_name)+"</p>"+html_table(grouped,list(grouped[0])))]
 sections.append(section("Existing candidate measurements", "<p>Values are descriptive. The memory and energy columns are explicitly unverified; a three-objective table is not a certified five-objective Pareto frontier.</p>"+html_table(candidates,fields)))
 for label in ("Step43 precision pairs","Step43 provisional frontier","Step44 training arms","Step44 matched order","Baseline resources","Energy models"):
  item=tables[label];records=item["data"];columns=item["columns"]
  sections.append(section(label,"<p>"+escape(item["path"])+" — "+escape(len(records))+" rows</p>"+(html_table(records,columns,1000) if records and columns else "<p>No rows found.</p>")))
 sections.append(section("Step44 individual result files",html_table(step44_raw,["experiment_dir","task","seed","best_validation_accuracy_pct","operation_order","result_path"]) if step44_raw else "<p>None found.</p>"))
 sections.append(section("Evidence gates (reported, not overridden)","<pre>"+escape(json.dumps(gates,indent=2))+"</pre>"))
 sections.append(section("Unresolved evidence","<ul>"+"".join("<li>"+escape(i)+"</li>" for i in issues)+"</ul>"))
 # Inventory table may be large; put it in separate CSV instead.
 sections.append(section("Files and provenance","<p>Full file inventory: <code>file_inventory.csv</code>. Candidate rows: <code>candidate_evidence.csv</code>. Summary: <code>exploration_summary.json</code>.</p>"))
 css="""body{font:14px system-ui,Segoe UI,Arial,sans-serif;margin:24px auto;padding:0 22px;max-width:1500px;color:#182536;background:#f7f9fc}h1{font-size:26px}h2{font-size:18px}section{background:white;border:1px solid #dce3eb;border-radius:12px;padding:18px;margin:18px 0} .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}.card{border:1px solid #e2e7ee;border-radius:8px;padding:14px}.card b{display:block;font-size:24px}.card span{color:#506072}.scroll{max-height:520px;overflow:auto;border:1px solid #e1e7ef}table{border-collapse:collapse;font-size:12px;width:100%}th,td{padding:8px;border-bottom:1px solid #edf0f5;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#eaf0f8;cursor:pointer}tr:nth-child(even){background:#fafbfd}pre{white-space:pre-wrap;word-break:break-word;background:#f2f5fa;padding:14px}li{margin:6px 0}input{padding:10px;width:min(500px,90%);border:1px solid #aab8c8;border-radius:8px}"""
 js="""document.querySelectorAll('table').forEach(t=>{let heads=t.querySelectorAll('th');heads.forEach((h,i)=>h.addEventListener('click',()=>{let b=t.tBodies[0],rs=[...b.rows],asc=h.dataset.asc!=='1';rs.sort((x,y)=>{let a=x.cells[i].textContent.trim(),c=y.cells[i].textContent.trim(),na=Number(a),nc=Number(c);return (a!==''&&c!==''&&!Number.isNaN(na)&&!Number.isNaN(nc)?na-nc:a.localeCompare(c,undefined,{numeric:true}))*(asc?1:-1)});rs.forEach(r=>b.appendChild(r));h.dataset.asc=asc?'1':'0'}) )});document.getElementById('filter').addEventListener('input',e=>{let q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))});"""
 page="<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CAJO Results Explorer</title><style>"+css+"</style><body><h1>CAJO — Existing Results Explorer</h1><p>Read-only evidence exploration. No new experiments, test access, or certification.</p><label>Filter displayed tables: <input id='filter' placeholder='dataset, seed, precision, candidate…'></label>"+"".join(sections)+"<script>"+js+"</script></body></html>"
 (out/"dashboard.html").write_text(page,encoding="utf-8")
 print(json.dumps({"dashboard":str(out/"dashboard.html"),"summary":str(out/"exploration_summary.json"),
                   "candidate_rows":len(candidates),"step44_training_arms":len(training),
                   "step44_matched_order_pairs":len(order),"config_matches":config_match,
                   "issues":len(issues),"read_errors":len(errors)},indent=2))
 return 0
if __name__=="__main__":sys.exit(main())
