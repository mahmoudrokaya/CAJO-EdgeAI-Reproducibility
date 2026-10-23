#!/usr/bin/env python3
"""CAJO Step49: read-only, targeted provenance/measurement verification.

No training, inference, checkpoint loading, or test access. Standard library only.
"""
import argparse, ast, csv, hashlib, json, re, statistics
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

TARGETS = {
 'R02': ('Native inference memory', ['memory','allocator','workspace','packed','activation','peak','rss','uss'], ['native_memory','memory_bytes','peak_memory','workspace','packed_weights']),
 'R04': ('CPU energy metrology', ['energy','hwinfo','power','idle','joule','watt','sample'], ['energy_mj','energy_j','power_w','idle_power','sample_count']),
 'R10': ('Independent NAS-only ablation', ['nas','architecture','search','screening','width','depth'], ['architecture','arch_id','search_budget','candidate_id']),
 'R11': ('Independent joint/sequential algorithms', ['joint','sequential','search','training','prun','quant','distill'], ['policy','search_mode','training_mode','joint','sequential']),
 'R12': ('Matched search budgets', ['budget','epoch','search','candidate','seed','wall_time'], ['epochs','seed','budget','candidate_count','search_time']),
 'R16': ('Matched baseline resource profiles', ['baseline','resource','cpu','latency','memory','energy'], ['latency_ms','memory_bytes','energy_mj','accuracy','task']),
}
SKIP_PARTS={'CAJO_Step46_Manuscript_Evidence_Audit','CAJO_Manuscript_Evidence_Audit','CAJO_Full_History_Audit','CAJO_Step48_Evidence_Decisions','CAJO_Step49_Targeted_Verification','CAJO_Results_Explorer'}

def load(path):
 try:
  if path.suffix.lower()=='.json':
   return json.loads(path.read_text(encoding='utf-8-sig'))
  if path.suffix.lower()=='.csv':
   for enc in ('utf-8-sig','cp1252','latin-1'):
    try:
     with path.open(encoding=enc,newline='') as f:
      return list(csv.DictReader(f))
    except UnicodeError: continue
 except Exception as e: return {'_read_error':str(e)}
 return None

def flatten(v,prefix='',depth=0):
 if depth>7:return []
 if isinstance(v,dict):
  a=[]
  for k,x in v.items():a+=flatten(x,(prefix+'.' if prefix else '')+str(k),depth+1)
  return a
 if isinstance(v,list):
  a=[]
  for x in v[:1000]:a+=flatten(x,prefix+'[]',depth+1)
  return a
 return [(prefix,v)]

def isnum(v):
 if isinstance(v,bool) or v is None:return False
 try:return str(v).strip() not in ('','None','nan','NOT_MEASURED','N/A') and float(v)==float(v)
 except (ValueError,TypeError):return False

def flags(fields):
 f=' '.join(fields).lower()
 return {k:any(x in f for x in terms) for k,terms in {
 'accuracy':['accuracy','acc_top','val_acc'], 'latency':['latency','inference_ms'],
 'storage':['storage','file_size','size_bytes'], 'memory':['memory','rss','uss','activation','workspace'],
 'energy':['energy','joule','power_w','package_power'], 'seed':['seed'],
 'architecture':['architecture','arch_id','width','depth'], 'search_budget':['budget','candidate_count','epochs'],
 'cpu':['cpu','onednn','num_threads'], 'split':['split','validation','test_set'],
 'export_hash':['export_sha256','export_hash','model_sha256']}.items()}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',type=Path,default=Path('..'))
 ap.add_argument('--out',type=Path)
 args=ap.parse_args();root=args.root.resolve();out=(args.out or root/'Reports'/'CAJO_Step49_Targeted_Verification').resolve()
 reports=root/'Reports';codes=root/'Codes'
 if not reports.is_dir() or not codes.is_dir():ap.error('Expected Codes/ and Reports/ beneath --root')
 out.mkdir(parents=True,exist_ok=True)
 rows=[]; numeric=[]; issues=[];code=[]
 for p in sorted(reports.rglob('*')):
  if not p.is_file() or p.suffix.lower() not in ('.csv','.json'):continue
  if any(x in SKIP_PARTS for x in p.relative_to(reports).parts):continue
  obj=load(p);rel=p.relative_to(root).as_posix()
  if isinstance(obj,dict) and '_read_error' in obj:issues.append({'path':rel,'issue':obj['_read_error']});continue
  if obj is None:continue
  records=obj if isinstance(obj,list) else [obj]
  if len(records)>20000:issues.append({'path':rel,'issue':'truncated inspection to 20000 records'})
  kv=[]
  for rec in records[:20000]:kv+=flatten(rec)
  keys=sorted({k for k,v in kv});nkeys=sorted({k for k,v in kv if isnum(v)})
  found=flags(keys)
  content=json.dumps(obj,ensure_ascii=False,default=str)[:300000].lower()
  for rid,(name,keywords,required) in TARGETS.items():
   score=sum(bool(re.search(r'(?<![a-z])'+re.escape(w),rel.lower()+' '+ ' '.join(keys).lower())) for w in keywords)
   if score<2:continue
   matching=[k for k in nkeys if any(t in k.lower() for t in required)]
   verified=any(k in content for k in ('"verified": true','"qualified": true','"status": "pass"'))
   rows.append({'requirement':rid,'source':rel,'records':len(records),'keyword_score':score,'numeric_matching_fields':'; '.join(matching[:24]),'has_numeric_measurements':bool(matching),'contains_explicit_positive_verification':verified,'field_sample':'; '.join(keys[:40])})
  if any(found.values()):
   numeric.append({'source':rel,'records':len(records),'numeric_fields':'; '.join(nkeys[:50]),**found})
 for p in sorted(codes.glob('*.py')):
  try:
   source=p.read_text(encoding='utf-8-sig');tree=ast.parse(source)
   calls=sorted({n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and (isinstance(n.func,ast.Name) or isinstance(n.func,ast.Attribute))})
   funs=sorted({n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))})
   text=(p.name+' '+ ' '.join(funs)+' '+ ' '.join(calls)).lower()
   for rid,(_,terms,_) in TARGETS.items():
    matches=[t for t in terms if t in text]
    if len(matches)>=2:code.append({'requirement':rid,'source':p.relative_to(root).as_posix(),'matching_terms':'; '.join(matches),'functions':'; '.join(funs[:30])})
  except Exception as e:issues.append({'path':p.relative_to(root).as_posix(),'issue':str(e)})
 # These are the decisive tests. Source hits alone NEVER pass them.
 checks={
 'R02':['A target-CPU run reports native peak inference allocation including packed weights, live activations and oneDNN workspaces','Measured values are traceable to each exact export and use the locked inference protocol'],
 'R04':['Power telemetry is timestamp-aligned to inference and idle controls','Per-inference net energy is nonnegative with documented uncertainty and adequate sensor resolution','Instrumentation wording matches CPU package power rather than direct V×I measurement'],
 'R10':['NAS-only and fixed-architecture arms share dataset splits, seeds, training and search budgets','Architecture choices and selected candidate scores are linked to original search logs'],
 'R11':['Joint and sequential policies are genuinely distinct in code, not merely recovery-operation order','All four components (architecture, pruning, quantization, KD) are included and budgets matched','Selection is validation-only with independent test held sealed'],
 'R12':['Both search policies have actual executed candidate counts and compute/epoch budgets','Logs demonstrate equal or normalized search resources, not only declared configuration'],
 'R16':['Baseline and CAJO candidates use same split, CPU, threads, batch size and measurement methodology','Baseline accuracy, latency, storage, memory and energy all use qualified measurements'],
 }
 gates={}
 gate=reports/'CAJO_Step38_Integrated_Closure_v1'/'evidence_gate.json'
 if gate.exists():
  obj=load(gate)
  if isinstance(obj,dict):
   gates=obj.get('checks',obj.get('gate_checks',obj))
 decisions=[]
 for rid,(name,_,_) in TARGETS.items():
  matches=[r for r in rows if r['requirement']==rid]
  numeric_matches=[r for r in matches if r['has_numeric_measurements']]
  code_matches=[r for r in code if r['requirement']==rid]
  decision='INSPECT EXISTING RECORDS' if matches else 'NO MATCHING EVIDENCE LOCATED — VERIFY BEFORE RERUN'
  if rid in ('R02','R04','R11','R16'):
   decision='UNVERIFIED — DO NOT CLAIM COMPLETE; INSPECT RAW EVIDENCE'
  if rid in ('R10','R12') and not numeric_matches:
   decision='NOT ESTABLISHED — CHECK ORIGINAL SEARCH LOGS'
  decisions.append({'requirement':rid,'claim':name,'decision':decision,'matched_artifacts':len(matches),'numeric_artifacts':len(numeric_matches),'related_code_files':len(code_matches),'top_sources':'; '.join(r['source'] for r in sorted(numeric_matches,key=lambda x:-x['keyword_score'])[:8]),'verification_checks':' | '.join(checks[rid]),'new_experiment_now':'NO — conditional on provenance verification'})
 def write_csv(name,data):
  if not data:return
  with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 for name,data in [('targeted_requirement_decisions.csv',decisions),('relevant_source_records.csv',rows),('numerical_field_inventory.csv',numeric),('related_source_code.csv',code),('parse_issues.csv',issues)]:write_csv(name,data)
 result={'schema':'cajo_step49_targeted_verification_v1','created_utc':datetime.now(timezone.utc).isoformat(),'root':str(root),'reports_parsed':len(set(x['source'] for x in numeric)),'relevant_source_matches':len(rows),'decisions':decisions,'step38_gate':gates,'parse_issues':issues,'limitations':['Source relevance and numeric-field presence do not validate metrology or causal ablations.','Code is parsed statically, never executed.','No checkpoint, training, inference, or test data are accessed.','New experiments are not authorized by this audit.']}
 (out/'targeted_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
 lines=['# CAJO Step 49 — Targeted verification of unresolved evidence','',f'Project: `{root}`','', 'This is an evidence-triage report, not a scientific certification. No training or sealed testing was performed.','', '| Requirement | Decision | Relevant artifacts | Numerical artifacts |','|---|---|---:|---:|']
 for d in decisions:lines.append(f"| {d['requirement']} — {d['claim']} | {d['decision']} | {d['matched_artifacts']} | {d['numeric_artifacts']} |")
 for d in decisions:
  lines+=['',f"## {d['requirement']}: {d['claim']}",f"Decision: **{d['decision']}**",'', 'Checks required:']
  lines+=['- '+x for x in checks[d['requirement']]]
  lines+=['','Best numerical source leads:',d['top_sources'] or 'No direct numerical source identified by this audit.']
 lines+=['','## Limits','Matching fields and scripts are leads only. An absent match does not establish that an experiment was never conducted. Do not open the sealed test.']
 (out/'targeted_verification_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('Saved:',out);print('Requirement decisions:',len(decisions),'candidate source links:',len(rows),'parse issues:',len(issues))
if __name__=='__main__':main()
