#!/usr/bin/env python3
"""CAJO Steps 36–45: auditable, fail-closed results consolidation.

Modes:
  inspect   inventory/evidence inspection, no test access
  analyze   consolidate evidence, energy QC, comparison audit, conditional Pareto,
            descriptive statistics, figures, readiness manifest
  freeze    ONLY after analyze and verified five-objective gates; explicit opt-in
  test      ONLY after freeze, explicit opt-in, external verified evaluator
  finalize  audit test results and publish a writing-readiness report

This script cannot invent native oneDNN allocation traces, missing independent
experiments, or a sealed-test evaluator. No model training is performed.
Requires Python stdlib; matplotlib optional for PNG figures.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,statistics,subprocess,sys,time
from collections import defaultdict
from pathlib import Path
from datetime import datetime,timezone

LOCK='56361ba3ef2cb530d54a923cf4f5b3758a22bee82a9119822395cef0be9bbdc5'
CONFIG='Reports/Optimization_Protocol/Final_Experiment_Config/cajo_experiment_config.json'
STEP35='Reports/CAJO_Step35_Closure_v1'
STEP34='Reports/CAJO_Step34_Audit'
OUTPUT='Reports/CAJO_Steps36_45_ResultsClosure_v1'
CAPS={'CIFAR10':(85,128*2**20,30,8*2**20), 'CIFAR100':(70,512*2**20,100,64*2**20), 'SpeechCommands_v0.02':(90,64*2**20,20,2*2**20)}
FIELDS=('error_pct','storage_bytes','latency_mean_ms','inference_memory_bytes','energy_incremental_mj')

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def table(p,rows,columns):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
def finite(v):
 try:
  n=float(v)
  return n if math.isfinite(n) else None
 except (ValueError,TypeError):return None
def now():return datetime.now(timezone.utc).isoformat()
def key(r):return (r['candidate_id'],r['precision'].upper())
def safe(s):return str(s).replace('/','_').replace('\\','_')
def check_lock(root):
 p=root/CONFIG
 if not p.is_file() or digest(p)!=LOCK:raise RuntimeError('Locked config missing/changed: STOP')
def ci95(x):
 if len(x)<2:return None
 # Student t quantiles for 95% two-sided CI at n=2..30, conservative normal beyond.
 t={2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262,11:2.228,12:2.201,13:2.179,14:2.160,15:2.145,16:2.131,17:2.120,18:2.110,19:2.101,20:2.093,21:2.086,22:2.080,23:2.074,24:2.069,25:2.064,26:2.060,27:2.056,28:2.052,29:2.048,30:2.045}.get(len(x),1.96)
 m=statistics.mean(x);d=t*statistics.stdev(x)/math.sqrt(len(x));return [m-d,m+d]
def summary(x):
 x=[v for v in (finite(a) for a in x) if v is not None]
 return {'n':len(x),'mean':statistics.mean(x) if x else None,'sd':statistics.stdev(x) if len(x)>1 else None,'ci95':ci95(x),'min':min(x) if x else None,'max':max(x) if x else None}
def load_evidence(root):
 check_lock(root);base=root/STEP35
 required=['combined_objectives.json','closure_summary.json','method_comparison_audit.json','selection_gate.json']
 for n in required:
  if not (base/n).is_file():raise RuntimeError('Missing Step35 evidence: '+str(base/n))
 records=load(base/'combined_objectives.json');gate=load(base/'selection_gate.json');closure=load(base/'closure_summary.json')
 if not isinstance(records,list) or len(records)!=36:raise RuntimeError('Expected 36 Step35 objective records')
 if gate.get('locked_config_sha256')!=LOCK:raise RuntimeError('Step35 lock mismatch')
 if len({key(r) for r in records})!=36:raise RuntimeError('Duplicate candidate/precision')
 for r in records:
  p=root/Path(r['export_path'].replace('\\','/'))
  if not p.is_file() or digest(p)!=r['export_sha256']:raise RuntimeError('Export SHA mismatch: '+str(p))
  if r['task'] not in CAPS:raise RuntimeError('Unknown task')
 return records,gate,closure,load(base/'method_comparison_audit.json')

def native_attestation(root,path,records):
 """External independent evidence, NOT auto-generated or silently trusted.
 Schema: {"schema":"cajo_native_memory_attestation_v1", "method":"...",
 "definition":"...", "complete_native_workspace_coverage":true,
 "auditor":"...", "records":[{"candidate_id":"...","precision":"...",
 "export_sha256":"...","inference_memory_bytes":123}]}
 Evidence MUST be reviewed independently before --accept-native-attestation.
 """
 if not path:return {},['Native memory attestation not supplied']
 a=load(path);errors=[];idx={key(r):r for r in records};seen={}
 if a.get('schema')!='cajo_native_memory_attestation_v1':errors.append('Wrong attestation schema')
 for f in ('method','definition','auditor'):
  if not a.get(f):errors.append('Missing attestation '+f)
 if a.get('complete_native_workspace_coverage') is not True:errors.append('Native workspaces not demonstrated')
 for v in a.get('records',[]):
  k=key(v);ref=idx.get(k);n=finite(v.get('inference_memory_bytes'))
  if k in seen or ref is None or n is None or n<=0 or ref['export_sha256']!=v.get('export_sha256'):errors.append('Invalid memory attestation row: '+str(k))
  else:seen[k]=n
 if len(seen)!=36:errors.append('Native attestation incomplete: '+str(len(seen))+'/36')
 return (seen if not errors else {}),errors

def energy_detail(root,records):
 p=root/STEP34/'energy_models.csv';detail={};errs=[]
 if not p.exists():return detail,['Step34 energy_models.csv missing']
 with p.open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   k=key(row)
   if k in detail:errs.append('Duplicate energy key: '+str(k))
   detail[k]=row
 return detail,errs

def baseline_sources(root):
 paths=[root/'Reports/Baseline_Resource_Profiling/baseline_resource_comparison.csv',root/'Reports/CAJO_Final_Pipeline/final_cpu_per_seed.csv']
 for p in paths:
  if p.is_file():
   with p.open(encoding='utf-8-sig',newline='') as f:
    rr=list(csv.DictReader(f));yield {'path':str(p.relative_to(root)),'sha256':digest(p),'rows':len(rr),'columns':list(rr[0]) if rr else []}

def pareto(rows):
 out=[]
 for a in rows:
  va=[a[f] for f in FIELDS]
  if any(v is None for v in va):continue
  dominated=False
  for b in rows:
   if a is b:continue
   vb=[b[f] for f in FIELDS]
   if all(v is not None for v in vb) and all(x<=y for x,y in zip(vb,va)) and any(x<y for x,y in zip(vb,va)):
    dominated=True;break
  if not dominated:out.append(a)
 return out

def plot(rows,out):
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
 except ImportError:return 'matplotlib unavailable; CSV tables generated'
 for task in CAPS:
  rr=[r for r in rows if r['task']==task and r['latency_mean_ms'] is not None]
  fig,ax=plt.subplots(figsize=(7,4.5))
  for prec in ('FP32','INT8'):
   pts=[r for r in rr if r['precision']==prec]
   ax.scatter([r['latency_mean_ms'] for r in pts],[r['error_pct'] for r in pts],label=prec)
  ax.set(xlabel='CPU latency (ms)',ylabel='Validation error (%)',title=task+' — observed trade-off (not certified Pareto)')
  ax.legend();fig.tight_layout();fig.savefig(out/('tradeoff_'+safe(task)+'.png'),dpi=180);plt.close(fig)
 return 'generated 3 descriptive PNG figures'

def analyze(root,out,args):
 records,oldgate,closure,method=load_evidence(root)
 energy,energy_err=energy_detail(root,records)
 native,native_err=native_attestation(root,args.native_attestation,records)
 if args.native_attestation and not args.accept_native_attestation:native_err.append('Attestation requires explicit --accept-native-attestation after independent review');native={}
 rows=[];failures=[];prov={}
 for r in records:
  k=key(r);m=r['metrics'];e=energy.get(k,{});acc=finite(r.get('validation_accuracy_pct'))
  v={'candidate_id':r['candidate_id'],'task':r['task'],'precision':r['precision'],'export_sha256':r['export_sha256'],
     'validation_accuracy_pct':acc,'error_pct':finite(m.get('error_pct')),'storage_bytes':finite(m.get('storage_bytes')),
     'latency_mean_ms':finite(m.get('latency_mean_ms')),'inference_memory_bytes':native.get(k),
     'process_memory_proxy_bytes':finite(m.get('process_memory_proxy_bytes')),
     'energy_incremental_mj':finite(m.get('energy_incremental_mj')) if r['checks'].get('energy_quality') else None,
     'energy_sd_mj':finite(e.get('energy_incremental_sd_mj')),'energy_trials':int(e.get('trials') or 0),
     'energy_quality':bool(r['checks'].get('energy_quality')),'energy_cv':None}
  if v['energy_incremental_mj'] is not None and v['energy_sd_mj'] is not None and v['energy_incremental_mj']>0:
   v['energy_cv']=v['energy_sd_mj']/v['energy_incremental_mj']
  ac,mem,lat,stor=CAPS[r['task']]
  v['passes_accuracy']=acc is not None and acc>=ac
  v['passes_storage']=v['storage_bytes'] is not None and v['storage_bytes']<=stor
  v['passes_latency']=v['latency_mean_ms'] is not None and v['latency_mean_ms']<=lat
  v['passes_memory']=v['inference_memory_bytes'] is not None and v['inference_memory_bytes']<=mem
  v['five_objective_complete']=all(v[x] is not None for x in FIELDS)
  v['five_objective_feasible']=v['five_objective_complete'] and all(v['passes_'+q] for q in ('accuracy','storage','latency','memory'))
  if not v['energy_quality']:
   failures.append({'candidate_id':r['candidate_id'],'precision':r['precision'],'reason':'Failed original energy-quality gate; retain raw trials, do not impute'})
  rows.append(v)
 # Separate energy precision groups and seed group; avoid cross-task comparison.
 grouped=defaultdict(list)
 for r in rows:
  policy='adaptive_sequential_search' if 'closure26' in r['candidate_id'] else 'reference_search_policy'
  grouped[(r['task'],policy,r['precision'])].append(r)
 stats=[]
 for (task,policy,precision),rs in sorted(grouped.items()):
  item={'task':task,'policy':policy,'precision':precision,'n_models':len(rs)}
  for f in FIELDS:item[f]=summary([r[f] for r in rs])
  stats.append(item)
 # Descriptive matched differences: same task, seed, precision; explicitly noncausal.
 pair=[]
 for task in CAPS:
  for prec in ('FP32','INT8'):
   for seed in (42,43,44):
    rr=[r for r in rows if r['task']==task and r['precision']==prec and ('__s'+str(seed)) in r['candidate_id']]
    a=next((r for r in rr if 'closure26' in r['candidate_id']),None)
    b=next((r for r in rr if 'final24' in r['candidate_id']),None)
    if not a or not b:continue
    d={'task':task,'precision':prec,'seed':seed,'policy_a':a['candidate_id'],'policy_b':b['candidate_id']}
    for f in FIELDS:d['delta_'+f]=a[f]-b[f] if a[f] is not None and b[f] is not None else None
    pair.append(d)
 pairsum=[]
 for task in CAPS:
  for prec in ('FP32','INT8'):
   pp=[r for r in pair if r['task']==task and r['precision']==prec]
   q={'task':task,'precision':prec,'matched_pairs':len(pp),'causal_training_comparison':False}
   for f in FIELDS:q['delta_'+f]=summary([r['delta_'+f] for r in pp])
   pairsum.append(q)
 frontier={};feasible=[r for r in rows if r['five_objective_feasible']]
 for task in CAPS:frontier[task]=[{'candidate_id':r['candidate_id'],'precision':r['precision'],'export_sha256':r['export_sha256']} for r in pareto([x for x in feasible if x['task']==task])]
 allfeasible=all(any(r['task']==t for r in feasible) for t in CAPS)
 native_ok=len(native)==36 and not native_err
 energy_ok=all(r['energy_quality'] for r in rows)
 # Separate fully measured subsets from task-level feasibility.
 evidence={'locked_config_sha256':LOCK,'step35_summary':closure,'energy_failures':failures,'native_memory_attestation_errors':native_err,
           'energy_source_errors':energy_err,'baseline_source_inventory':list(baseline_sources(root)),
           'method_comparison':method,'method_training_algorithm_comparison_verified':False,
           'matched_policy_pairs':len(pair),'energy_quality_pass':sum(r['energy_quality'] for r in rows),
           'native_memory_verified':native_ok,'full_five_objective_vectors':sum(r['five_objective_complete'] for r in rows),
           'five_objective_feasible':len(feasible),'all_tasks_have_feasible_candidate':allfeasible,
           'all_36_energy_quality_pass':energy_ok,
           'five_objective_frontier_conditional':frontier,
           'certified_original_protocol':native_ok and energy_ok and allfeasible,
           'statistical_caveat':'Three seed-level observations per group; descriptive paired differences only. Matching seeds does not remove search-budget/implementation confounding.',
           'energy_caveat':'HWiNFO package-power integration is sensor-derived and not independently calibrated. Baseline subtraction can yield nonpositive values.',
           'memory_caveat':'Process memory includes runtime; exact native memory requires independent allocation coverage attestation.'}
 table(out/'objective_table.csv',rows,list(rows[0]));table(out/'energy_failures.csv',failures,['candidate_id','precision','reason'])
 table(out/'paired_search_policy_differences.csv',pair,list(pair[0]) if pair else ['task','precision','seed'])
 dump(out/'descriptive_statistics.json',stats);dump(out/'paired_policy_statistics.json',pairsum)
 dump(out/'five_objective_frontiers.json',{'certified_original_protocol':evidence['certified_original_protocol'],'frontiers':frontier,'feasible_count':len(feasible)})
 dump(out/'evidence_audit.json',evidence)
 fig=plot(rows,out)
 # Deliberately do NOT freeze if even one original objective definition unverified.
 gate={'schema':'cajo_36_45_gate_v1','created_utc':now(),'locked_config_sha256':LOCK,
       'all_original_objectives_verified':native_ok and energy_ok,'all_tasks_feasible':allfeasible,
       'method_algorithm_comparison_verified':False,
       'baseline_and_ablation_independent_verification':False,
       'predeclared_selection_tiebreak':'minimum validation error; then minimum energy; then latency; then storage; then memory; then export SHA256 lexicographic',
       'selection_ready':native_ok and energy_ok and allfeasible,
       'writing_ready':False,'sealed_test_access':'SEALED',
       'reason':'Method and baseline/ablation provenance require independent audit; memory/energy/feasibility and sealed test may be incomplete',
       'figures':fig}
 dump(out/'readiness_gate.json',gate)
 manifest={'generated_utc':now(),'config_sha256':LOCK,'inputs':{}}
 for p in [root/CONFIG,root/STEP35/'combined_objectives.json',root/STEP35/'selection_gate.json',root/STEP34/'energy_models.csv']:
  if p.is_file():manifest['inputs'][str(p.relative_to(root))]=digest(p)
 if args.native_attestation:manifest['inputs'][str(Path(args.native_attestation).resolve())]=digest(args.native_attestation)
 dump(out/'input_manifest.json',manifest)
 return gate,evidence

def freeze(root,out,args):
 gate=load(out/'readiness_gate.json');ev=load(out/'evidence_audit.json')
 if not args.authorize_freeze:raise RuntimeError('Freeze requires --authorize-freeze')
 if not gate['selection_ready']:raise RuntimeError('Selection gate FAIL: no freeze, no sealed test')
 if (out/'frozen_selection.json').exists():raise RuntimeError('Already frozen; immutable, no overwrite')
 rr=list(csv.DictReader((out/'objective_table.csv').open(encoding='utf-8-sig',newline='')))
 selections=[]
 for task in CAPS:
  eligible=[r for r in rr if r['task']==task and r['five_objective_feasible']=='True']
  if not eligible:raise RuntimeError('No feasible '+task)
  eligible.sort(key=lambda r:(float(r['error_pct']),float(r['energy_incremental_mj']),float(r['latency_mean_ms']),float(r['storage_bytes']),float(r['inference_memory_bytes']),r['export_sha256']))
  r=eligible[0];selections.append({'task':task,'candidate_id':r['candidate_id'],'precision':r['precision'],'export_sha256':r['export_sha256']})
 frozen={'schema':'cajo_frozen_selection_v1','created_utc':now(),'config_sha256':LOCK,'selection_rule':gate['predeclared_selection_tiebreak'],'selected':selections,'test_access':'SEALED','immutable':True}
 dump(out/'frozen_selection.json',frozen)
 print('Frozen selections; inspect manifest. Test is STILL SEALED.');return frozen

def run_test(root,out,args):
 if not args.authorize_sealed_test:raise RuntimeError('Test requires explicit --authorize-sealed-test')
 fp=out/'frozen_selection.json'
 if not fp.exists():raise RuntimeError('No frozen selection; sealed test remains closed')
 if (out/'test_results.json').exists() or (out/'test_attempt.json').exists():raise RuntimeError('Test already attempted; cannot rerun automatically')
 if not args.test_evaluator:raise RuntimeError('Provide a separately audited evaluator via --test-evaluator')
 evaluator=Path(args.test_evaluator).resolve()
 if not evaluator.is_file() or evaluator.suffix.lower()!='.py':raise RuntimeError('Evaluator must be an existing .py file')
 if not args.evaluator_sha256 or digest(evaluator)!=args.evaluator_sha256:raise RuntimeError('Evaluator SHA256 missing/mismatch')
 # The test attempt marker is written BEFORE execution, so a crash cannot lead
 # to an accidental second evaluation. No shell; no user-controlled arguments.
 attempt={'utc':now(),'evaluator':str(evaluator),'evaluator_sha256':args.evaluator_sha256,'frozen_selection_sha256':digest(fp),'status':'STARTED'}
 dump(out/'test_attempt.json',attempt)
 dest=out/'test_results.json'
 cmd=[sys.executable,str(evaluator),'--manifest',str(fp.resolve()),'--output',str(dest.resolve())]
 with (out/'test_evaluator_stdout.txt').open('w',encoding='utf-8') as so,(out/'test_evaluator_stderr.txt').open('w',encoding='utf-8') as se:
  proc=subprocess.run(cmd,cwd=str(root),stdout=so,stderr=se,timeout=args.test_timeout_seconds,check=False)
 attempt['returncode']=proc.returncode;attempt['status']='COMPLETED' if proc.returncode==0 and dest.exists() else 'FAILED'
 dump(out/'test_attempt.json',attempt)
 if attempt['status']!='COMPLETED':raise RuntimeError('Test failed; no automatic retry. Review test logs.')
 print('Test evaluator finished. Validate dataset provenance and test output before claims.')

def finalize(root,out,args):
 gate=load(out/'readiness_gate.json');ev=load(out/'evidence_audit.json')
 fp=out/'frozen_selection.json';tp=out/'test_results.json'
 report={'created_utc':now(),'original_five_objective_verified':ev['certified_original_protocol'],
         'selection_frozen':fp.is_file(),'sealed_test_completed':tp.is_file(),
         'method_training_algorithm_comparison_verified':False,
         'baseline_ablation_verified':False,
         'all_results_and_figures_verified':False,'writing_ready':False,
         'unresolved':[]}
 if not ev['native_memory_verified']:report['unresolved'].append('Native memory incomplete; original inference-memory constraint not certified')
 if not ev['all_36_energy_quality_pass']:report['unresolved'].append('Four energy-quality failures require documented disposition')
 if not ev['all_tasks_have_feasible_candidate']:report['unresolved'].append('At least one task has no verified feasible candidate')
 report['unresolved'].append('Independent matched baseline/ablation and search-policy provenance audit not established by these files')
 if not fp.is_file():report['unresolved'].append('Selection not frozen')
 if not tp.is_file():report['unresolved'].append('Sealed test not completed')
 else:
  try:
   result=load(tp);selected=load(fp)['selected']
   if result.get('schema')!='cajo_sealed_test_results_v1' or len(result.get('results',[]))!=len(selected):
    report['unresolved'].append('Sealed test output schema or count invalid')
   else:
    for a,b in zip(selected,result['results']):
     if any(a[k]!=b.get(k) for k in ('task','candidate_id','precision','export_sha256')) or finite(b.get('test_accuracy_pct')) is None:
      report['unresolved'].append('Test record identity/metric mismatch');break
  except Exception as ex:report['unresolved'].append('Test output could not be validated: '+str(ex))
 dump(out/'writing_readiness_report.json',report)
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--root',type=Path,default=Path('..'));p.add_argument('--mode',choices=('inspect','analyze','all','freeze','test','finalize'),default='all')
 p.add_argument('--native-attestation',type=Path);p.add_argument('--accept-native-attestation',action='store_true')
 p.add_argument('--authorize-freeze',action='store_true');p.add_argument('--authorize-sealed-test',action='store_true')
 p.add_argument('--test-evaluator',type=Path);p.add_argument('--evaluator-sha256');p.add_argument('--test-timeout-seconds',type=int,default=7200)
 args=p.parse_args();root=args.root.resolve();out=root/OUTPUT
 if args.mode=='inspect':
  rr,g,c,m=load_evidence(root);print(json.dumps({'models':len(rr),'step35':c,'step34_energy_models_present':(root/STEP34/'energy_models.csv').exists(),'native_memory_verified':False,'sealed_test':g.get('test_access')},indent=2));return
 if args.mode in ('analyze','all'):
  gate,evidence=analyze(root,out,args)
  print(json.dumps({'mode':args.mode,'exports':36,'energy_quality_pass':evidence['energy_quality_pass'],'native_memory_verified':evidence['native_memory_verified'],'full_vectors':evidence['full_five_objective_vectors'],'five_objective_feasible':evidence['five_objective_feasible'],'selection_ready':gate['selection_ready'],'sealed_test_access':'SEALED','output':str(out)},indent=2))
  if args.mode=='all':print(json.dumps(finalize(root,out,args),indent=2))
 elif args.mode=='freeze':print(json.dumps(freeze(root,out,args),indent=2))
 elif args.mode=='test':run_test(root,out,args)
 elif args.mode=='finalize':print(json.dumps(finalize(root,out,args),indent=2))
if __name__=='__main__':
 try:main()
 except Exception as ex:print('FAIL CLOSED:',str(ex),file=sys.stderr);sys.exit(2)
