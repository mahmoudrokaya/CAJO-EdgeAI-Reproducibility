#!/usr/bin/env python3
"""CAJO Step 43: execute real validation and deployment experiments.

From Codes directory:
  python 43_cajo_experimental_execution.py run --root .. --limit 2
  python 43_cajo_experimental_execution.py run --root ..
  python 43_cajo_experimental_execution.py baselines --root ..
  python 43_cajo_experimental_execution.py compare --root ..
  python 43_cajo_experimental_execution.py pareto --root ..
  python 43_cajo_experimental_execution.py close --root ..

This program performs actual inference, validation, CPU latency measurements,
CPU allocation profiling (exploratory, NOT exact native memory), paired existing
search-policy comparisons, and real frozen-baseline validation/latency. It
produces provisional 3D Pareto sets, never falsely certifies five objectives.
Step38 owns final 5D feasibility, freeze and sealed-test access. No test split
is loaded by this program. All 36 exports are hash-verified before inference.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,os,statistics,subprocess,sys,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

LOCK='56361ba3ef2cb530d54a923cf4f5b3758a22bee82a9119822395cef0be9bbdc5'
CFG='Reports/Optimization_Protocol/Final_Experiment_Config/cajo_experiment_config.json'
INV='Reports/CAJO_Independent_Closure_v1/finalist_unique_seed_precision.csv'
OUT='Reports/CAJO_Step43_Experimental_Execution'
SHAPES={'CIFAR10':(1,3,32,32),'CIFAR100':(1,3,32,32),'SpeechCommands_v0.02':(1,1,40,101)}
LIMITS={'CIFAR10':(85,30,8*2**20),'CIFAR100':(70,100,64*2**20),'SpeechCommands_v0.02':(90,20,2*2**20)}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
 with open(p,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def atomic_json(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.partial')
 with tmp.open('w',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def atomic_csv(p,items):
 if not items:return
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
 fields=list(dict.fromkeys(k for r in items for k in r))
 with tmp.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(items)
 os.replace(tmp,p)
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def context(root):
 if sha(root/CFG)!=LOCK:raise RuntimeError('LOCKED CONFIG HASH MISMATCH')
 inv=rows(root/INV)
 if len(inv)!=36:raise RuntimeError('Expected exactly 36 immutable exports')
 seen=set()
 for r in inv:
  k=(r['candidate_id'],r['precision'])
  if k in seen:raise RuntimeError('Duplicate export '+str(k))
  seen.add(k)
  p=root/r['export_path']
  if not p.is_file() or sha(p)!=r['export_sha256']:raise RuntimeError('Export drift '+str(p))
  if r['task'] not in SHAPES:raise RuntimeError('Unexpected task')
 return inv

def validation_loader(root,task,batch=128):
 """Exactly the original Phase-3 validation transforms; never load test rows."""
 import torch
 m=module(root/'Codes'/'03_train_baseline_models.py','cajo43_phase3')
 report=root/'Reports'/'Dataset_Verification'/'dataset_verification_report.json'
 verify=readj(report)
 manifest=root/'Reports'/'Dataset_Verification'/m.DATASETS[task][0]
 if sha(manifest)!=verify['datasets'][task]['manifest_sha256']:raise RuntimeError('Split manifest drift')
 allrows=m.read_manifest(manifest)
 selected=[r for r in allrows if r['split']=='validation']
 if not selected:raise RuntimeError('No validation examples')
 if len(selected)!=int(verify['datasets'][task]['split_counts']['validation']):raise RuntimeError('Validation split count drift')
 data=root/'Data'
 if task.startswith('CIFAR'):ds=m.CIFARManifestDataset(selected,data,task,False,42)
 else:
  # Match Phase-3 baseline training exactly: class_names comes from the
  # verified Phase-1 dataset report, not from an invented class_id column.
  labels=verify['datasets'][task]['class_names']
  if not isinstance(labels,list) or len(labels)!=35 or len(set(labels))!=35:
   raise RuntimeError('Invalid verified SpeechCommands 35-class vocabulary')
  observed={r['original_word_label'] for r in allrows}
  if observed!=set(labels):
   raise RuntimeError('SpeechCommands manifest vocabulary differs from verified class_names')
  ds=m.SpeechManifestDataset(selected,data,labels)
 return torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0),len(selected),sha(manifest)

def infer(model,loader):
 import torch
 correct=total=0
 model.eval()
 with torch.inference_mode():
  for x,y in loader:
   out=model(x.cpu());pred=out.argmax(dim=1)
   correct+=int((pred==y).sum().item());total+=int(y.numel())
 return correct,total

def measure(model,shape,warmups=50,runs=100):
 import torch
 torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 x=torch.randn(*shape)
 with torch.inference_mode():
  for _ in range(warmups):model(x)
  ts=[]
  for _ in range(runs):
   t=time.perf_counter_ns();model(x);ts.append((time.perf_counter_ns()-t)/1e6)
 return {'latency_mean_ms':statistics.mean(ts),'latency_median_ms':statistics.median(ts),
         'latency_sd_ms':statistics.stdev(ts) if len(ts)>1 else 0,'warmups':warmups,'runs':runs}

def profile_allocations(model,shape):
 """Real CPU torch profiler allocation events, NOT a complete native peak."""
 import torch
 x=torch.randn(*shape)
 try:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True) as prof:
   with torch.inference_mode():model(x)
  ev=prof.key_averages();positive=sum(max(0,int(e.self_cpu_memory_usage)) for e in ev)
  return {'torch_profiler_positive_self_allocated_bytes':positive,
          'scope':'PYTORCH_ALLOCATOR_DIAGNOSTIC_NOT_EXACT_NATIVE_MEMORY',
          'native_memory_certified':False}
 except Exception as e:return {'profiler_error':repr(e),'native_memory_certified':False}

def run_exports(root,inv,a):
 import torch
 torch.set_num_threads(1)
 out=root/OUT/'exports';out.mkdir(parents=True,exist_ok=True)
 count=0
 for r in inv:
  k=hashlib.sha256((r['candidate_id']+'|'+r['precision']).encode()).hexdigest()[:24]
  target=out/(k+'.json')
  if target.exists():
   d=readj(target)
   if d['export_sha256']!=r['export_sha256']:raise RuntimeError('Existing experiment drift '+str(target))
   print('SKIP',r['candidate_id'],r['precision'],flush=True);continue
  if a.limit and count>=a.limit:break
  print('RUN',r['candidate_id'],r['precision'],flush=True)
  model=torch.jit.load(str(root/r['export_path']),map_location='cpu').eval()
  loader,n,manifest_hash=validation_loader(root,r['task'],a.batch_size)
  correct,total=infer(model,loader)
  if total!=n:raise RuntimeError('Validation sample mismatch')
  latency=measure(model,SHAPES[r['task']],a.warmups,a.runs)
  memory=profile_allocations(model,SHAPES[r['task']])
  obj={'schema':'cajo43_measured_export_v1','task':r['task'],'candidate_id':r['candidate_id'],
       'precision':r['precision'],'export_sha256':r['export_sha256'],
       'validation_manifest_sha256':manifest_hash,'validation_correct':correct,
       'validation_total':total,'validation_accuracy_pct':100*correct/total,
       'storage_bytes':(root/r['export_path']).stat().st_size,
       'test_accessed':False,'config_sha256':LOCK,'created_utc':datetime.now(timezone.utc).isoformat(),
       **latency,**memory}
  atomic_json(target,obj);count+=1
 print('New exports evaluated:',count)

def baselines(root,a):
 import torch
 torch.set_num_threads(1)
 base=module(root/'Codes'/'03_train_baseline_models.py','cajo43_baseline')
 arch=module(root/'Codes'/'02_prepare_baseline_models.py','cajo43_arch')
 dest=root/OUT/'baselines';dest.mkdir(parents=True,exist_ok=True)
 for task in SHAPES:
  target=dest/(task+'.json');p=root/'Reports'/'Baseline_Training'/task/'best_model.pt'
  if not p.is_file():raise FileNotFoundError(p)
  digest=sha(p)
  if target.exists():
   if readj(target)['checkpoint_sha256']!=digest:raise RuntimeError('Baseline drift')
   print('SKIP baseline',task);continue
  model=base.build_model(arch,task).cpu().eval()
  blob=torch.load(p,map_location='cpu',weights_only=False) # local trusted checkpoint only
  model.load_state_dict(blob['model'],strict=True)
  loader,n,manifest_hash=validation_loader(root,task,a.batch_size)
  correct,total=infer(model,loader)
  if total!=n:raise RuntimeError('Baseline validation count mismatch')
  latency=measure(model,SHAPES[task],a.warmups,a.runs)
  memory=profile_allocations(model,SHAPES[task])
  atomic_json(target,{'schema':'cajo43_real_baseline_v1','task':task,'checkpoint_sha256':digest,
    'validation_manifest_sha256':manifest_hash,'validation_accuracy_pct':100*correct/total,
    'validation_total':total,'validation_correct':correct,'checkpoint_bytes':p.stat().st_size,
    'test_accessed':False,'baseline':'uncompressed_FP32','config_sha256':LOCK,**latency,**memory})
  print('DONE baseline',task,round(100*correct/total,3))

def aggregate(root,inv):
 records=[]
 for r in inv:
  key=hashlib.sha256((r['candidate_id']+'|'+r['precision']).encode()).hexdigest()[:24]
  p=root/OUT/'exports'/(key+'.json')
  if not p.exists():continue
  d=readj(p)
  if d['export_sha256']!=r['export_sha256']:raise RuntimeError('Export result mismatch')
  records.append(d)
 atomic_csv(root/OUT/'measured_validation_latency_storage.csv',records)
 return records

def compare(root,inv):
 """Real paired precision comparisons and existing search-policy contrasts; no invented training algorithms."""
 data=aggregate(root,inv);by=defaultdict(dict)
 for r in data:by[(r['task'],r['candidate_id'])][r['precision'].upper()]=r
 pairs=[]
 for (task,cid),group in by.items():
  if 'FP32' not in group or 'INT8' not in group:continue
  f,i=group['FP32'],group['INT8']
  pairs.append({'task':task,'candidate_id':cid,'comparison':'INT8_minus_FP32_same_checkpoint',
   'accuracy_delta_pp':i['validation_accuracy_pct']-f['validation_accuracy_pct'],
   'latency_delta_ms':i['latency_mean_ms']-f['latency_mean_ms'],
   'storage_delta_bytes':i['storage_bytes']-f['storage_bytes'],
   'causal_scope':'post_training_precision_only','distinct_training_algorithm':False})
 atomic_csv(root/OUT/'paired_precision_ablation.csv',pairs)
 # Historical joint vs sequential rows: observed policy comparison, never algorithm certification.
 candidates=[root/'Reports'/'CAJO_Steps36_45_ResultsClosure_v1'/'paired_comparisons.csv',
             root/'Reports'/'CAJO_Independent_Closure_v1'/'paired_comparisons.csv']
 for p in candidates:
  if p.is_file():
   source=rows(p);atomic_json(root/OUT/'existing_policy_comparison.json',
     {'source_path':str(p.relative_to(root)),'source_sha256':sha(p),'rows':source,
      'scope':'EXISTING_SEARCH_POLICY_COMPARISON_NOT_DISTINCT_TRAINING_ALGORITHMS'});break
 print(json.dumps({'new_matched_FP32_INT8_pairs':len(pairs),'distinct_joint_sequential_algorithms_executed':False},indent=2))

def pareto(root,inv):
 data=aggregate(root,inv);front=[]
 for t in SHAPES:
  subset=[r for r in data if r['task']==t]
  acc,lat,storage=LIMITS[t]
  feasible=[r for r in subset if r['validation_accuracy_pct']>=acc and r['latency_mean_ms']<=lat and r['storage_bytes']<=storage]
  for r in feasible:
   keys=lambda z:(100-z['validation_accuracy_pct'],z['storage_bytes'],z['latency_mean_ms'])
   v=keys(r)
   if not any(all(a<=b for a,b in zip(keys(other),v)) and any(a<b for a,b in zip(keys(other),v)) for other in feasible if other is not r):
    front.append({'task':t,'candidate_id':r['candidate_id'],'precision':r['precision'],
                  'validation_accuracy_pct':r['validation_accuracy_pct'],
                  'storage_bytes':r['storage_bytes'],'latency_mean_ms':r['latency_mean_ms'],
                  'pareto_dimensions':3,'five_objective_certified':False})
 atomic_csv(root/OUT/'provisional_three_objective_pareto.csv',front)
 print(json.dumps({'measured_exports':len(data),'provisional_3d_frontier':len(front),
  'five_objective_certified':False,'note':'Exact native memory and qualified energy are not substituted'},indent=2))

def close(root,inv):
 """Run the existing REAL five-objective evidence/selection gate; no test access."""
 cmd=[sys.executable,str(root/'Codes'/'38_cajo_integrated_experimental_closure.py'),'audit','--root',str(root)]
 result=subprocess.run(cmd,cwd=root/'Codes',check=False)
 if result.returncode:raise RuntimeError('Step38 full-objective audit failed')
 gate=readj(root/'Reports'/'CAJO_Step38_Integrated_Closure_v1'/'evidence_gate.json')
 print(json.dumps({'ready_to_freeze':gate['ready_to_freeze'],'checks':gate['checks'],
 'sealed_test_access':'SEALED','next':'Run Step38 freeze ONLY if ready_to_freeze is true'},indent=2))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('action',choices=['run','baselines','compare','pareto','close'])
 p.add_argument('--root',type=Path,default=Path('..'))
 p.add_argument('--limit',type=int,default=0)
 p.add_argument('--batch-size',type=int,default=128)
 p.add_argument('--warmups',type=int,default=50)
 p.add_argument('--runs',type=int,default=100)
 a=p.parse_args();root=a.root.resolve();inv=context(root)
 if a.limit<0 or a.runs<2 or a.warmups<1 or a.batch_size<1:raise ValueError('Invalid run settings')
 if a.action=='run':run_exports(root,inv,a)
 elif a.action=='baselines':baselines(root,a)
 elif a.action=='compare':compare(root,inv)
 elif a.action=='pareto':pareto(root,inv)
 elif a.action=='close':close(root,inv)
if __name__=='__main__':
 try:main()
 except Exception as e:print('ERROR:',type(e).__name__,str(e),file=sys.stderr);sys.exit(2)
