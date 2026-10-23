#!/usr/bin/env python3
"""CAJO final single-entry, resumable research pipeline (Steps 1--23 preserved).

Run from Codes:
 python 24_cajo_final_experiments_and_manuscript.py inspect --root ..
 python 24_cajo_final_experiments_and_manuscript.py authorize --root .. --confirm APPROVE_CAJO_FINAL_RECOVERY
 python 24_cajo_final_experiments_and_manuscript.py run --root ..
 python 24_cajo_final_experiments_and_manuscript.py finalize --root .. --confirm FREEZE_VALIDATION_SELECTION
 python 24_cajo_final_experiments_and_manuscript.py test --root .. --confirm AUTHORIZE_SINGLE_SEALED_TEST
 python 24_cajo_final_experiments_and_manuscript.py report --root ..

Evidence rules:
 - Immutable original config, original 72 FP32, 72 INT8, and 24/48 screenings.
 - Predeclared validation-only screening and deterministic selection; no test leakage.
 - Final recovery is real 20-epoch, 3-seed training using the Step 02 training
   engine and Step 04 physical surgery; per-seed source pretraining is traced.
 - Same-source sequential and joint finalist recovery can be compared, but
   a greedy sequential search is NOT retrospectively fabricated.
 - CPU FP32 and INT8 are paired; GPU FP32 is a different deployment target.
 - CPU process RSS is NOT complete inference memory, and CPU energy is not
   available from the existing measurement backend. Therefore complete five-
   objective Pareto feasibility is NOT asserted. No silent substitutions.
 - Test access is a separate explicit command, requires a frozen selection,
   and is single-use. If evidence is insufficient, the command refuses.
 - Reporting is honest and can run before final test with limitations.

Dependencies: existing Codes 02,03,04,16,18,20,21,22,23; torch and local data.
No web access. No automatic manuscript overwriting. No rewriting old artifacts.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,os,platform,statistics,sys,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

LOCK='56361ba3ef2cb530d54a923cf4f5b3758a22bee82a9119822395cef0be9bbdc5'
CONFIG=Path('Reports/Optimization_Protocol/Final_Experiment_Config/cajo_experiment_config.json')
OUT=Path('Reports/CAJO_Final_Pipeline')
SRC=['02_experiment_runner.py','03_architecture_search_space.py','04_compression_recovery.py',
     '16_cajo_gpu_hardware_benchmark.py','18_cajo_final_joint_training.py',
     '20_cajo_alternative_fp32_benchmark.py','21_cajo_completion_pipeline.py',
     '22_cajo_int8_experiments.py','23_cajo_matched_joint_search.py',
     '24_cajo_final_experiments_and_manuscript.py']
TASKS=('CIFAR10','CIFAR100','SpeechCommands_v0.02')
ALT={'CIFAR10':'ResNet8_width075','CIFAR100':'ResNet18_width075',
     'SpeechCommands_v0.02':'DSCNN_width075'}
SEEDS=(42,43,44)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def mod(p,name):
 s=importlib.util.spec_from_file_location(name,str(p))
 if not s or not s.loader:raise RuntimeError('Cannot load module '+str(p))
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def atomic(p,obj,immutable=False):
 p.parent.mkdir(parents=True,exist_ok=True)
 if immutable and p.exists():
  if read(p)!=obj:raise RuntimeError('Immutable artifact changed: '+str(p))
  return
 tmp=p.with_name(p.name+'.partial')
 if tmp.exists():raise RuntimeError('Interrupted write; inspect '+str(tmp))
 with tmp.open('x',encoding='utf-8') as f:
  json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def csvout(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 cols=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
 tmp=p.with_name(p.name+'.partial')
 if tmp.exists():raise RuntimeError('Interrupted CSV write: '+str(tmp))
 with tmp.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def finite(x):
 try:
  v=float(x);return v if math.isfinite(v) else None
 except (ValueError,TypeError):return None

def initialize(root):
 cp=root/CONFIG
 if not cp.is_file() or sha(cp)!=LOCK:raise RuntimeError('LOCKED CONFIG HASH MISMATCH')
 cfg=read(cp)
 if cfg.get('status')!='LOCKED' or cfg['selection']['test_access']!='sealed_until_selection_frozen':
  raise RuntimeError('Locked protocol/test firewall changed')
 for task in TASKS:
  t=cfg['tasks'][task];p=root/Path(t['frozen_checkpoint'].replace('\\','/'))
  if not p.is_file() or sha(p)!=t['checkpoint_sha256']:
   raise RuntimeError('Frozen baseline changed: '+task)
 for name in SRC:
  if not (root/'Codes'/name).is_file():raise FileNotFoundError(root/'Codes'/name)
 s22=mod(root/'Codes'/SRC[7],'cajo24_s22')
 cfg2,items,s16,s20=s22.inventory(root)
 s22.verify_auth(root,items)
 if cfg2!=cfg:raise RuntimeError('Step22 inventory config differs')
 index={i['candidate']['candidate_id']:i for i in items}
 if len(index)!=72:raise RuntimeError('Expected exactly 72 historical FP32 runs')
 for item in items:
  s22.verify_item(root,cfg,item,s16,s20)
  cid=item['candidate']['candidate_id'];rp=root/'Reports'/'CAJO_INT8_CPU'/cid/'result.json'
  if not rp.is_file():raise RuntimeError('Step22 result missing: '+cid)
  r=read(rp)
  if r.get('test_access')!='SEALED' or r.get('source_checkpoint_sha256')!=sha(item['checkpoint']):
   raise RuntimeError('Step22 result drift: '+cid)
  for precision in ('fp32','int8'):
   ex=root/'Reports'/'CAJO_INT8_CPU'/cid/(precision+'.pt')
   if not ex.is_file() or sha(ex)!=r['exports'][precision]['sha256']:
    raise RuntimeError('Step22 export drift: '+cid)
 s23=mod(root/'Codes'/SRC[8],'cajo24_s23')
 _,screen_entries,_=s23.plan(root)
 s23.verify_auth(root,screen_entries)
 if len(screen_entries)!=24:raise RuntimeError('Expected 24 Step23 screenings')
 screens=[]
 for entry in screen_entries:
  c=entry['candidate'];cid=c['candidate_id']
  if not s23.check_history(root,c):raise RuntimeError('Incomplete Step23 training: '+cid)
  _,hist,ck,folder=s23.paths(root,c);p=folder/'result.json'
  if not p.is_file():raise RuntimeError('Step23 CPU result missing: '+cid)
  r=read(p)
  if r.get('test_access')!='SEALED' or r.get('checkpoint_sha256')!=sha(ck):
   raise RuntimeError('Step23 provenance mismatch: '+cid)
  for precision in ('fp32','int8'):
   ex=folder/(precision+'.pt')
   if not ex.is_file() or sha(ex)!=r['exports'][precision]['sha256']:
    raise RuntimeError('Step23 export drift: '+cid)
   screens.append(dict(candidate_id=cid,task=c['task'],architecture=c['architecture'],
    pruning_ratio=c['pruning_ratio'],kd=c['kd'],seed=42,precision=precision.upper(),
    accuracy_pct=r[precision+'_validation']['accuracy_percent'],
    latency_mean_ms=r['cpu_'+precision+'_benchmark']['latency_mean_ms'],
    latency_median_ms=r['cpu_'+precision+'_benchmark']['latency_median_ms'],
    storage_bytes=r['exports'][precision]['serialized_bytes'],
    checkpoint_sha256=sha(ck),result_sha256=sha(p),
    source_checkpoint_sha256=entry['source_sha256'],
    source_pretraining_not_matched=True))
 if len(screens)!=48:raise RuntimeError('Expected 48 verified Step23 precision variants')
 return cfg,items,index,screen_entries,screens,s22,s23

def score(row,cfg):
 t=cfg['tasks'][row['task']]
 return dict(accuracy_ok=row['accuracy_pct']>=t['accuracy_floor_percent'],
  storage_ok=row['storage_bytes']<=t['max_storage_mib']*1048576,
  latency_ok=row['latency_median_ms']<=t['max_latency_ms'])

def choose(cfg,screens):
 """Deterministic, predefined validation-only proxy shortlist, not full feasibility."""
 bytask=defaultdict(list)
 for r in screens:
  flags=score(r,cfg)
  x={**r,**flags, 'screening_proxy_eligible':all(flags.values())}
  bytask[r['task']].append(x)
 selections=[]
 for task in TASKS:
  rows=bytask[task]
  # Joint: screen all candidate tuples simultaneously; no greedy dimensional freezing.
  eligible=[r for r in rows if r['screening_proxy_eligible']]
  pool=eligible or rows
  # Locked primary preference accuracy, then latency and storage; CPU energy and
  # full runtime memory unavailable, so this is a PROVISIONAL proxy only.
  joint=sorted(pool,key=lambda r:(-r['accuracy_pct'],r['latency_median_ms'],r['storage_bytes'],r['candidate_id'],r['precision']))[0]
  # Sequential proxy: architecture chosen from unpruned pretrained reference
  # validation where available; source pretraining unequal. Then choose P25/P50
  # within that architecture. This is descriptive, NOT matched causal comparator.
  base_arch='baseline'
  seqpool=[r for r in rows if r['architecture']==base_arch and r['screening_proxy_eligible']]
  if not seqpool:seqpool=[r for r in rows if r['architecture']==base_arch]
  sequential=sorted(seqpool,key=lambda r:(-r['accuracy_pct'],r['latency_median_ms'],r['storage_bytes'],r['candidate_id'],r['precision']))[0]
  for strategy,r in (('joint',joint),('sequential',sequential)):
   selections.append({**r,'selection_strategy':strategy,
    'proxy_feasible_only':r['screening_proxy_eligible'],
    'screening_accuracy_floor':cfg['tasks'][task]['accuracy_floor_percent'],
    'selection_scope':'validation-only CPU accuracy/latency/serialized-storage proxy',
    'full_pareto_feasible':None})
 return selections

def selection_manifest(root,cfg,screens):
 selected=choose(cfg,screens)
 return dict(schema='cajo_step24_shortlist_v1',locked_config_sha256=LOCK,
  step23_authorization_sha256=sha(root/'Reports'/'CAJO_Matched_Search'/'authorization.json'),
  step23_result_digests={r['candidate_id']:r['result_sha256'] for r in screens},
  algorithm='validation-only deterministic accuracy-first; ties median CPU latency, serialized storage, ID',
  caveat='Proxy shortlist; no energy/full-memory evidence; not a full Pareto winner',
  source_pretraining_unequal=True,sequential_search_causal_comparator_established=False,
  selected=selected,test_access='SEALED')

def final_candidates(cfg,shortlist,screen_entries):
 lookup={e['candidate']['candidate_id']:e for e in screen_entries}
 runs=[]
 for s in shortlist['selected']:
  origin=lookup[s['candidate_id']]
  for seed in SEEDS:
   cid=f"{s['task']}__final24__{s['selection_strategy']}__"+('base' if s['architecture']=='baseline' else 'alt')+f"__P{round(100*s['pruning_ratio'])}__"+('KD' if s['kd'] else 'CE')+f'__s{seed}'
   c=dict(candidate_id=cid,task=s['task'],architecture=s['architecture'],
    pruning_ratio=s['pruning_ratio'],precision='FP32',kd=s['kd'],strategy=s['selection_strategy'],
    seed=seed,epochs=20,mode='recovery' if s['architecture']=='baseline' else 'architecture_training',
    operation_order=['verified_pretrained_source','physical_importance_pruning',
     'KD_recovery' if s['kd'] else 'CE_recovery'],pruning_policy='importance')
   source=origin['source_checkpoint']
   # IMPORTANT: source for alternative architecture must be seed-specific for
   # final training; do not silently use s42 P0 for seeds 43/44.
   if s['architecture']!='baseline':
    prior=f"{s['task']}__{s['architecture']}__pretrain_P0_CE__s{seed}"
    source=Path('Reports/CAJO_Experiments')/s['task']/'joint'/prior/f'seed_{seed}'/'best_model.pt'
   else:
    source=Path(cfg['tasks'][s['task']]['frozen_checkpoint'].replace('\\','/'))
   runs.append(dict(candidate=c,source_checkpoint=str(source),
    source_sha256=None,selected_precision=s['precision'],screening_candidate_id=s['candidate_id'],
    comparison_label=s['selection_strategy']))
 if len(runs)!=18:raise RuntimeError('Expected 6 finalist families x 3 seeds')
 return runs

def recovery_authorization(root,cfg,shortlist,runs):
 entries=[]
 for e in runs:
  p=root/e['source_checkpoint']
  if not p.is_file():raise RuntimeError('Finalist source missing: '+str(p))
  entries.append({**e,'source_sha256':sha(p)})
 return dict(schema='cajo_step24_recovery_authorization_v1',config_sha256=LOCK,
  source_code_sha256={'Codes/'+x:sha(root/'Codes'/x) for x in SRC},
  shortlist_sha256=sha(root/OUT/'shortlist.json'),runs=entries,
  scope='20_EPOCH_THREE_SEED_FINALIST_RECOVERY_VALIDATION_ONLY',
  no_test_access=True,immutable_original_experiments=True,
  caveats=['Source pretraining budgets differ','Sequential selection is a proxy, not a matched causal search',
   'No complete five-objective feasibility until memory and energy are available'])

def verify_auth(root,cfg,shortlist,runs):
 p=root/OUT/'recovery_authorization.json'
 if not p.is_file() or read(p)!=recovery_authorization(root,cfg,shortlist,runs):
  raise RuntimeError('Missing/drifted final recovery authorization. Run authorize after inspection.')

def runpath(root,c):
 d=root/OUT/'Training'/c['task']/c['strategy']/c['candidate_id']/f"seed_{c['seed']}"
 return d,d/'history.json',d/'best_model.pt'

def history_verified(root,c):
 _,p,b=runpath(root,c)
 if not p.is_file():return False
 h=read(p)
 if h.get('status')!='COMPLETE':return False
 if h.get('candidate')!=c or h.get('test_access')!='SEALED' or len(h.get('history',[]))!=20:
  raise RuntimeError('Finalist history drift: '+c['candidate_id'])
 if not b.is_file() or sha(b)!=h.get('best_checkpoint_sha256'):
  raise RuntimeError('Finalist checkpoint drift: '+c['candidate_id'])
 return True

def train_one(root,cfg,e,runs,shortlist):
 c=e['candidate'];verify_auth(root,cfg,shortlist,runs)
 if history_verified(root,c):print('SKIP TRAIN VERIFIED',c['candidate_id'],flush=True);return
 d,h,b=runpath(root,c)
 if h.exists() and not (d/'resume_state.pt').is_file():
  raise RuntimeError('Partial training without resume_state: '+str(d))
 runner=mod(root/'Codes'/'02_experiment_runner.py','cajo24_runner')
 adapter=mod(root/'Codes'/'04_compression_recovery.py','cajo24_adapter')
 runner.candidate_contract(c,cfg)
 source=root/e['source_checkpoint'];expected=sha(source)
 if expected!=next(x['source_sha256'] for x in read(root/OUT/'recovery_authorization.json')['runs'] if x['candidate']['candidate_id']==c['candidate_id']):
  raise RuntimeError('Pretraining source drift')
 def prepared(*,root,config,candidate,device):
  import torch
  arch,data,surgery,importance=adapter.sources(root)
  splits,manifest=adapter.verified_manifest(root,config,candidate['task'],data)
  teacher,baseline=adapter.frozen_teacher(root,config,candidate['task'],arch,torch)
  model=arch.build_architecture(candidate['task'],candidate['architecture'],config,root)
  if candidate['architecture']=='baseline':model.load_state_dict(teacher.state_dict(),strict=True)
  else:
   blob=torch.load(source,map_location='cpu',weights_only=True)
   expected_prior=f"{candidate['task']}__{candidate['architecture']}__pretrain_P0_CE__s{candidate['seed']}"
   if blob.get('candidate',{}).get('candidate_id')!=expected_prior:
    raise RuntimeError('Alternative P0 source mismatch')
   model.load_state_dict(blob['model'],strict=True)
  details=adapter.physical_prune(model,candidate['task'],candidate['pruning_ratio'],importance,surgery,
    arch,'importance',candidate['seed'],root)
  tr,val=adapter.make_data(root,candidate['task'],splits,config,candidate['seed'],data)
  prov=dict(test_access='sealed',frozen_manifest_sha256=sha(manifest),
   frozen_teacher_checkpoint_sha256=sha(baseline),architecture=candidate['architecture'],
   operation_order=candidate['operation_order'],initialization='verified_source',
   pretrained_checkpoint_sha256=sha(source),pruning_details=details,
   student_parameters=sum(p.numel() for p in model.parameters()),precision_training='FP32')
  return dict(student=model,teacher=teacher if candidate['kd'] else None,
   train_epoch_batches=tr,validation_batches=val,provenance=prov)
 class Proxy:prepare_candidate=staticmethod(prepared)
 old_gate,old_import,old_out=runner.run_gate,runner.import_adapter,runner.OUT_REL
 def gate(config):
  if config is not cfg:raise RuntimeError('Config identity drift')
  verify_auth(root,cfg,shortlist,runs)
  errors=[z for z in old_gate(config) if z not in ('Manuscript energy alignment pending','Full training not authorized')]
  if errors:raise RuntimeError('Training gate: '+'; '.join(errors))
  return []
 runner.run_gate=gate;runner.import_adapter=lambda _root:(Proxy,root/'Codes'/'04_compression_recovery.py')
 runner.OUT_REL=OUT/'Training'
 contract=root/OUT/'Contracts'/(c['candidate_id']+'.json')
 atomic(contract,c,immutable=True)
 try:
  print('TRAIN',c['candidate_id'],'RESUME' if (d/'resume_state.pt').is_file() else 'NEW',flush=True)
  runner.fit(root,cfg,root/CONFIG,c,contract,resume=(d/'resume_state.pt').exists())
 finally:
  runner.run_gate,runner.import_adapter,runner.OUT_REL=old_gate,old_import,old_out

def evaluate_final(root,cfg,e,shortlist,runs):
 import torch
 torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 c=e['candidate'];verify_auth(root,cfg,shortlist,runs)
 if not history_verified(root,c):raise RuntimeError('Finalist not trained: '+c['candidate_id'])
 d,h,checkpoint=runpath(root,c)
 dest=root/OUT/'CPU_Final'/c['candidate_id'];p=dest/'result.json'
 if p.exists():
  r=read(p)
  if r.get('checkpoint_sha256')!=sha(checkpoint) or r.get('test_access')!='SEALED':
   raise RuntimeError('Existing finalist CPU result drift')
  for precision in ('fp32','int8'):
   ex=dest/(precision+'.pt')
   if not ex.is_file() or sha(ex)!=r['exports'][precision]['sha256']:
    raise RuntimeError('Existing CPU export drift')
  print('SKIP CPU VERIFIED',c['candidate_id'],flush=True);return
 if dest.exists() and any(dest.iterdir()):raise RuntimeError('Partial CPU result: '+str(dest))
 adapter=mod(root/'Codes'/'04_compression_recovery.py','cajo24_cpu_adapter')
 arch=mod(root/'Codes'/'03_architecture_search_space.py','cajo24_cpu_arch')
 s22=mod(root/'Codes'/'22_cajo_int8_experiments.py','cajo24_cpu22')
 model=arch.build_architecture(c['task'],c['architecture'],cfg,root)
 source=root/e['source_checkpoint']
 if sha(source)!=next(x['source_sha256'] for x in read(root/OUT/'recovery_authorization.json')['runs'] if x['candidate']['candidate_id']==c['candidate_id']):
  raise RuntimeError('Source drift')
 if c['architecture']=='baseline':
  teacher,_=adapter.frozen_teacher(root,cfg,c['task'],arch,torch)
  model.load_state_dict(teacher.state_dict(),strict=True)
 else:
  blob=torch.load(source,map_location='cpu',weights_only=True)
  model.load_state_dict(blob['model'],strict=True)
 _,_,surgery,importance=adapter.sources(root)
 adapter.physical_prune(model,c['task'],c['pruning_ratio'],importance,surgery,arch,'importance',c['seed'],root)
 blob=torch.load(checkpoint,map_location='cpu',weights_only=True)
 if blob.get('candidate')!=c:raise RuntimeError('Final checkpoint candidate drift')
 model.load_state_dict(blob['model'],strict=True);model.cpu().eval()
 factory,manifest=s22.validation_factory(root,cfg,c,adapter)
 val0=s22.evaluate(model,factory,torch)
 bench0=s22.cpu_measure(model,arch.SHAPES[c['task']],torch,warmups=50,runs=100)
 quant,calibration=adapter.quantize_recovered_model(model,task=c['task'],root=root,config=cfg,seed=c['seed'])
 ops=s22.int8_graph_evidence(quant,torch)
 if not ops:raise RuntimeError('No quantized operators')
 val1=s22.evaluate(quant,factory,torch)
 bench1=s22.cpu_measure(quant,arch.SHAPES[c['task']],torch,warmups=50,runs=100)
 dest.mkdir(parents=True,exist_ok=True)
 exports={}
 for label,m in (('fp32',model),('int8',quant)):
  digest,nbytes=s22.export(m,dest/(label+'.pt'),torch,arch.SHAPES[c['task']])
  exports[label]=dict(sha256=digest,serialized_bytes=nbytes)
 atomic(p,dict(schema='cajo_step24_final_cpu_v1',candidate_id=c['candidate_id'],candidate=c,
  checkpoint_sha256=sha(checkpoint),history_sha256=sha(h),source_sha256=sha(source),
  dataset_manifest_sha256=manifest,calibration=calibration,int8_operator_evidence=ops,
  fp32_validation=val0,int8_validation=val1,cpu_fp32_benchmark=bench0,cpu_int8_benchmark=bench1,
  exports=exports,hardware='CPU_1_THREAD',test_access='SEALED',
  complete_runtime_memory_bytes=None,cpu_energy_mj_per_inference=None,
  limitations=['CPU energy not instrumented','Complete process/native memory not measured']),immutable=True)
 print('CPU DONE',c['candidate_id'],flush=True)

def rows_final(root,cfg,runs):
 rows=[];missing=[]
 for e in runs:
  c=e['candidate'];cid=c['candidate_id'];_,_,ck=runpath(root,c)
  if not history_verified(root,c):missing.append('Training '+cid);continue
  p=root/OUT/'CPU_Final'/cid/'result.json'
  if not p.is_file():missing.append('CPU '+cid);continue
  r=read(p)
  if r['checkpoint_sha256']!=sha(ck):raise RuntimeError('Final checkpoint/result drift')
  for precision in ('fp32','int8'):
   ex=root/OUT/'CPU_Final'/cid/(precision+'.pt')
   if sha(ex)!=r['exports'][precision]['sha256']:raise RuntimeError('Final export drift')
   val=r[precision+'_validation'];bm=r['cpu_'+precision+'_benchmark']
   row=dict(candidate_id=cid,task=c['task'],architecture=c['architecture'],strategy=c['strategy'],
    seed=c['seed'],pruning_ratio=c['pruning_ratio'],kd=c['kd'],precision=precision.upper(),
    validation_accuracy_pct=val['accuracy_percent'],latency_mean_ms=bm['latency_mean_ms'],
    latency_median_ms=bm['latency_median_ms'],storage_bytes=r['exports'][precision]['serialized_bytes'],
    total_runtime_memory_bytes=None,energy_mj_per_inference=None,
    hardware='CPU_ONE_THREAD',test_access='SEALED',source_sha256=r['source_sha256'],
    checkpoint_sha256=r['checkpoint_sha256'],result_sha256=sha(p),
    latency_scope='CPU model forward wall clock; batch1; 50 warmups; 100 runs',
    storage_scope='TorchScript serialized deployment model')
   row.update(score(dict(task=c['task'],accuracy_pct=row['validation_accuracy_pct'],
    storage_bytes=row['storage_bytes'],latency_median_ms=row['latency_median_ms']),cfg))
   row['partial_constraints_pass']=row['accuracy_ok'] and row['storage_ok'] and row['latency_ok']
   row['full_feasibility']='UNKNOWN: CPU energy and complete memory missing'
   rows.append(row)
 return rows,missing

def grouped(rows):
 groups=defaultdict(list)
 for r in rows:
  key=(r['task'],r['architecture'],r['strategy'],r['pruning_ratio'],r['kd'],r['precision'])
  groups[key].append(r)
 out=[]
 for key,items in sorted(groups.items()):
  rec=dict(zip(('task','architecture','strategy','pruning_ratio','kd','precision'),key))
  rec['n']=len(items);rec['seeds']=','.join(str(z['seed']) for z in sorted(items,key=lambda z:z['seed']))
  for metric in ('validation_accuracy_pct','latency_mean_ms','latency_median_ms','storage_bytes'):
   vals=[z[metric] for z in items]
   rec[metric+'_mean']=statistics.mean(vals)
   rec[metric+'_sd']=statistics.stdev(vals) if len(vals)>1 else None
  rec['partial_constraints_all_seeds']=all(z['partial_constraints_pass'] for z in items)
  rec['full_feasibility']='UNKNOWN'
  out.append(rec)
 return out

def pareto_proxy(rows):
 """Only descriptive three-objective CPU proxy; NEVER label full five-objective Pareto."""
 out=[]
 for task in TASKS:
  rs=[r for r in rows if r['task']==task and r['partial_constraints_pass']]
  for a in rs:
   metrics=('validation_accuracy_pct','latency_median_ms','storage_bytes')
   dominated=False
   for b in rs:
    if a is b:continue
    better=(b[metrics[0]]>=a[metrics[0]] and b[metrics[1]]<=a[metrics[1]] and b[metrics[2]]<=a[metrics[2]])
    strict=(b[metrics[0]]>a[metrics[0]] or b[metrics[1]]<a[metrics[1]] or b[metrics[2]]<a[metrics[2]])
    if better and strict:dominated=True;break
   out.append({**a,'proxy_nondominated':not dominated,'scope':'CPU accuracy/latency/storage only; NOT full Pareto'})
 return out

def freeze(root,cfg,shortlist,runs):
 rows,missing=rows_final(root,cfg,runs)
 if missing:raise RuntimeError('Cannot freeze; missing '+str(missing[:8]))
 # Deterministic family selection on 3-seed means; do not use sealed test.
 groups=grouped(rows)
 chosen=[]
 for task in TASKS:
  candidates=[g for g in groups if g['task']==task and g['n']==3 and g['partial_constraints_all_seeds']]
  if not candidates:
   chosen.append(dict(task=task,status='NO_PARTIALLY_QUALIFIED_FINALIST',full_feasibility='UNKNOWN'))
   continue
  best=sorted(candidates,key=lambda g:(-g['validation_accuracy_pct_mean'],
   g['latency_median_ms_mean'],g['storage_bytes_mean'],g['strategy'],g['precision']))[0]
  chosen.append(dict(task=task,status='VALIDATION_PROXY_SELECTED',
   configuration={k:best[k] for k in ('architecture','strategy','pruning_ratio','kd','precision')},
   validation_accuracy_mean=best['validation_accuracy_pct_mean'],
   full_feasibility='UNKNOWN: mandatory memory and energy objectives unavailable'))
 obj=dict(schema='cajo_step24_frozen_validation_selection_v1',locked_config_sha256=LOCK,
  shortlist_sha256=sha(root/OUT/'shortlist.json'),
  recovery_authorization_sha256=sha(root/OUT/'recovery_authorization.json'),
  final_cpu_result_digests={r['candidate_id']:r['result_sha256'] for r in rows},
  selected=chosen,selection_basis='3-seed validation, CPU accuracy/storage/latency proxy only',
  full_five_objective_feasibility=False,full_pareto_calculated=False,
  matched_sequential_vs_joint_causal_search=False,test_access='SEALED',
  test_authorization_required=True)
 atomic(root/OUT/'frozen_validation_selection.json',obj,immutable=True)
 return obj

def final_test(root,cfg,shortlist,runs):
 """Fail closed: a single test pass is scientifically inappropriate until full
 deployment feasibility and a genuine validated selection rule are available.
 We do not invent a test-loader API or bypass the frozen split manifest."""
 p=root/OUT/'frozen_validation_selection.json'
 if not p.is_file():raise RuntimeError('Validation selection not frozen')
 frozen=read(p)
 if frozen.get('full_five_objective_feasibility') is not True:
  raise RuntimeError('SEALED TEST BLOCKED: CPU energy and complete runtime memory are missing; '+
   'full constrained selection is not established. No test data accessed. '+
   'The manuscript must report this limitation or obtain verified measurements '+
   'and explicitly revise/approve the protocol before final testing.')
 raise RuntimeError('SEALED TEST BLOCKED: no independently verified test-only evaluation adapter; '+
  'do not access test through validation loaders.')

def write_report(root,cfg,shortlist,runs,screens):
 out=root/OUT;out.mkdir(parents=True,exist_ok=True)
 csvout(out/'screening_verified.csv',screens)
 csvout(out/'shortlisted_families.csv',shortlist['selected'])
 rows,missing=rows_final(root,cfg,runs)
 csvout(out/'final_cpu_per_seed.csv',rows)
 gr=grouped(rows);csvout(out/'final_cpu_three_seed.csv',gr)
 proxy=pareto_proxy(rows);csvout(out/'cpu_three_objective_proxy.csv',proxy)
 screening_groups=defaultdict(list)
 for r in screens:screening_groups[(r['task'],r['architecture'],r['pruning_ratio'],r['kd'],r['precision'])].append(r)
 issues=['No CPU energy measurement; five-objective Pareto unavailable',
  'No independently verified complete inference memory; full feasibility unavailable',
  'Source pretraining not matched across architectures',
  'Joint-vs-sequential search is not a matched causal comparator',
  'Step23 one screening seed; final three-seed runs are distinct',
  'CPU and GPU results are separate hardware targets',
  'GPU allocator peak is not complete runtime memory',
  'NVML GPU energy is not total-system energy or CPU energy',
  'Historical GPU benchmark warmups may differ from locked 50 warmups']
 status=dict(schema='cajo_step24_report_status_v1',timestamp_utc=datetime.now(timezone.utc).isoformat(),
  existing_fp32=72,existing_cpu_int8=72,screening_fp32=24,screening_precision_variants=48,
  finalist_training_complete=sum(history_verified(root,e['candidate']) for e in runs),
  finalist_training_expected=len(runs),final_cpu_precision_rows=len(rows),
  final_cpu_precision_expected=2*len(runs),missing=missing,
  frozen_selection=(out/'frozen_validation_selection.json').is_file(),
  test_access='SEALED',full_feasibility=False,full_pareto=False,limitations=issues)
 atomic(out/'status.json',status)
 def fmt(v):return '—' if v is None else f'{v:.3f}'
 lines=['# CAJO manuscript evidence package', '',
  '**Status:** '+('FINALIST EXPERIMENTS COMPLETE; full feasibility not established.' if not missing else 'PROVISIONAL: finalist experiments incomplete.'),
  '', '## Section 3. Experimental setup (verified scope)', '',
  'Datasets: CIFAR-10, CIFAR-100, Speech Commands v0.02. Frozen dataset split manifests and baseline checkpoints are SHA-verified.',
  'Training uses SGD, cosine schedule and the locked configuration. Screening comprises 24 five-epoch FP32 training configurations at seed 42,',
  'with 48 CPU FP32/INT8 validation and inference variants. Finalist recovery comprises up to 18 twenty-epoch three-seed training runs.',
  'INT8 uses train-only post-training calibration. CPU model-forward timings use batch 1, one thread, 50 warmups and 100 runs.',
  'Source pretraining is not compute-matched across architectures. This study cannot claim a matched causal sequential-vs-joint search.',
  '', '## Section 4. Screening and finalist results', '',
  '### Step 23 screening evidence (one seed)', '',
  '| Task | Architecture | Pruning | KD | Precision | Validation accuracy (%) | CPU median latency (ms) | Serialized storage (KiB) |',
  '|---|---|---:|---|---|---:|---:|---:|']
 for r in sorted(screens,key=lambda z:(z['task'],z['architecture'],z['pruning_ratio'],z['kd'],z['precision'])):
  lines.append(f"| {r['task']} | {r['architecture']} | {r['pruning_ratio']:.2f} | {r['kd']} | {r['precision']} | {fmt(r['accuracy_pct'])} | {fmt(r['latency_median_ms'])} | {r['storage_bytes']/1024:.1f} |")
 lines.extend(['','### Finalist three-seed results', '',
  '| Task | Architecture | Strategy label | Pruning | KD | Precision | N | Validation accuracy mean ± SD (%) | CPU latency median mean (ms) | Storage mean (KiB) |',
  '|---|---|---|---:|---|---|---:|---|---:|---:|'])
 for g in gr:
  lines.append(f"| {g['task']} | {g['architecture']} | {g['strategy']} | {g['pruning_ratio']:.2f} | {g['kd']} | {g['precision']} | {g['n']} | {fmt(g['validation_accuracy_pct_mean'])} ± {fmt(g['validation_accuracy_pct_sd'])} | {fmt(g['latency_median_ms_mean'])} | {g['storage_bytes_mean']/1024:.1f} |")
 lines.extend(['','## Section 5. Discussion and limitations', '',
  'The completed comparisons directly support quantization and pruning/knowledge-distillation effects on validation accuracy,',
  'serialized model size, and CPU model-forward latency. CPU INT8 must not be compared against GPU FP32 as a controlled precision ablation.',
  'CPU energy and complete native inference memory were not measured. Thus the manuscript’s complete five-objective',
  'feasibility and Pareto claims are not established, and any three-objective CPU Pareto is labeled descriptive only.',
  'A five-epoch search with unequal upstream pretraining cannot establish equal total-compute optimization,',
  'and selecting a baseline-family candidate after joint screening is not a prospective sequential search comparator.',
  'No sealed-test performance is reported unless an independently authorized and validated test-only adapter is executed.',
  '', '## Claims-to-evidence matrix', '',
  '| Claim | Evidence | Status |','|---|---|---|',
  '| 72 FP32 candidate evaluations | Steps 16/20/21 | Verified |',
  '| 72 CPU FP32/INT8 PTQ comparisons | Step 22 | Verified |',
  '| 24 matched five-epoch screening trainings, 48 precision variants | Step 23 | Verified (training epochs only) |',
  '| Three-seed finalist recovery | Step 24 training histories | '+('Verified' if not missing else 'Incomplete')+' |',
  '| Same-target accuracy, latency and storage comparisons | Step 24 CPU results | '+('Verified' if not missing else 'Incomplete')+' |',
  '| Full native inference memory | No validated instrumentation | Not established |',
  '| CPU inference energy | No CPU energy backend | Not established |',
  '| Five-objective constrained Pareto optimality | Missing mandatory objectives | Not established |',
  '| Matched causal sequential vs joint search | No prospective matched sequential search | Not established |',
  '| Independent sealed-test metrics | Test not accessed | Not available |',
  '', '## Remaining evidence gaps', ''])
 lines.extend('- '+i for i in issues)
 if missing:lines.extend(['','## Missing candidate results']+['- '+m for m in missing])
 (out/'manuscript_sections_3_to_5.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (out/'figure_generation_notes.md').write_text(
  '# Figures and data\n\nUse final_cpu_three_seed.csv for accuracy/storage/latency plots, '+
  'and cpu_three_objective_proxy.csv only for an explicitly labeled three-objective CPU proxy. '+
  'Do not draw energy or full-memory plots without valid measurements.\n',encoding='utf-8')
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  for task in TASKS:
   subset=[g for g in gr if g['task']==task]
   if not subset:continue
   fig,ax=plt.subplots(figsize=(7.5,5))
   for g in subset:
    ax.scatter(g['storage_bytes_mean']/1048576,g['validation_accuracy_pct_mean'],
      marker='o' if g['precision']=='FP32' else 's',s=50,
      label=f"{g['strategy']} {g['architecture']} P{int(g['pruning_ratio']*100)} {'KD' if g['kd'] else 'CE'} {g['precision']}")
   ax.axhline(cfg['tasks'][task]['accuracy_floor_percent'],linestyle='--',linewidth=1)
   ax.set_xlabel('Serialized CPU inference model (MiB)');ax.set_ylabel('Validation accuracy (%)')
   ax.set_title(task+' — CPU accuracy/storage (validation)');ax.grid(alpha=.2)
   ax.legend(fontsize=6,loc='best');fig.tight_layout()
   fig.savefig(out/('figure_accuracy_storage_'+task+'.png'),dpi=180);plt.close(fig)
 except ImportError:issues.append('Matplotlib not installed; figures skipped')
 print(json.dumps(status,indent=2))
 return status

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('action',choices=['inspect','authorize','run','finalize','test','report'])
 ap.add_argument('--root',type=Path,default=Path('..'))
 ap.add_argument('--confirm');ap.add_argument('--candidate-id',help='Optional single finalist for run')
 args=ap.parse_args();root=args.root.resolve()
 cfg,items,index,entries,screens,s22,s23=initialize(root)
 out=root/OUT;out.mkdir(parents=True,exist_ok=True)
 desired=selection_manifest(root,cfg,screens)
 # A previously frozen shortlist cannot silently change when rerun.
 atomic(out/'shortlist.json',desired,immutable=True)
 runs=final_candidates(cfg,desired,entries)
 if args.action=='inspect':
  summary=dict(config_sha256=LOCK,existing_fp32=72,existing_int8=72,
   screening_fp32=24,screening_precision_variants=48,
   shortlist=desired['selected'],finalist_runs=len(runs),
   completed_finalist_runs=sum(history_verified(root,e['candidate']) for e in runs),
   authorized=(out/'recovery_authorization.json').exists(),
   test_access='SEALED',full_feasibility=False,
   limitations=['No complete native memory','No CPU energy','No matched sequential-search causal comparison'])
  print(json.dumps(summary,indent=2));return
 if args.action=='authorize':
  if args.confirm!='APPROVE_CAJO_FINAL_RECOVERY':raise RuntimeError('Explicit authorization confirmation required')
  obj=recovery_authorization(root,cfg,desired,runs)
  atomic(out/'recovery_authorization.json',obj,immutable=True)
  print('AUTHORIZED exactly',len(runs),'finalist recoveries; test remains SEALED');return
 if args.action=='run':
  verify_auth(root,cfg,desired,runs)
  selected=[e for e in runs if not args.candidate_id or e['candidate']['candidate_id']==args.candidate_id]
  if not selected:raise RuntimeError('Unknown --candidate-id')
  for e in selected:
   train_one(root,cfg,e,runs,desired)
   evaluate_final(root,cfg,e,desired,runs)
  write_report(root,cfg,desired,runs,screens);return
 if args.action=='finalize':
  if args.confirm!='FREEZE_VALIDATION_SELECTION':raise RuntimeError('Explicit freeze confirmation required')
  verify_auth(root,cfg,desired,runs)
  print(json.dumps(freeze(root,cfg,desired,runs),indent=2));return
 if args.action=='test':
  if args.confirm!='AUTHORIZE_SINGLE_SEALED_TEST':raise RuntimeError('Explicit test confirmation required')
  final_test(root,cfg,desired,runs);return
 if args.action=='report':write_report(root,cfg,desired,runs,screens)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  print('CAJO STEP24 STOPPED:',type(exc).__name__+':',exc,file=sys.stderr)
  sys.exit(2)
