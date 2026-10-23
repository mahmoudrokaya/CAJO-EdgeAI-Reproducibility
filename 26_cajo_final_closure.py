#!/usr/bin/env python3
"""CAJO independent sequential-search experiment and five-objective evidence closure.

ALL outputs go exclusively to Reports/CAJO_Independent_Closure_v1. Steps 23-25
and locked artifacts are READ ONLY. Never changes their outputs. No test access.

Commands (run in Codes):
 python 26_cajo_final_closure.py inspect --root ..
 python 26_cajo_final_closure.py authorize --root .. --confirm APPROVE_NEW_SEQUENTIAL_CAMPAIGN
 python 26_cajo_final_closure.py run --root ..
 python 26_cajo_final_closure.py meter-template --root ..
 python 26_cajo_final_closure.py finalize --root ..

A sequential search here is a genuinely *executed adaptive search policy*, not
just relabelled Step24 training. The joint comparator reuses Step23's complete
8-configuration, 5-epoch grid per task. Sequential uses the same 8 x 5-epoch
screening-job budget per task, but adaptively allocates the jobs: two architecture
probes, one pruning probe, two KD probes, and three independent seed-43
replications of validation-ranked candidates. The underlying physical-pruning
and recovery implementation is shared. This is a comparison of SEARCH POLICIES,
not proof of simultaneous differentiable co-optimization or matched total
pretraining compute. The baseline/alternative sources have different pretraining
histories. Finalists use 20 epochs x 3 seeds, stored only in this new folder.

Deployment uses CPU, batch=1, torch one thread, 50 warmups, 100 timings,
FP32 and calibrated INT8. CPU energy requires actual external meter evidence;
no power reading is fabricated. Process RSS is a lower-bound/process-memory
proxy, NOT a verified complete peak activation/workspace metric; therefore it
cannot by itself close the manuscript's native-memory requirement. External
measurements must be accompanied by raw logs and verified hashes. A full five-
objective Pareto front is only certified when verified full-scope memory and
energy are provided. Test remains SEALED; no test adapter is available.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,os,platform,statistics,sys,time
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone

LOCK='56361ba3ef2cb530d54a923cf4f5b3758a22bee82a9119822395cef0be9bbdc5'
CONFIG=Path('Reports/Optimization_Protocol/Final_Experiment_Config/cajo_experiment_config.json')
OUT=Path('Reports/CAJO_Independent_Closure_v1')
TASKS=('CIFAR10','CIFAR100','SpeechCommands_v0.02')
ALT={'CIFAR10':'ResNet8_width075','CIFAR100':'ResNet18_width075','SpeechCommands_v0.02':'DSCNN_width075'}
SEEDS=(42,43,44)
SRC=('02_experiment_runner.py','03_architecture_search_space.py','04_compression_recovery.py','22_cajo_int8_experiments.py','23_cajo_matched_joint_search.py','24_cajo_final_experiments_and_manuscript.py','25_cajo_evidence_completion.py')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,str(path))
 if not spec or not spec.loader:raise RuntimeError('Cannot import '+str(path))
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def immutable_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if read(path)!=obj:raise RuntimeError('IMMUTABLE ARTIFACT DRIFT '+str(path))
  return
 tmp=path.with_name(path.name+'.partial')
 if tmp.exists():raise RuntimeError('Unresolved partial write '+str(tmp))
 with tmp.open('x',encoding='utf-8') as f:
  json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def write_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.partial')
 if tmp.exists():raise RuntimeError('Unresolved partial write '+str(tmp))
 with tmp.open('x',encoding='utf-8') as f:
  json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def write_csv(path,rows,cols=None):
 path.parent.mkdir(parents=True,exist_ok=True)
 cols=cols or list(dict.fromkeys(k for r in rows for k in r)) or ['status']
 tmp=path.with_name(path.name+'.partial')
 if tmp.exists():raise RuntimeError('Unresolved partial write '+str(tmp))
 with tmp.open('x',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
 os.replace(tmp,path)

def context(root):
 if sha(root/CONFIG)!=LOCK:raise RuntimeError('Locked config hash changed')
 for name in SRC:
  if not (root/'Codes'/name).is_file():raise FileNotFoundError(root/'Codes'/name)
 s24=module(root/'Codes'/'24_cajo_final_experiments_and_manuscript.py','closure24')
 cfg,items,index,entries,screens,s22,s23=s24.initialize(root)
 shortlist_path=root/'Reports/CAJO_Final_Pipeline/shortlist.json'
 if not shortlist_path.is_file():raise RuntimeError('Step24 shortlist missing')
 shortlist=read(shortlist_path)
 runs=s24.final_candidates(cfg,shortlist,entries)
 s24.verify_auth(root,cfg,shortlist,runs)
 final_rows,missing=s24.rows_final(root,cfg,runs)
 if missing:raise RuntimeError('Step24 incomplete: '+str(missing[:5]))
 if len(screens)!=48 or len(final_rows)!=36:raise RuntimeError('Historical evidence count changed')
 return dict(root=root,cfg=cfg,s24=s24,s22=s22,s23=s23,screens=screens,final_rows=final_rows,entries=entries)

def approval_expected(ctx):
 root=ctx['root']
 return dict(schema='cajo_independent_closure_authorization_v1',locked_config_sha256=LOCK,
  source_hashes={'Codes/'+n:sha(root/'Codes'/n) for n in SRC},
  step23_auth_sha256=sha(root/'Reports/CAJO_Matched_Search/authorization.json'),
  step24_auth_sha256=sha(root/'Reports/CAJO_Final_Pipeline/recovery_authorization.json'),
  scope='NEW_FOLDER_ONLY; 24 adaptive sequential screening jobs (5 epochs); 9 sequential finalist recoveries (20 epochs); CPU FP32/INT8; validation only',
  search_budget='8 independent 5-epoch screening jobs per task for each policy; joint Step23 reused',
  caveats=['Joint arm is prior Step23 grid search; sequential arm is new adaptive search',
   'Architecture pretraining unequal; total compute not fully matched',
   'Same compression/recovery kernel; difference is candidate search policy',
   'No full energy or complete native-memory claim without external metering',
   'Sealed test never accessed'])

def require_approval(ctx):
 p=ctx['root']/OUT/'authorization.json'
 if not p.is_file() or read(p)!=approval_expected(ctx):
  raise RuntimeError('Authorization missing or drifted; run authorize with explicit confirmation')

def source_for(ctx,task,arch,seed):
 root,cfg=ctx['root'],ctx['cfg']
 if arch=='baseline':
  p=root/Path(cfg['tasks'][task]['frozen_checkpoint'].replace('\\','/'))
  expected=cfg['tasks'][task]['checkpoint_sha256']
 else:
  cid=f'{task}__{arch}__pretrain_P0_CE__s{seed}'
  p=root/'Reports/CAJO_Experiments'/task/'joint'/cid/f'seed_{seed}'/'best_model.pt'
  expected=sha(p)
 if sha(p)!=expected:raise RuntimeError('Source checkpoint drift '+str(p))
 return p,expected

def candidate(ctx,task,arch,prune,kd,seed,epochs,stage):
 label='base' if arch=='baseline' else 'alt'
 cid=f'{task}__closure26__{stage}__{label}__P{int(prune*100)}__'+('KD' if kd else 'CE')+f'__s{seed}'
 return dict(candidate_id=cid,task=task,architecture=arch,pruning_ratio=prune,
  precision='FP32',kd=bool(kd),strategy='screening' if epochs==5 else 'sequential',
  seed=seed,epochs=epochs,mode='recovery' if arch=='baseline' else 'architecture_training',
  operation_order=['verified_pretrained_source','physical_importance_pruning',
   'KD_recovery' if kd else 'CE_recovery'],pruning_policy='importance')

def paths(ctx,c):
 d=ctx['root']/OUT/'Training'/c['task']/c['strategy']/c['candidate_id']/f"seed_{c['seed']}"
 return d,d/'history.json',d/'best_model.pt'

def history_ok(ctx,c):
 _,h,b=paths(ctx,c)
 if not h.is_file():return False
 v=read(h)
 if v.get('candidate')!=c or v.get('test_access')!='SEALED':raise RuntimeError('History provenance drift')
 if v.get('status')!='COMPLETE':return False
 if len(v.get('history',[]))!=c['epochs'] or not b.is_file() or sha(b)!=v.get('best_checkpoint_sha256'):
  raise RuntimeError('Completed history/checkpoint corrupt '+c['candidate_id'])
 return True

def train(ctx,c):
 require_approval(ctx)
 root,cfg=ctx['root'],ctx['cfg']
 if history_ok(ctx,c):print('SKIP VERIFIED TRAIN',c['candidate_id'],flush=True);return
 runner=module(root/'Codes'/'02_experiment_runner.py','closure_runner')
 adapter=module(root/'Codes'/'04_compression_recovery.py','closure_adapter')
 runner.candidate_contract(c,cfg)
 src,expected=source_for(ctx,c['task'],c['architecture'],c['seed'])
 def prepared(*,root,config,candidate,device):
  import torch
  arch,data,surgery,importance=adapter.sources(root)
  splits,manifest=adapter.verified_manifest(root,config,candidate['task'],data)
  teacher,baseline=adapter.frozen_teacher(root,config,candidate['task'],arch,torch)
  model=arch.build_architecture(candidate['task'],candidate['architecture'],config,root)
  if candidate['architecture']=='baseline':model.load_state_dict(teacher.state_dict(),strict=True)
  else:
   blob=torch.load(src,map_location='cpu',weights_only=True)
   prior=f"{candidate['task']}__{candidate['architecture']}__pretrain_P0_CE__s{candidate['seed']}"
   if blob.get('candidate',{}).get('candidate_id')!=prior:raise RuntimeError('Alternative source provenance mismatch')
   model.load_state_dict(blob['model'],strict=True)
  if sha(src)!=expected:raise RuntimeError('Source mutated')
  details=adapter.physical_prune(model,candidate['task'],candidate['pruning_ratio'],importance,
    surgery,arch,'importance',candidate['seed'],root)
  tr,val=adapter.make_data(root,candidate['task'],splits,config,candidate['seed'],data)
  prov=dict(test_access='sealed',frozen_manifest_sha256=sha(manifest),
   frozen_teacher_checkpoint_sha256=sha(baseline),architecture=candidate['architecture'],
   operation_order=candidate['operation_order'],initialization='verified_pretrained_source',
   pretrained_checkpoint_sha256=expected,pruning_details=details,
   student_parameters=sum(p.numel() for p in model.parameters()),precision_training='FP32')
  return dict(student=model,teacher=teacher if candidate['kd'] else None,
   train_epoch_batches=tr,validation_batches=val,provenance=prov)
 class Proxy:prepare_candidate=staticmethod(prepared)
 old_gate,old_import,old_out=runner.run_gate,runner.import_adapter,runner.OUT_REL
 def gate(conf):
  if conf is not cfg:raise RuntimeError('Config identity drift')
  require_approval(ctx)
  blockers=[x for x in old_gate(conf) if x not in ('Manuscript energy alignment pending','Full training not authorized')]
  if blockers:raise RuntimeError('Training gate: '+'; '.join(blockers))
  return []
 runner.run_gate=gate
 runner.import_adapter=lambda _:(Proxy,root/'Codes'/'04_compression_recovery.py')
 runner.OUT_REL=OUT/'Training'
 contract=root/OUT/'Contracts'/(c['candidate_id']+'.json')
 immutable_json(contract,c)
 d,_,_=paths(ctx,c)
 try:
  print('TRAIN',c['candidate_id'],flush=True)
  runner.fit(root,cfg,root/CONFIG,c,contract,resume=(d/'resume_state.pt').is_file())
 finally:
  runner.run_gate,runner.import_adapter,runner.OUT_REL=old_gate,old_import,old_out

def cpu(ctx,c):
 require_approval(ctx)
 root,cfg=ctx['root'],ctx['cfg']
 if not history_ok(ctx,c):raise RuntimeError('Training incomplete '+c['candidate_id'])
 _,hist,ck=paths(ctx,c)
 dest=root/OUT/'CPU'/c['candidate_id'];report=dest/'result.json'
 if report.is_file():
  r=read(report)
  if r['checkpoint_sha256']!=sha(ck):raise RuntimeError('CPU evidence checkpoint drift')
  for prec in ('fp32','int8'):
   if sha(dest/(prec+'.pt'))!=r['exports'][prec]['sha256']:raise RuntimeError('CPU export drift')
  print('SKIP VERIFIED CPU',c['candidate_id'],flush=True);return
 if dest.exists() and any(dest.iterdir()):
  raise RuntimeError('Partial CPU evidence requires manual forensic inspection: '+str(dest))
 import torch
 torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 adapter=module(root/'Codes'/'04_compression_recovery.py','closure_cpu_adapter')
 arch=module(root/'Codes'/'03_architecture_search_space.py','closure_cpu_arch')
 s22=ctx['s22']
 src,expected=source_for(ctx,c['task'],c['architecture'],c['seed'])
 model=arch.build_architecture(c['task'],c['architecture'],cfg,root)
 if c['architecture']=='baseline':
  teacher,_=adapter.frozen_teacher(root,cfg,c['task'],arch,torch)
  model.load_state_dict(teacher.state_dict(),strict=True)
 else:model.load_state_dict(torch.load(src,map_location='cpu',weights_only=True)['model'],strict=True)
 _,_,surgery,importance=adapter.sources(root)
 adapter.physical_prune(model,c['task'],c['pruning_ratio'],importance,surgery,arch,'importance',c['seed'],root)
 blob=torch.load(ck,map_location='cpu',weights_only=True)
 if blob.get('candidate')!=c:raise RuntimeError('Checkpoint candidate mismatch')
 model.load_state_dict(blob['model'],strict=True);model.cpu().eval()
 factory,manifest=s22.validation_factory(root,cfg,c,adapter)
 v0=s22.evaluate(model,factory,torch)
 b0=s22.cpu_measure(model,arch.SHAPES[c['task']],torch,warmups=50,runs=100)
 quant,calibration=adapter.quantize_recovered_model(model,task=c['task'],root=root,config=cfg,seed=c['seed'])
 ops=s22.int8_graph_evidence(quant,torch)
 if not ops:raise RuntimeError('No real INT8 operator evidence')
 v1=s22.evaluate(quant,factory,torch)
 b1=s22.cpu_measure(quant,arch.SHAPES[c['task']],torch,warmups=50,runs=100)
 dest.mkdir(parents=True,exist_ok=True)
 exports={}
 for name,obj in (('fp32',model),('int8',quant)):
  dig,n=s22.export(obj,dest/(name+'.pt'),torch,arch.SHAPES[c['task']]);exports[name]=dict(sha256=dig,serialized_bytes=n)
 immutable_json(report,dict(schema='cajo_closure_cpu_v1',candidate=c,candidate_id=c['candidate_id'],
  checkpoint_sha256=sha(ck),history_sha256=sha(hist),source_sha256=expected,
  dataset_manifest_sha256=manifest,calibration=calibration,int8_operator_evidence=ops,
  fp32_validation=v0,int8_validation=v1,cpu_fp32_benchmark=b0,cpu_int8_benchmark=b1,
  exports=exports,test_access='SEALED',hardware='CPU_ONE_THREAD',
  complete_native_memory_bytes=None,cpu_energy_mj_per_inference=None))
 print('CPU COMPLETE',c['candidate_id'],flush=True)

def record(ctx,c,prec):
 root=ctx['root'];dest=root/OUT/'CPU'/c['candidate_id'];p=dest/'result.json'
 if not p.is_file():raise RuntimeError('CPU evidence missing '+c['candidate_id'])
 r=read(p);_,_,ck=paths(ctx,c)
 if r.get('test_access')!='SEALED' or r['checkpoint_sha256']!=sha(ck):raise RuntimeError('CPU provenance mismatch')
 name=prec.lower()
 if sha(dest/(name+'.pt'))!=r['exports'][name]['sha256']:raise RuntimeError('Export hash mismatch')
 return dict(candidate_id=c['candidate_id'],task=c['task'],architecture=c['architecture'],
  pruning_ratio=c['pruning_ratio'],kd=c['kd'],seed=c['seed'],precision=prec,
  accuracy_pct=r[name+'_validation']['accuracy_percent'],
  latency_mean_ms=r['cpu_'+name+'_benchmark']['latency_mean_ms'],
  storage_bytes=r['exports'][name]['serialized_bytes'],export_sha256=r['exports'][name]['sha256'],
  export_path=str((dest/(name+'.pt')).relative_to(root)).replace('\\','/'),
  policy='sequential',checkpoint_sha256=sha(ck))

def ranking(ctx,r):
 t=ctx['cfg']['tasks'][r['task']]
 eligible=(r['accuracy_pct']>=t['accuracy_floor_percent'] and
  r['storage_bytes']<=t['max_storage_mib']*1048576 and
  r['latency_mean_ms']<=t['max_latency_ms'])
 return (0 if eligible else 1,-r['accuracy_pct'],r['latency_mean_ms'],r['storage_bytes'],r['candidate_id'])

def best_precision(ctx,c):
 return min([record(ctx,c,'FP32'),record(ctx,c,'INT8')],key=lambda r:ranking(ctx,r))

def stage_plan(ctx,task,stage):
 """Adaptive policy: each next stage is determined only by completed prior stages."""
 base='baseline';alt=ALT[task]
 if stage==1:return [candidate(ctx,task,a,.25,False,42,5,'arch') for a in (base,alt)]
 arch_rows=[best_precision(ctx,c) for c in stage_plan(ctx,task,1)]
 chosen_arch=min(arch_rows,key=lambda r:ranking(ctx,r))['architecture']
 if stage==2:return [candidate(ctx,task,chosen_arch,.50,False,42,5,'prune')]
 if stage==3:return [candidate(ctx,task,chosen_arch,p,True,42,5,'kd') for p in (.25,.50)]
 if stage==4:
  eligible=[*stage_plan(ctx,task,1),*stage_plan(ctx,task,2),*stage_plan(ctx,task,3)]
  # Replicate the best THREE configurations with seed 43, to match eight
  # five-epoch training jobs (not eight unique configurations).
  scored=sorted(((best_precision(ctx,c),c) for c in eligible),key=lambda pair:ranking(ctx,pair[0]))
  return [candidate(ctx,task,c['architecture'],c['pruning_ratio'],c['kd'],43,5,'replicate') for _,c in scored[:3]]
 raise ValueError('Unknown stage')

def run_screen(ctx):
 for task in TASKS:
  for stage in (1,2,3,4):
   jobs=stage_plan(ctx,task,stage)
   for c in jobs:train(ctx,c);cpu(ctx,c)
   print('STAGE COMPLETE',task,stage,flush=True)

def screening_candidates(ctx,task):
 return [c for stage in (1,2,3,4) for c in stage_plan(ctx,task,stage)]

def finalist(ctx,task,seed):
 screen=screening_candidates(ctx,task)
 # Select from 5 seed-42 configurations; seed43 replicates inform stability
 # but cannot dominate the selection through duplicate representation.
 unique=[c for c in screen if c['seed']==42]
 scored=[best_precision(ctx,c) for c in unique]
 winner=min(scored,key=lambda r:ranking(ctx,r))
 return candidate(ctx,task,winner['architecture'],winner['pruning_ratio'],winner['kd'],seed,20,'final')

def run_final(ctx):
 for task in TASKS:
  for seed in SEEDS:
   c=finalist(ctx,task,seed);train(ctx,c);cpu(ctx,c)

def joint_screen(ctx):
 out=[]
 for r in ctx['screens']:
  out.append(dict(candidate_id=r['candidate_id'],task=r['task'],architecture=r['architecture'],
   pruning_ratio=r['pruning_ratio'],kd=r['kd'],seed=r['seed'],precision=r['precision'],
   accuracy_pct=r['accuracy_pct'],latency_mean_ms=r['latency_mean_ms'],
   storage_bytes=r['storage_bytes'],policy='joint_grid',source_pretraining_matched=False))
 return out

def sequential_screen(ctx):
 out=[]
 for task in TASKS:
  for c in screening_candidates(ctx,task):
   for prec in ('FP32','INT8'):out.append(record(ctx,c,prec))
 return out

def finalist_rows(ctx):
 rows=[]
 for task in TASKS:
  for seed in SEEDS:
   c=finalist(ctx,task,seed)
   for prec in ('FP32','INT8'):rows.append(record(ctx,c,prec))
 # Step24 finalist rows are duplicated by strategy; retain only JOINT labels
 # and avoid counting the identical sequential-label trajectories.
 for r in ctx['final_rows']:
  if r['strategy']!='joint':continue
  rows.append(dict(candidate_id=r['candidate_id'],task=r['task'],architecture=r['architecture'],
   pruning_ratio=r['pruning_ratio'],kd=r['kd'],seed=r['seed'],precision=r['precision'],
   accuracy_pct=r['validation_accuracy_pct'],latency_mean_ms=r['latency_mean_ms'],
   storage_bytes=r['storage_bytes'],policy='joint_grid',
   export_sha256=sha(ctx['root']/'Reports/CAJO_Final_Pipeline/CPU_Final'/r['candidate_id']/(r['precision'].lower()+'.pt')),
   export_path=str(Path('Reports/CAJO_Final_Pipeline/CPU_Final')/r['candidate_id']/(r['precision'].lower()+'.pt')),
   checkpoint_sha256=r['checkpoint_sha256']))
 return rows

METER_COLS=['candidate_id','precision','export_sha256','hardware','memory_bytes','memory_scope',
 'memory_instrument','energy_mj_per_inference','energy_scope','energy_instrument',
 'warmups','timed_inferences','raw_log_path','raw_log_sha256','notes']

def meter_template(ctx):
 rows=finalist_rows(ctx)
 path=ctx['root']/OUT/'external_full_scope_meter.csv'
 if path.exists():raise RuntimeError('Refusing to overwrite meter file '+str(path))
 write_csv(path,[dict(candidate_id=r['candidate_id'],precision=r['precision'],
  export_sha256=r['export_sha256'],hardware='CPU_ONE_THREAD') for r in rows],METER_COLS)
 print('Meter template created:',path)

def meters(ctx,rows):
 root=ctx['root'];p=root/OUT/'external_full_scope_meter.csv'
 if not p.is_file():return {},['No external meter evidence provided']
 lookup={(r['candidate_id'],r['precision']):r for r in rows};out={};errors=[]
 with p.open(newline='',encoding='utf-8-sig') as f:
  rd=csv.DictReader(f)
  if not set(METER_COLS).issubset(rd.fieldnames or []):raise RuntimeError('Meter CSV schema mismatch')
  for line,v in enumerate(rd,2):
   k=(v['candidate_id'].strip(),v['precision'].strip().upper())
   if k not in lookup:errors.append(f'Unknown candidate/precision line {line}');continue
   if not v['memory_bytes'].strip() and not v['energy_mj_per_inference'].strip():continue
   try:
    mem=float(v['memory_bytes']);en=float(v['energy_mj_per_inference'])
    warm=int(v['warmups']);timed=int(v['timed_inferences'])
   except (ValueError,TypeError):errors.append(f'Incomplete numeric fields line {line}');continue
   if not all(math.isfinite(x) and x>0 for x in (mem,en)) or warm<50 or timed<100:
    errors.append(f'Invalid measurement/counts line {line}');continue
   if k in out:errors.append(f'Duplicate meter row line {line}');continue
   if v['export_sha256'].strip().lower()!=lookup[k]['export_sha256']:
    errors.append(f'Model export digest mismatch line {line}');continue
   if v['hardware'].strip()!='CPU_ONE_THREAD':errors.append(f'Hardware mismatch line {line}');continue
   # These explicit scopes are a contract, not automatic certification of a meter.
   if v['memory_scope'].strip()!='TOTAL_DEPLOYMENT_PEAK_INCLUDING_PARAMETERS_ACTIVATIONS_WORKSPACE':
    errors.append(f'Memory scope insufficient line {line}');continue
   if v['energy_scope'].strip() not in ('CPU_PACKAGE_ENERGY','DEVICE_ENERGY'):
    errors.append(f'Energy scope unsupported line {line}');continue
   if not v['memory_instrument'].strip() or not v['energy_instrument'].strip():
    errors.append(f'Instruments unspecified line {line}');continue
   rel=Path(v['raw_log_path'].strip());path=(root/rel).resolve()
   if not rel.as_posix() or not path.is_relative_to(root.resolve()) or not path.is_file():
    errors.append(f'Meter log absent/outside project line {line}');continue
   if sha(path)!=v['raw_log_sha256'].strip().lower():
    errors.append(f'Raw meter log digest mismatch line {line}');continue
   out[k]=dict(memory_bytes=mem,energy_mj_per_inference=en,energy_scope=v['energy_scope'],
    raw_log_sha256=sha(path),instrument_memory=v['memory_instrument'],instrument_energy=v['energy_instrument'])
 return out,errors

def feasibility(ctx,rows,meter):
 out=[]
 for r in rows:
  q=dict(r);t=ctx['cfg']['tasks'][r['task']];m=meter.get((r['candidate_id'],r['precision']))
  q['memory_bytes']=m['memory_bytes'] if m else None
  q['energy_mj_per_inference']=m['energy_mj_per_inference'] if m else None
  q['energy_scope']=m['energy_scope'] if m else None
  q['accuracy_pass']=r['accuracy_pct']>=t['accuracy_floor_percent']
  q['storage_pass']=r['storage_bytes']<=t['max_storage_mib']*1048576
  q['latency_pass']=r['latency_mean_ms']<=t['max_latency_ms']
  q['memory_pass']=m is not None and m['memory_bytes']<=t['max_memory_mib']*1048576
  cap=t.get('max_energy_mj_per_inference',t.get('max_energy_mj'))
  q['energy_cap_defined']=cap is not None
  q['energy_pass']=None if cap is None else (m is not None and m['energy_mj_per_inference']<=float(cap))
  q['five_objectives_measured']=m is not None
  q['fully_feasible']=bool(m and q['accuracy_pass'] and q['storage_pass'] and q['latency_pass'] and q['memory_pass'] and q['energy_pass'] is not False)
  q['feasibility_scope']='measured CPU one-thread; energy objective unless cap locked'
  out.append(q)
 return out

def pareto(rows):
 out=[]
 for a in rows:
  if not a['fully_feasible']:continue
  v=(a['accuracy_pct'],a['storage_bytes'],a['memory_bytes'],a['latency_mean_ms'],a['energy_mj_per_inference'])
  dominated=False
  for b in rows:
   if b is a or b['task']!=a['task'] or not b['fully_feasible']:continue
   w=(b['accuracy_pct'],b['storage_bytes'],b['memory_bytes'],b['latency_mean_ms'],b['energy_mj_per_inference'])
   if w[0]>=v[0] and all(w[i]<=v[i] for i in range(1,5)) and (w[0]>v[0] or any(w[i]<v[i] for i in range(1,5))):
    dominated=True;break
  if not dominated:out.append(a)
 return out

def aggregate(rows):
 groups=defaultdict(list)
 for r in rows:groups[(r['task'],r['policy'],r['architecture'],r['pruning_ratio'],r['kd'],r['precision'])].append(r)
 out=[]
 for k,rs in sorted(groups.items()):
  if len(rs)!=3 or {r['seed'] for r in rs}!={42,43,44}:continue
  z=dict(zip(('task','policy','architecture','pruning_ratio','kd','precision'),k))
  for field in ('accuracy_pct','latency_mean_ms','storage_bytes'):
   values=[float(r[field]) for r in rs]
   z[field+'_mean']=statistics.mean(values);z[field+'_sd']=statistics.stdev(values)
  z['all_three_seeds_full_feasible']=all(r['fully_feasible'] for r in rs)
  out.append(z)
 return out

def finalize(ctx):
 require_approval(ctx)
 root=ctx['root'];joint=joint_screen(ctx);seq=sequential_screen(ctx)
 if len(joint)!=48 or len(seq)!=48:raise RuntimeError('Expected 48 variants per search policy')
 finals=finalist_rows(ctx)
 if len(finals)!=36:raise RuntimeError('Expected 18 unique finalists x 2 precisions')
 meter,errors=meters(ctx,finals)
 if errors and errors!=['No external meter evidence provided']:
  raise RuntimeError('External meter evidence rejected: '+'; '.join(errors[:12]))
 evaluated=feasibility(ctx,finals,meter)
 front=pareto(evaluated);stats=aggregate(evaluated)
 # Prospective policy comparison is independent in search and data selection,
 # but NOT a causal claim about simultaneously trained joint optimization.
 selections=[]
 for task in TASKS:
  for policy,pool in (('joint_grid',joint),('sequential',seq)):
   candidates=[r for r in pool if r['task']==task and (policy!='sequential' or r['seed']==42)]
   winner=min(candidates,key=lambda r:ranking(ctx,r))
   selections.append(dict(task=task,policy=policy,candidate_id=winner['candidate_id'],
    architecture=winner['architecture'],pruning_ratio=winner['pruning_ratio'],kd=winner['kd'],
    precision=winner['precision'],accuracy_pct=winner['accuracy_pct'],
    latency_mean_ms=winner['latency_mean_ms'],storage_bytes=winner['storage_bytes'],
    screening_training_jobs=8,screening_epochs_per_job=5,
    upstream_pretraining_matched=False,causal_joint_training_comparison=False))
 out=root/OUT
 write_csv(out/'joint_screening_verified.csv',joint)
 write_csv(out/'prospective_sequential_screening.csv',seq)
 write_csv(out/'matched_screening_policy_comparison.csv',selections)
 write_csv(out/'finalist_unique_seed_precision.csv',evaluated)
 write_csv(out/'finalist_three_seed_aggregates.csv',stats)
 write_csv(out/'five_objective_pareto_verified.csv',front)
 complete=len(meter)==36
 status=dict(schema='cajo_independent_closure_status_v1',utc=datetime.now(timezone.utc).isoformat(),
  locked_config_sha256=LOCK,original_step23_25_immutable=True,
  joint_screening_training_jobs=24,sequential_screening_training_jobs=24,
  sequential_finalist_training_jobs=9,screening_precision_records_per_policy=48,
  unique_finalist_seed_precision_records=36,externally_metered_records=len(meter),
  five_objective_complete=complete,fully_feasible_records=sum(x['fully_feasible'] for x in evaluated),
  verified_pareto_records=len(front),independent_adaptive_search_comparison=True,
  independent_joint_vs_sequential_training_algorithm=False,
  full_manuscript_claims_supported=False,
  test_access='SEALED',limitations=[
   'Distinct search policies, not distinct simultaneous joint training algorithms',
   'Upstream pretraining not compute-matched across architectures',
   'External full-scope energy and native memory mandatory for five-objective closure',
   'No independently verified test-only evaluation adapter; sealed test not accessed']+errors)
 write_json(out/'status.json',status)
 lines=['# CAJO final reproducible evidence package','',
  '## Methods-to-experiments alignment','',
  'Joint search reuses the eight Step23 configurations per task, five epochs each. Sequential search executes eight NEW five-epoch jobs per task, selected adaptively: architecture probes (2), pruning probe (1), KD probes (2), and replication jobs (3). These are distinct search policies with matched 40 screening epochs per task, **not** independent simultaneously optimized training algorithms. Architecture pretraining histories remain unequal.','',
  'Finalists use three seeds and 20 epochs. CPU FP32/INT8 validation, mean inference latency, and serialized storage are measured. The sealed test has not been used.','',
  '## Prospective search-policy comparison','',
  '| Dataset | Policy | Architecture | Pruning | KD | Precision | Validation accuracy (%) | Mean CPU latency (ms) |',
  '|---|---|---|---:|---|---|---:|---:|']
 for r in selections:
  lines.append(f"| {r['task']} | {r['policy']} | {r['architecture']} | {r['pruning_ratio']:.2f} | {r['kd']} | {r['precision']} | {r['accuracy_pct']:.3f} | {r['latency_mean_ms']:.4f} |")
 lines+=['','## Three-seed finalists','',
  '| Dataset | Policy | Precision | Accuracy mean ± SD (%) | Latency mean (ms) | All seeds fully feasible |',
  '|---|---|---|---:|---:|---|']
 for r in stats:
  lines.append(f"| {r['task']} | {r['policy']} | {r['precision']} | {r['accuracy_pct_mean']:.3f} ± {r['accuracy_pct_sd']:.3f} | {r['latency_mean_ms_mean']:.4f} | {r['all_three_seeds_full_feasible']} |")
 lines+=['','## Five-objective deployment feasibility','',
  f"Verified full-scope energy and memory pairs: **{len(meter)}/36**. Feasible individual seed-precision records: **{status['fully_feasible_records']}**. Verified nondominated records: **{len(front)}**.",
  'Accuracy, serialized model storage, complete deployment peak memory, mean CPU latency, and measured energy are the five objectives. Energy is a Pareto objective; it is a hard constraint only if an energy cap exists in the LOCKED protocol. No meter data are inferred.','',
  '## Claims and limitations','',
  '- The independent prospective adaptive sequential-search policy can be compared to the previously completed exhaustive joint-grid screening at a matched **screening training epoch** budget. It does not establish causal superiority of a simultaneously optimized joint-training algorithm.',
  '- Total pretraining cost is unequal between baseline and alternative architectures; claims of matched total compute are unsupported.',
  '- Complete memory and CPU/device energy require verifiable external instrumentation and raw logs; missing values remain UNKNOWN.',
  '- The test set remains SEALED; results are validation results only. Do not report final generalization performance.',
  '- Full CAJO deployment superiority and Pareto claims remain unsupported until all required evidence and independent test evaluation are obtained.','',
  '## Machine-readable outputs','',
  '`joint_screening_verified.csv`, `prospective_sequential_screening.csv`, `matched_screening_policy_comparison.csv`, `finalist_unique_seed_precision.csv`, `finalist_three_seed_aggregates.csv`, `five_objective_pareto_verified.csv`, `status.json`.']
 (out/'manuscript_evidence.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(status,indent=2))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',choices=['inspect','authorize','run','meter-template','finalize'])
 p.add_argument('--root',type=Path,default=Path('..'))
 p.add_argument('--confirm')
 a=p.parse_args();root=a.root.resolve();ctx=context(root)
 if a.command=='inspect':
  print(json.dumps(dict(config_sha256=LOCK,existing_step23_screening_variants=len(ctx['screens']),
   existing_step24_precision_records=len(ctx['final_rows']),new_output=str(root/OUT),
   planned_sequential_screening_jobs=24,planned_sequential_finalist_jobs=9,
   old_artifacts_read_only=True,test_access='SEALED',
   energy_and_complete_memory='external instrumentation required',
   claim_scope='prospective search-policy comparison, not end-to-end joint training'),indent=2));return
 if a.command=='authorize':
  if a.confirm!='APPROVE_NEW_SEQUENTIAL_CAMPAIGN':raise RuntimeError('Explicit approval required')
  immutable_json(root/OUT/'authorization.json',approval_expected(ctx))
  print('AUTHORIZED new-folder-only adaptive sequential campaign');return
 require_approval(ctx)
 if a.command=='run':run_screen(ctx);run_final(ctx);print('NEW CAMPAIGN COMPLETED; run meter-template then finalize')
 elif a.command=='meter-template':meter_template(ctx)
 else:finalize(ctx)

if __name__=='__main__':main()
