#!/usr/bin/env python3
"""CAJO Step 23: deterministic, resumable, validation-only, five-epoch search.

Searches 2 architectures x 2 pruning ratios x 2 KD settings = 8 FP32
screening trainings per task, initialized from VERIFIED pretrained sources.
Each screened checkpoint is measured in both CPU FP32 and calibrated CPU INT8
on the SAME CPU, via the verified Step22 implementation. This is a genuine
bounded, matched-training-budget search over (architecture, pruning, KD,
precision); precision is a post-training deployment choice.

The initial source pretraining is NOT matched across architectures, and this
is NOT an isolated/sequential versus joint causal comparison. No final 20-
epoch finalist retraining, complete memory, CPU energy, Pareto feasibility,
or test evaluation is claimed. Such steps require further evidence.

Commands from Codes/:
 python 23_cajo_matched_joint_search.py inspect --root ..
 python 23_cajo_matched_joint_search.py authorize --root .. --confirm APPROVE_STEP23_SCREENING
 python 23_cajo_matched_joint_search.py run-one --root .. --candidate-id ID
 python 23_cajo_matched_joint_search.py run-all --root ..
 python 23_cajo_matched_joint_search.py summarize --root ..
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,os,platform,statistics,sys
from pathlib import Path

LOCKED='56361ba3ef2cb530d54a923cf4f5b3758a22bee82a9119822395cef0be9bbdc5'
CONFIG=Path('Reports/Optimization_Protocol/Final_Experiment_Config/cajo_experiment_config.json')
OUT=Path('Reports/CAJO_Matched_Search')
TRAIN=OUT/'Training'
SOURCES=('02_experiment_runner.py','03_architecture_search_space.py','04_compression_recovery.py',
         '16_cajo_gpu_hardware_benchmark.py','18_cajo_final_joint_training.py',
         '20_cajo_alternative_fp32_benchmark.py','22_cajo_int8_experiments.py',
         '23_cajo_matched_joint_search.py')
ALT={'CIFAR10':'ResNet8_width075','CIFAR100':'ResNet18_width075','SpeechCommands_v0.02':'DSCNN_width075'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p)
 if not spec or not spec.loader:raise RuntimeError('Cannot load '+str(p))
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def once(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if read(p)!=obj:raise RuntimeError('Immutable artifact differs: '+str(p))
  return
 tmp=p.with_name(p.name+'.partial')
 with tmp.open('x',encoding='utf-8') as f:
  json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)

def plan(root):
 if sha(root/CONFIG)!=LOCKED:raise RuntimeError('Locked configuration SHA256 mismatch')
 cfg=read(root/CONFIG)
 if cfg['status']!='LOCKED' or cfg['selection']['test_access']!='sealed_until_selection_frozen':
  raise RuntimeError('Protocol/test firewall invalid')
 for s in SOURCES:
  if not (root/'Codes'/s).is_file():raise FileNotFoundError(root/'Codes'/s)
 # Verify that the completed INT8 campaign exists and remains immutable.
 s22=module(root/'Codes'/SOURCES[6],'cajo23_s22')
 _,items,s16,s20=s22.inventory(root)
 s22.verify_auth(root,items)
 for item in items:
  r=root/'Reports'/'CAJO_INT8_CPU'/item['candidate']['candidate_id']/'result.json'
  if not r.is_file():raise RuntimeError('Step22 result missing: '+str(r))
  d=read(r)
  if d.get('test_access')!='SEALED' or d.get('source_checkpoint_sha256')!=sha(item['checkpoint']):
   raise RuntimeError('Step22 provenance drift: '+str(r))
  for name in ('fp32','int8'):
   e=root/'Reports'/'CAJO_INT8_CPU'/item['candidate']['candidate_id']/(name+'.pt')
   if not e.is_file() or sha(e)!=d['exports'][name]['sha256']:
    raise RuntimeError('Step22 export drift: '+str(e))
 index={(i['candidate']['task'],i['candidate']['architecture'],i['candidate']['seed'],
         float(i['candidate']['pruning_ratio']),bool(i['candidate']['kd'])):i for i in items}
 runner=module(root/'Codes'/SOURCES[0],'cajo23_runner')
 candidates=[]
 for task in sorted(ALT):
  for architecture in ('baseline',ALT[task]):
   for pruning in (0.25,0.5):
    for kd in (False,True):
     seed=42
     # Source is frozen baseline or independently pretrained alternative P0.
     if architecture=='baseline':
      source=root/Path(cfg['tasks'][task]['frozen_checkpoint'].replace('\\','/'))
      expected=cfg['tasks'][task]['checkpoint_sha256']
     else:
      cid0=f'{task}__{architecture}__pretrain_P0_CE__s{seed}'
      src=index.get((task,architecture,seed,0.0,False))
      if src is None:raise RuntimeError('Alternative P0 source not found: '+cid0)
      source=src['checkpoint'];expected=sha(source)
     if not source.is_file() or sha(source)!=expected:raise RuntimeError('Source checkpoint drift: '+str(source))
     cid=f'{task}__'+('base' if architecture=='baseline' else 'alt')+f'__P{int(pruning*100)}__'+('KD' if kd else 'CE')+'__screen5__s42'
     c=dict(candidate_id=cid,task=task,architecture=architecture,pruning_ratio=pruning,
       precision='FP32',kd=kd,strategy='screening',seed=seed,epochs=5,
       mode='recovery' if architecture=='baseline' else 'architecture_training',
       operation_order=['verified_pretrained_source','physical_importance_pruning',
                        'KD_recovery' if kd else 'CE_recovery'],pruning_policy='importance')
     runner.candidate_contract(c,cfg)
     candidates.append(dict(candidate=c,source_checkpoint=str(source.relative_to(root)),
                            source_sha256=expected,initialization='frozen_baseline_copy' if architecture=='baseline' else 'pretrained_alternative_P0'))
 if len(candidates)!=24:raise RuntimeError('Expected exactly 8 screening configurations per task')
 return cfg,candidates,runner

def authorization(root,candidates):
 return dict(schema='cajo_step23_screening_authorization_v1',config_sha256=LOCKED,
  source_sha256={'Codes/'+s:sha(root/'Codes'/s) for s in SOURCES},
  candidates=candidates,scope='VALIDATION_ONLY_5_EPOCH_SCREENING_2ARCH_2PRUNE_2KD_AND_CPU_FP32_INT8_PTQ',
  training_budget='5 epochs per configuration; seed 42; same SGD/augmentation/scheduler policy',
  precision_policy='CPU FP32 and post-training INT8 from each screened FP32 checkpoint',
  no_test_access=True,no_baseline_retraining=True,no_final_feasibility_claim=True,
  limitations=['Only one screening seed; finalists require 3-seed 20-epoch recovery',
   'Pretraining effort is not matched between frozen baseline and alternative P0',
   'No matched sequential-versus-joint causal control',
   'CPU energy and complete inference memory not measured',
   'No Pareto selection, final test, or full-deployment feasibility claim'])

def verify_auth(root,candidates):
 p=root/OUT/'authorization.json'
 if not p.is_file() or read(p)!=authorization(root,candidates):
  raise RuntimeError('Missing/drifted Step23 authorization. Inspect and authorize this exact scope.')

def paths(root,c):
 d=root/TRAIN/c['task']/'screening'/c['candidate_id']/('seed_'+str(c['seed']))
 return d,d/'history.json',d/'best_model.pt',root/OUT/'CPU_FP32_INT8'/c['candidate_id']

def check_history(root,c):
 _,hist,checkpoint,_=paths(root,c)
 if not hist.is_file():return False
 h=read(hist)
 if h.get('candidate')!=c or h.get('test_access')!='SEALED':raise RuntimeError('History provenance mismatch')
 if h.get('status')!='COMPLETE':return False
 if len(h.get('history',[]))!=5 or not checkpoint.is_file() or sha(checkpoint)!=h.get('best_checkpoint_sha256'):
  raise RuntimeError('Completed screening checkpoint/history invalid: '+c['candidate_id'])
 return True

def screen(root,cfg,entry,runner,candidates):
 c=entry['candidate'];verify_auth(root,candidates)
 if check_history(root,c):print('SKIP TRAIN VERIFIED',c['candidate_id'],flush=True);return
 dest,hist,checkpoint,_=paths(root,c)
 if hist.exists() and not (dest/'resume_state.pt').is_file():
  raise RuntimeError('Partial run lacks resume_state: '+str(dest))
 # Existing runner is reused without modifying its source, training implementation,
 # optimizer, checkpoint logic, or data loaders. Scoped override redirects ONLY outputs.
 adapter=module(root/'Codes'/'04_compression_recovery.py','cajo23_adapter')
 old_gate,old_import,old_out=runner.run_gate,runner.import_adapter,runner.OUT_REL
 def gate(d):
  if d is not cfg:raise RuntimeError('Config object changed')
  verify_auth(root,candidates)
  unexpected=[x for x in old_gate(d) if x not in ('Manuscript energy alignment pending','Full training not authorized')]
  if unexpected:raise RuntimeError('Runner gate: '+'; '.join(unexpected))
  return []
 def prepared(*,root,config,candidate,device):
  import torch
  arch,data,surgery,importance=adapter.sources(root)
  splits,manifest=adapter.verified_manifest(root,config,candidate['task'],data)
  teacher,baseline=adapter.frozen_teacher(root,config,candidate['task'],arch,torch)
  model=arch.build_architecture(candidate['task'],candidate['architecture'],config,root)
  source=root/entry['source_checkpoint']
  if sha(source)!=entry['source_sha256']:raise RuntimeError('Source checkpoint changed')
  if candidate['architecture']=='baseline':model.load_state_dict(teacher.state_dict(),strict=True)
  else:
   blob=torch.load(source,map_location='cpu',weights_only=True)
   if blob.get('candidate',{}).get('candidate_id')!=f"{candidate['task']}__{candidate['architecture']}__pretrain_P0_CE__s42":
    raise RuntimeError('Alternative source candidate mismatch')
   model.load_state_dict(blob['model'],strict=True)
  details=adapter.physical_prune(model,candidate['task'],candidate['pruning_ratio'],importance,surgery,
                                  arch,'importance',candidate['seed'],root)
  train,val=adapter.make_data(root,candidate['task'],splits,config,candidate['seed'],data)
  provenance=dict(test_access='sealed',frozen_manifest_sha256=sha(manifest),
   frozen_teacher_checkpoint_sha256=sha(baseline),architecture=candidate['architecture'],
   operation_order=candidate['operation_order'],initialization=entry['initialization'],
   pretrained_checkpoint_sha256=entry['source_sha256'],pruning_details=details,
   student_parameters=sum(p.numel() for p in model.parameters()),precision_training='FP32')
  return dict(student=model,teacher=teacher if candidate['kd'] else None,
              train_epoch_batches=train,validation_batches=val,provenance=provenance)
 class Proxy:prepare_candidate=staticmethod(prepared)
 runner.run_gate=gate;runner.import_adapter=lambda _root:(Proxy,root/'Codes'/'04_compression_recovery.py')
 runner.OUT_REL=TRAIN
 try:
  contract=root/OUT/'Contracts'/(c['candidate_id']+'.json')
  once(contract,c)
  print('SCREEN',c['candidate_id'],'RESUME' if (dest/'resume_state.pt').exists() else 'NEW',flush=True)
  runner.fit(root,cfg,root/CONFIG,c,contract,resume=(dest/'resume_state.pt').exists())
 finally:
  runner.run_gate=old_gate;runner.import_adapter=old_import;runner.OUT_REL=old_out

def cpu_evaluate(root,cfg,entry,candidates):
 import torch
 torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 c=entry['candidate'];verify_auth(root,candidates)
 if not check_history(root,c):raise RuntimeError('Screen training not complete')
 _,hist,checkpoint,dest=paths(root,c)
 report=dest/'result.json'
 if report.exists():
  r=read(report)
  if r.get('checkpoint_sha256')!=sha(checkpoint) or r.get('test_access')!='SEALED':
   raise RuntimeError('CPU result provenance mismatch')
  for label in ('fp32','int8'):
   p=dest/(label+'.pt')
   if not p.is_file() or sha(p)!=r['exports'][label]['sha256']:
    raise RuntimeError('CPU exported model mismatch: '+str(p))
  print('SKIP CPU VERIFIED',c['candidate_id'],flush=True);return
 if dest.exists() and any(dest.iterdir()):
  raise RuntimeError('Partial CPU output; inspect manually before rerunning: '+str(dest))
 s22=module(root/'Codes'/'22_cajo_int8_experiments.py','cajo23_s22_cpu')
 adapter=module(root/'Codes'/'04_compression_recovery.py','cajo23_adapter_cpu')
 arch=module(root/'Codes'/'03_architecture_search_space.py','cajo23_arch_cpu')
 # Reconstruct physically pruned model using the SAME source and surgery.
 _,data,surgery,importance=adapter.sources(root)
 model=arch.build_architecture(c['task'],c['architecture'],cfg,root)
 if c['architecture']=='baseline':
  teacher,_=adapter.frozen_teacher(root,cfg,c['task'],arch,torch)
  model.load_state_dict(teacher.state_dict(),strict=True)
 else:
  src=torch.load(root/entry['source_checkpoint'],map_location='cpu',weights_only=True)
  model.load_state_dict(src['model'],strict=True)
 adapter.physical_prune(model,c['task'],c['pruning_ratio'],importance,surgery,arch,'importance',c['seed'],root)
 blob=torch.load(checkpoint,map_location='cpu',weights_only=True)
 if blob.get('candidate')!=c:raise RuntimeError('Screen checkpoint candidate mismatch')
 model.load_state_dict(blob['model'],strict=True)
 model.cpu().eval()
 factory,manifest=s22.validation_factory(root,cfg,c,adapter)
 fp32_val=s22.evaluate(model,factory,torch)
 fp32_bench=s22.cpu_measure(model,arch.SHAPES[c['task']],torch)
 qmodel,qprov=adapter.quantize_recovered_model(model,task=c['task'],root=root,config=cfg,seed=c['seed'])
 ops=s22.int8_graph_evidence(qmodel,torch)
 if not ops:raise RuntimeError('Quantized model has no verified quantized ops')
 int8_val=s22.evaluate(qmodel,factory,torch)
 int8_bench=s22.cpu_measure(qmodel,arch.SHAPES[c['task']],torch)
 dest.mkdir(parents=True,exist_ok=True)
 exports={}
 for label,m in [('fp32',model),('int8',qmodel)]:
  digest,nbytes=s22.export(m,dest/(label+'.pt'),torch,arch.SHAPES[c['task']])
  exports[label]=dict(sha256=digest,serialized_bytes=nbytes)
 r=dict(schema='cajo_step23_screen_cpu_v1',candidate_id=c['candidate_id'],candidate=c,
  config_sha256=LOCKED,authorization_sha256=sha(root/OUT/'authorization.json'),
  source_checkpoint_sha256=entry['source_sha256'],checkpoint_sha256=sha(checkpoint),
  history_sha256=sha(hist),dataset_manifest_sha256=manifest,calibration=qprov,
  fp32_validation=fp32_val,int8_validation=int8_val,cpu_fp32_benchmark=fp32_bench,
  cpu_int8_benchmark=int8_bench,int8_operator_evidence=ops,exports=exports,
  test_access='SEALED',test_accessed=False,full_feasibility=False,
  matched_screening_epochs=5,screening_seed=42,
  environment=dict(python=sys.version.split()[0],torch=torch.__version__,cpu=platform.processor(),threads=torch.get_num_threads()))
 once(report,r)
 print('CPU COMPLETE',c['candidate_id'],'FP32',round(fp32_val['accuracy_percent'],3),
       'INT8',round(int8_val['accuracy_percent'],3),flush=True)

def summarize(root,candidates):
 rows=[];issues=[]
 for entry in candidates:
  c=entry['candidate'];_,_,checkpoint,dest=paths(root,c)
  p=dest/'result.json'
  if not p.is_file():continue
  try:
   if not check_history(root,c):raise RuntimeError('Training incomplete')
   r=read(p)
   if r['checkpoint_sha256']!=sha(checkpoint) or r['test_access']!='SEALED':raise RuntimeError('Provenance drift')
   for precision in ('fp32','int8'):
    e=dest/(precision+'.pt')
    if not e.is_file() or sha(e)!=r['exports'][precision]['sha256']:raise RuntimeError('Export digest mismatch')
    rows.append(dict(candidate_id=c['candidate_id'],task=c['task'],architecture=c['architecture'],
      pruning_ratio=c['pruning_ratio'],kd=c['kd'],seed=c['seed'],precision=precision.upper(),
      validation_accuracy_pct=r[precision+'_validation']['accuracy_percent'],
      cpu_latency_mean_ms=r['cpu_'+precision+'_benchmark']['latency_mean_ms'],
      cpu_latency_median_ms=r['cpu_'+precision+'_benchmark']['latency_median_ms'],
      serialized_bytes=r['exports'][precision]['serialized_bytes'],
      training_epochs=5,test_access='SEALED'))
  except Exception as exc:issues.append(c['candidate_id']+': '+str(exc))
 out=root/OUT;out.mkdir(parents=True,exist_ok=True)
 with (out/'screening_evidence.csv').open('w',newline='',encoding='utf-8') as f:
  cols=['candidate_id','task','architecture','pruning_ratio','kd','seed','precision','validation_accuracy_pct',
        'cpu_latency_mean_ms','cpu_latency_median_ms','serialized_bytes','training_epochs','test_access']
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
 result=dict(schema='cajo_step23_screening_summary_v1',expected_fp32_screenings=len(candidates),
  completed_fp32_screenings=len(rows)//2,expected_precision_variants=2*len(candidates),
  completed_precision_variants=len(rows),issues=issues,screening_seed=42,
  selection_frozen=False,finalists_trained=False,full_feasibility=False,
  sequential_vs_joint_matched_comparator=False,test_access='SEALED')
 (out/'screening_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result,indent=2))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('action',choices=['inspect','authorize','run-one','run-all','summarize'])
 p.add_argument('--root',type=Path,default=Path('..'))
 p.add_argument('--candidate-id');p.add_argument('--confirm')
 a=p.parse_args();root=a.root.resolve()
 cfg,candidates,runner=plan(root)
 if a.action=='inspect':
  print(json.dumps(dict(locked_config_sha256=LOCKED,planned_screenings=len(candidates),
   tasks=sorted(ALT),per_task=8,screening_seed=42,epochs_per_candidate=5,
   precision_variants_per_task=16,existing_fp32_and_int8_reused=True,
   test_access='SEALED',authorization_required=True,
   candidates=[e['candidate']['candidate_id'] for e in candidates],
   limitations=authorization(root,candidates)['limitations']),indent=2));return
 if a.action=='authorize':
  if a.confirm!='APPROVE_STEP23_SCREENING':raise RuntimeError('Explicit --confirm APPROVE_STEP23_SCREENING required')
  once(root/OUT/'authorization.json',authorization(root,candidates))
  print('AUTHORIZED scoped Step23 screening. No training has started.');return
 if a.action=='summarize':summarize(root,candidates);return
 verify_auth(root,candidates)
 selected=candidates if a.action=='run-all' else [e for e in candidates if e['candidate']['candidate_id']==a.candidate_id]
 if not selected:raise ValueError('Unknown candidate ID; use inspect to list IDs')
 for entry in selected:
  screen(root,cfg,entry,runner,candidates)
  cpu_evaluate(root,cfg,entry,candidates)
 summarize(root,candidates)

if __name__=='__main__':main()
