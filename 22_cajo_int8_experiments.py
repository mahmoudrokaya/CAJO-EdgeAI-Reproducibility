#!/usr/bin/env python3
"""Step 22: real, resumable, train-calibrated oneDNN INT8 PTQ and paired CPU FP32.

Run from Codes:
 python 22_cajo_int8_experiments.py inspect --root ..
 python 22_cajo_int8_experiments.py authorize --root ..
 python 22_cajo_int8_experiments.py run-one --root .. --candidate-id ID
 python 22_cajo_int8_experiments.py run-all --root ..
 python 22_cajo_int8_experiments.py summarize --root ..

Uses verified existing FP32 weights and Step 04's train-only 256-example
calibration. Evaluates validation only. Never accesses test, trains, changes
locked config, overwrites existing results, or claims full feasibility.

Authorization is separate from the locked FP32 training authorization. The
first run must be a reviewed, explicitly authorized run. All outputs are
immutable, per-candidate, and SHA-bound to inputs and source code.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,platform,statistics,sys,time
from pathlib import Path

LOCKED='56361ba3ef2cb530d54a923cf4f5b3758a22bee82a9119822395cef0be9bbdc5'
CONFIG=Path('Reports/Optimization_Protocol/Final_Experiment_Config/cajo_experiment_config.json')
OUT=Path('Reports/CAJO_INT8_CPU')
SOURCES=('02_experiment_runner.py','03_architecture_search_space.py','04_compression_recovery.py',
         '16_cajo_gpu_hardware_benchmark.py','18_cajo_final_joint_training.py',
         '20_cajo_alternative_fp32_benchmark.py','22_cajo_int8_experiments.py')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p)
 if not s or not s.loader:raise RuntimeError('Cannot import '+str(p))
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def write_once(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError('Immutable output already exists: '+str(p))
 tmp=p.with_name(p.name+'.partial')
 with tmp.open('x',encoding='utf-8') as f:
  json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)

def inventory(root):
 if sha(root/CONFIG)!=LOCKED:raise RuntimeError('Locked config changed')
 cfg=read(root/CONFIG)
 if cfg['status']!='LOCKED' or cfg['selection']['test_access']!='sealed_until_selection_frozen':
  raise RuntimeError('Locked/test firewall invalid')
 for s in SOURCES:
  if not (root/'Codes'/s).is_file():raise FileNotFoundError(root/'Codes'/s)
 s16=module(root/'Codes'/SOURCES[3],'cajo22_s16')
 s20=module(root/'Codes'/SOURCES[5],'cajo22_s20')
 _,old=s16.inventory(root)
 _,new=s20.discover(root)
 items=[]
 for cohort,group in [('baseline_fp32',old),('alternative_fp32',new)]:
  for item in group:
   p,c,h,b=item
   items.append(dict(cohort=cohort,contract=p,candidate=c,history=h,checkpoint=b))
 if len(items)!=72 or len({i['candidate']['candidate_id'] for i in items})!=72:
  raise RuntimeError('Expected 72 unique existing FP32 candidates')
 return cfg,items,s16,s20

def auth_expected(root,items):
 return dict(schema='cajo_int8_authorization_v1',config_sha256=LOCKED,
  source_sha256={'Codes/'+s:sha(root/'Codes'/s) for s in SOURCES},
  candidate_ids=sorted(i['candidate']['candidate_id'] for i in items),
  scope='VALIDATION_ONLY_CPU_FP32_INT8_PTQ_EXISTING_72_CHECKPOINTS',
  calibration='256 frozen TRAIN samples per task; no validation/test calibration',
  no_training=True,no_test_access=True,no_baseline_checkpoint_mutation=True,
  no_complete_deployment_feasibility_claim=True)

def verify_auth(root,items):
 p=root/OUT/'authorization.json'
 if not p.is_file() or read(p)!=auth_expected(root,items):
  raise RuntimeError('Step22 authorization missing or source/config/candidate drift. Review and run authorize first.')

def verify_item(root,cfg,item,s16,s20):
 t=(item['contract'],item['candidate'],item['history'],item['checkpoint'])
 if item['cohort']=='baseline_fp32':h,d=s16.validate_item(root,cfg,t)
 else:h,d=s20.validate(root,t,True)
 if d!=sha(item['checkpoint']):raise RuntimeError('Checkpoint changed during verification')
 return h,d

def reconstruct(root,cfg,item,s16,s20,torch):
 c=item['candidate']
 if item['cohort']=='baseline_fp32':model,_,_=s16.build_model(root,cfg,c,torch)
 else:model,_,_=s20.reconstruct(root,cfg,c,torch)
 blob=torch.load(item['checkpoint'],map_location='cpu',weights_only=True)
 if blob.get('candidate')!=c:raise RuntimeError('Checkpoint candidate contract mismatch')
 model.load_state_dict(blob['model'],strict=True)
 return model.cpu().eval()

def validation_factory(root,cfg,c,adapter):
 arch,data,_,_=adapter.sources(root)
 splits,manifest=adapter.verified_manifest(root,cfg,c['task'],data)
 _,factory=adapter.make_data(root,c['task'],splits,cfg,int(c['seed']),data)
 return factory,sha(manifest)

def evaluate(model,factory,torch):
 correct=total=0
 model.eval()
 with torch.inference_mode():
  for batch in factory():
   x,y=batch[:2];logits=model(x.cpu())
   if logits.ndim!=2 or logits.shape[0]!=y.shape[0] or not bool(torch.isfinite(logits).all()):
    raise RuntimeError('Invalid/nonfinite validation output')
   correct+=int((logits.argmax(1)==y.cpu()).sum().item());total+=int(y.numel())
 if total==0:raise RuntimeError('Empty validation partition')
 return dict(correct=correct,total=total,accuracy_percent=100*correct/total)

def cpu_measure(model,shape,torch,warmups=50,runs=100):
 x=torch.zeros(shape,dtype=torch.float32)
 with torch.inference_mode():
  y=model(x)
  if not bool(torch.isfinite(y).all()):raise RuntimeError('Nonfinite CPU inference')
  for _ in range(warmups):model(x)
  samples=[]
  for _ in range(runs):
   t=time.perf_counter_ns();model(x);samples.append((time.perf_counter_ns()-t)/1e6)
 return dict(warmups=warmups,timed_runs=runs,latency_mean_ms=statistics.mean(samples),
             latency_median_ms=statistics.median(samples),latency_p95_ms=sorted(samples)[94],
             latency_samples_ms=samples,scope='CPU wall-clock model forward, batch=1, single thread')

def int8_graph_evidence(model,torch):
 # Presence of actual quantized kernels, not just observers or a declared label.
 graph=str(model.graph) if hasattr(model,'graph') else str(model)
 ops=[]
 for needle in ('quantized::','quantized_decomposed::','onednn::'):
  if needle in graph:ops.append(needle)
 for m in model.modules():
  name=type(m).__module__+'.'+type(m).__name__
  if 'quantized' in name.lower() or 'quantize' in name.lower():ops.append(name)
 return sorted(set(ops))

def export(model,path,torch,shape):
 # Trace an already-quantized CPU graph; avoid arbitrary external TorchScript.
 x=torch.zeros(shape)
 with torch.inference_mode():
  traced=torch.jit.trace(model,x,strict=False,check_trace=False)
  ref=model(x);got=traced(x)
  if not bool(torch.allclose(ref,got,atol=1e-3,rtol=1e-3)):
   raise RuntimeError('TorchScript output differs from eager model')
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.partial')
 if tmp.exists() or path.exists():raise FileExistsError('Export exists/partial: '+str(path))
 try:
  torch.jit.save(traced,str(tmp))
  check=torch.jit.load(str(tmp),map_location='cpu').eval()
  with torch.inference_mode():
   if not bool(torch.allclose(ref,check(x),atol=1e-3,rtol=1e-3)):
    raise RuntimeError('Reloaded TorchScript mismatch')
  os.replace(tmp,path)
 finally:
  if tmp.exists():tmp.unlink()
 return sha(path),path.stat().st_size

def run_one(root,cfg,items,s16,s20,item):
 import torch
 torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 torch.backends.mkldnn.enabled=True
 verify_auth(root,items)
 c=item['candidate'];cid=c['candidate_id'];dest=root/OUT/cid
 report=dest/'result.json'
 if report.exists():
  r=read(report)
  if r.get('schema')!='cajo_int8_cpu_v1' or r.get('source_checkpoint_sha256')!=sha(item['checkpoint']):
   raise RuntimeError('Existing report invalid; do not overwrite: '+cid)
  for key in ('fp32','int8'):
   p=dest/(key+'.pt')
   if not p.exists() or sha(p)!=r['exports'][key]['sha256']:
    raise RuntimeError('Existing export invalid: '+str(p))
  print('SKIP VERIFIED:',cid,flush=True);return
 if dest.exists() and any(dest.iterdir()):
  raise RuntimeError('Partial candidate output; inspect manually before resuming: '+str(dest))
 h,ck=verify_item(root,cfg,item,s16,s20)
 adapter=module(root/'Codes'/'04_compression_recovery.py','cajo22_adapter')
 arch=module(root/'Codes'/'03_architecture_search_space.py','cajo22_arch')
 fp32=reconstruct(root,cfg,item,s16,s20,torch)
 factory,manifest_sha=validation_factory(root,cfg,c,adapter)
 val_fp32=evaluate(fp32,factory,torch)
 fp32_bench=cpu_measure(fp32,arch.SHAPES[c['task']],torch)
 quant,provenance=adapter.quantize_recovered_model(fp32,task=c['task'],root=root,config=cfg,seed=int(c['seed']))
 qops=int8_graph_evidence(quant,torch)
 if not qops:raise RuntimeError('No independently observable INT8 operator/module evidence; refuse INT8 label')
 val_int8=evaluate(quant,factory,torch)
 int8_bench=cpu_measure(quant,arch.SHAPES[c['task']],torch)
 dest.mkdir(parents=True,exist_ok=True)
 exports={}
 for label,model in [('fp32',fp32),('int8',quant)]:
  digest,nbytes=export(model,dest/(label+'.pt'),torch,arch.SHAPES[c['task']])
  exports[label]=dict(path=str(dest/(label+'.pt')),sha256=digest,serialized_bytes=nbytes)
 result=dict(schema='cajo_int8_cpu_v1',candidate_id=cid,task=c['task'],architecture=c['architecture'],
  seed=c['seed'],pruning_ratio=c['pruning_ratio'],kd=c['kd'],cohort=item['cohort'],
  config_sha256=LOCKED,source_checkpoint_sha256=ck,source_history_sha256=sha(item['history']),
  source_contract_sha256=sha(item['contract']),authorization_sha256=sha(root/OUT/'authorization.json'),
  dataset_manifest_sha256=manifest_sha,calibration=provenance,
  fp32_validation=val_fp32,int8_validation=val_int8,
  cpu_fp32_benchmark=fp32_bench,cpu_int8_benchmark=int8_bench,
  int8_operator_evidence=qops,exports=exports,
  environment=dict(python=sys.version.split()[0],torch=torch.__version__,cpu=platform.processor(),
                   quantized_engine=torch.backends.quantized.engine,threads=torch.get_num_threads()),
  energy_measured=False,complete_runtime_memory_measured=False,full_feasibility=False,
  matched_budget_search=False,test_access='SEALED',test_accessed=False)
 write_once(report,result)
 print('COMPLETE:',cid,'FP32 val',round(val_fp32['accuracy_percent'],3),
       'INT8 val',round(val_int8['accuracy_percent'],3),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('action',choices=['inspect','authorize','run-one','run-all','summarize'])
 p.add_argument('--root',type=Path,default=Path('..'))
 p.add_argument('--candidate-id')
 a=p.parse_args();root=a.root.resolve()
 cfg,items,s16,s20=inventory(root)
 if a.action=='inspect':
  print(json.dumps(dict(locked_config_sha256=LOCKED,existing_fp32=len(items),
   baseline=sum(i['cohort']=='baseline_fp32' for i in items),
   alternative=sum(i['cohort']=='alternative_fp32' for i in items),
   planned_int8=len(items),calibration='256 train-only examples, oneDNN',
   CPU_FP32_comparator='same candidate/same CPU/single thread',test_access='SEALED',
   authorization_required=True,quantization_executed=False),indent=2));return
 if a.action=='authorize':
  write_once(root/OUT/'authorization.json',auth_expected(root,items))
  print('AUTHORIZED Step22 scope only. No model or dataset accessed.');return
 if a.action=='summarize':
  records=[]
  for item in items:
   cid=item['candidate']['candidate_id'];p=root/OUT/cid/'result.json'
   if p.exists():
    r=read(p)
    records.append(dict(candidate_id=cid,task=r['task'],seed=r['seed'],
      fp32_accuracy=r['fp32_validation']['accuracy_percent'],
      int8_accuracy=r['int8_validation']['accuracy_percent'],
      fp32_latency_ms=r['cpu_fp32_benchmark']['latency_mean_ms'],
      int8_latency_ms=r['cpu_int8_benchmark']['latency_mean_ms'],
      fp32_storage_bytes=r['exports']['fp32']['serialized_bytes'],
      int8_storage_bytes=r['exports']['int8']['serialized_bytes']))
  print(json.dumps(dict(expected=len(items),complete=len(records),missing=len(items)-len(records),
                        records=records),indent=2));return
 verify_auth(root,items)
 if a.action=='run-one':
  selected=[i for i in items if i['candidate']['candidate_id']==a.candidate_id]
  if len(selected)!=1:raise RuntimeError('Specify a valid --candidate-id from Step21 register')
 else:selected=items
 for i,item in enumerate(selected,1):
  print(f'[{i}/{len(selected)}] {item["candidate"]["candidate_id"]}',flush=True)
  run_one(root,cfg,items,s16,s20,item)

if __name__=='__main__':
 try:main()
 except Exception as e:
  print('STEP22 FAILED:',type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(2)
