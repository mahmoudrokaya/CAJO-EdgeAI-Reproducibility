#!/usr/bin/env python3
"""CAJO Step 20: benchmark the 45 existing alternative FP32 checkpoints.

Read-only with respect to training/configuration/checkpoints and test data.
Uses Step 16's exact GPU timing, allocator, NVML measurement engine; adds
mean latency and raw GPU energy per inference. Outputs immutable JSON in
Reports/CAJO_Alternative_GPU_Benchmark. Does NOT assert complete deployment
memory/energy feasibility or that this is matched-budget joint NAS.

From Codes:
  python 20_cajo_alternative_fp32_benchmark.py inspect --root ..
  python 20_cajo_alternative_fp32_benchmark.py run-one --root .. --candidate-id ID
  python 20_cajo_alternative_fp32_benchmark.py run-all --root ..
  python 20_cajo_alternative_fp32_benchmark.py summarize --root ..

Requires existing 16_cajo_gpu_hardware_benchmark.py, 18_cajo_final_joint_training.py,
02_experiment_runner.py, 03_architecture_search_space.py, and
04_compression_recovery.py in Codes. Trusted locally generated checkpoints only.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,os,statistics,sys
from pathlib import Path

LOCKED='56361ba3ef2cb530d54a923cf4f5b3758a22bee82a9119822395cef0be9bbdc5'
CONFIG=Path('Reports/Optimization_Protocol/Final_Experiment_Config/cajo_experiment_config.json')
EXPERIMENTS=Path('Reports/CAJO_Experiments')
CANDIDATES=Path('Reports/Optimization_Protocol/Alternative_FP32_Training/candidates')
AUTH=Path('Reports/Optimization_Protocol/Alternative_FP32_Training/authorization.json')
OUT=Path('Reports/CAJO_Alternative_GPU_Benchmark')
SOURCES=('16_cajo_gpu_hardware_benchmark.py','18_cajo_final_joint_training.py',
         '03_architecture_search_space.py','04_compression_recovery.py')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def module(p,name):
    s=importlib.util.spec_from_file_location(name,p)
    if s is None or s.loader is None:raise RuntimeError('Cannot import '+str(p))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def discover(root):
    if sha(root/CONFIG)!=LOCKED:raise RuntimeError('Locked configuration SHA mismatch')
    config=read(root/CONFIG)
    if config.get('status')!='LOCKED' or config.get('selection',{}).get('test_access')!='sealed_until_selection_frozen':
        raise RuntimeError('Config not locked or test firewall changed')
    for source in SOURCES:
        if not (root/'Codes'/source).is_file():raise FileNotFoundError(root/'Codes'/source)
    if not (root/AUTH).is_file():raise FileNotFoundError('Step18 authorization missing')
    auth=read(root/AUTH)
    if auth.get('locked_config_sha256')!=LOCKED or not auth.get('no_test_access'):
        raise RuntimeError('Step18 authorization mismatch')
    for source,expected in auth.get('source_sha256',{}).items():
        if sha(root/source)!=expected:raise RuntimeError('Step18 training source changed: '+source)
    step18=module(root/'Codes'/'18_cajo_final_joint_training.py','cajo20_step18')
    expected=step18.stages(config)
    if len(expected)!=45 or len(set(c['candidate_id'] for c in expected))!=45:
        raise RuntimeError('Expected exactly 45 unique alternative candidates')
    if auth.get('candidate_ids')!=[c['candidate_id'] for c in expected]:
        raise RuntimeError('Step18 authorization candidate list mismatch')
    items=[]
    for c in expected:
        cid=c['candidate_id'];p=root/CANDIDATES/(cid+'.json')
        if not p.is_file() or read(p)!=c:raise RuntimeError('Candidate contract mismatch: '+cid)
        folder=root/EXPERIMENTS/c['task']/c['strategy']/cid/('seed_'+str(c['seed']))
        hp=folder/'history.json';bp=folder/'best_model.pt'
        items.append((p,c,hp,bp))
    return config,items

def validate(root,item,hash_bytes=True):
    p,c,hp,bp=item;cid=c['candidate_id']
    if not hp.is_file() or not bp.is_file():raise FileNotFoundError('Missing history/checkpoint: '+cid)
    h=read(hp)
    if h.get('status')!='COMPLETE' or h.get('test_access')!='SEALED':raise RuntimeError('Incomplete or unsealed: '+cid)
    if h.get('candidate')!=c:raise RuntimeError('History candidate differs: '+cid)
    if len(h.get('history',[]))!=20 or h['history'][-1].get('epoch')!=20:
        raise RuntimeError('Not 20 complete epochs: '+cid)
    fp=h.get('fingerprints',{})
    if fp.get('config_sha256')!=LOCKED or fp.get('candidate_sha256')!=sha(p):
        raise RuntimeError('Training provenance mismatch: '+cid)
    if fp.get('adapter_sha256')!=sha(root/'Codes'/'04_compression_recovery.py'):
        raise RuntimeError('Training adapter SHA mismatch: '+cid)
    digest=sha(bp) if hash_bytes else None
    if hash_bytes and h.get('best_checkpoint_sha256')!=digest:
        raise RuntimeError('Checkpoint SHA mismatch: '+cid)
    return h,digest

def reconstruct(root,config,c,torch):
    """Rebuild topology from original trained P0 weights before deterministic surgery."""
    arch=module(root/'Codes'/'03_architecture_search_space.py','cajo20_arch')
    adapter=module(root/'Codes'/'04_compression_recovery.py','cajo20_adapter')
    model=arch.build_architecture(c['task'],c['architecture'],config,root)
    if c['pruning_ratio']==0:return model,None,[]
    cid=f"{c['task']}__{c['architecture']}__pretrain_P0_CE__s{c['seed']}"
    folder=root/EXPERIMENTS/c['task']/'joint'/cid/('seed_'+str(c['seed']))
    prior=folder/'best_model.pt';prior_hist=folder/'history.json'
    if not prior.is_file() or not prior_hist.is_file():raise FileNotFoundError('Missing P0 prerequisite: '+cid)
    ph=read(prior_hist)
    if ph.get('status')!='COMPLETE' or ph.get('test_access')!='SEALED' or ph.get('best_checkpoint_sha256')!=sha(prior):
        raise RuntimeError('P0 prerequisite provenance invalid: '+cid)
    blob=torch.load(prior,map_location='cpu',weights_only=True)
    if blob.get('candidate')!=ph.get('candidate') or blob['candidate']['candidate_id']!=cid:
        raise RuntimeError('P0 checkpoint candidate mismatch: '+cid)
    model.load_state_dict(blob['model'],strict=True)
    _,_,surgery,importance=adapter.sources(root)
    details=adapter.physical_prune(model,c['task'],float(c['pruning_ratio']),importance,surgery,
                                   arch,'importance',int(c['seed']),root)
    if not details:raise RuntimeError('Pruning did not change topology: '+c['candidate_id'])
    return model,prior,details

def benchmark(root,config,item,args,step16):
    import torch
    # Step16 measures on RTX CUDA and does not open any dataset.
    def scoped_validate(_root,_config,_item):return validate(_root,_item,True)
    def scoped_build(_root,_config,_c,_torch):
        model,source,details=reconstruct(_root,_config,_c,_torch)
        return model,source if source is not None else _item_source_placeholder,details
    # Step16 expects a baseline checkpoint to fingerprint. For P0, use its
    # own trained checkpoint as the explicitly labeled provenance anchor.
    _item_source_placeholder=item[3]
    original_validate,original_build=step16.validate_item,step16.build_model
    step16.validate_item=scoped_validate;step16.build_model=scoped_build
    try:result=step16.benchmark(root,config,item,args)
    finally:
        step16.validate_item=original_validate;step16.build_model=original_build
    c=item[1];samples=result['latency_samples_ms'];energy=result['energy']['repeats']
    raw=[z['raw_device_energy_mj']/z['calls'] for z in energy
         if z.get('raw_device_energy_mj') is not None and z.get('calls',0)>0]
    result.update(schema='cajo_alternative_gpu_benchmark_v2',architecture=c['architecture'],
                  pruning_ratio=c['pruning_ratio'],mode=c['mode'],
                  operation_order=c['operation_order'],
                  candidate_contract_path=str(item[0].relative_to(root)),
                  source_checkpoint_role='trained_alternative_P0' if c['pruning_ratio']>0 else 'self_P0',
                  source_checkpoint_sha256=result.pop('baseline_checkpoint_sha256'),
                  latency_mean_ms=statistics.mean(samples),
                  latency_sd_ms=statistics.stdev(samples) if len(samples)>1 else 0,
                  model_weight_and_buffer_bytes=result['parameter_bytes']+result['buffer_bytes'],
                  model_weight_and_buffer_note='FP32 tensors, not a serialized deployment package',
                  device_energy_raw_mj_per_inference_mean=statistics.mean(raw) if raw else None,
                  device_energy_raw_mj_per_inference_median=statistics.median(raw) if raw else None,
                  energy_measurement_scope='NVML total GPU device energy, not supply V*I, not full-system energy',
                  protocol_alignment={'mean_latency_equation_6':True,
                      'complete_memory_equation_3':False,
                      'device_energy_measured_not_supply_equation_8':True,
                      'matched_budget_search_demonstrated':False,
                      'full_four_dimensional_joint_search_demonstrated':False},
                  step16_source_sha256=sha(root/'Codes'/'16_cajo_gpu_hardware_benchmark.py'))
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['inspect','run-one','run-all','summarize'])
    ap.add_argument('--root',type=Path,default=Path('..'))
    ap.add_argument('--candidate-id')
    ap.add_argument('--device-index',type=int,default=0)
    ap.add_argument('--batch-size',type=int,default=1)
    ap.add_argument('--warmups',type=int,default=20)
    ap.add_argument('--timed-runs',type=int,default=100)
    ap.add_argument('--energy-repeats',type=int,default=5)
    ap.add_argument('--min-energy-seconds',type=float,default=10)
    ap.add_argument('--energy-chunk',type=int,default=100)
    ap.add_argument('--max-energy-calls',type=int,default=100000)
    args=ap.parse_args();root=args.root.resolve()
    config,items=discover(root)
    if args.action=='inspect':
        print(json.dumps({'config_sha256':LOCKED,'expected':len(items),
          'complete_history_checkpoint_pairs':sum(h.is_file() and b.is_file() for _,_,h,b in items),
          'already_benchmarked':sum((root/OUT/(c['candidate_id']+'.json')).is_file() for _,c,_,_ in items),
          'candidates':[{'candidate_id':c['candidate_id'],'history':h.is_file(),
             'checkpoint':b.is_file(),'benchmark':(root/OUT/(c['candidate_id']+'.json')).is_file()}
             for _,c,h,b in items],
          'scope':'FP32 RTX GPU only; no test, retraining, INT8 or Pareto selection'},indent=2));return
    if args.action=='summarize':
        rows=[]
        for _,c,_,_ in items:
            p=root/OUT/(c['candidate_id']+'.json')
            if not p.is_file():continue
            d=read(p)
            if d.get('locked_config_sha256')!=LOCKED or d.get('candidate_id')!=c['candidate_id']:
                raise RuntimeError('Invalid benchmark metadata: '+str(p))
            rows.append({k:d.get(k) for k in ('candidate_id','task','architecture','seed','pruning_ratio','kd',
              'best_validation_accuracy_percent','latency_mean_ms','latency_median_ms',
              'parameter_bytes','buffer_bytes','model_weight_and_buffer_bytes',
              'device_energy_raw_mj_per_inference_mean')})
        print(json.dumps({'expected':45,'benchmarked':len(rows),'missing':45-len(rows),
                          'rows':rows},indent=2));return
    if min(args.batch_size,args.warmups,args.timed_runs,args.energy_repeats,args.energy_chunk,args.max_energy_calls)<1 or args.min_energy_seconds<1:
        raise ValueError('Benchmark quantities must be positive')
    if args.action=='run-one' and not args.candidate_id:raise ValueError('--candidate-id required')
    selected=[item for item in items if args.action=='run-all' or item[1]['candidate_id']==args.candidate_id]
    if not selected:raise ValueError('Unknown candidate ID')
    step16=module(root/'Codes'/'16_cajo_gpu_hardware_benchmark.py','cajo20_step16')
    for item in selected:
        c=item[1];dest=root/OUT/(c['candidate_id']+'.json')
        if dest.exists():
            d=read(dest)
            if d.get('schema')!='cajo_alternative_gpu_benchmark_v2' or d.get('candidate_id')!=c['candidate_id'] or d.get('best_checkpoint_sha256')!=sha(item[3]) or d.get('locked_config_sha256')!=LOCKED:
                raise RuntimeError('Existing benchmark invalid, refusing overwrite: '+str(dest))
            print('SKIP VERIFIED',c['candidate_id'],flush=True);continue
        print('BENCHMARK',c['candidate_id'],flush=True)
        report=benchmark(root,config,item,args,step16)
        step16.write_new(dest,report)
        print('SAVED',c['candidate_id'],'mean_latency_ms=',round(report['latency_mean_ms'],5),
              'raw_gpu_energy_mj_per_inference=',report['device_energy_raw_mj_per_inference_mean'],flush=True)
    print('STEP 20 FINISHED. TEST SEALED. NO TRAINING. NO FEASIBILITY CLAIM.')

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('STEP 20 BLOCKED:',type(e).__name__+':',e,file=sys.stderr);sys.exit(2)
