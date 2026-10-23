#!/usr/bin/env python3
"""CAJO Step 16: read-only, validation-selected FP32 RTX GPU checkpoint benchmark.

Run from Codes:
  python 16_cajo_gpu_hardware_benchmark.py inspect --root ..
  python 16_cajo_gpu_hardware_benchmark.py run-one --root .. --candidate-id ID
  python 16_cajo_gpu_hardware_benchmark.py run-all --root ..

Never loads train/validation/test examples, trains, modifies the locked config,
uses INT8, or declares full-runtime-memory/energy feasibility. Only trusted
locally generated PyTorch checkpoints should be loaded.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, platform, statistics, sys, time
from datetime import datetime, timezone
from pathlib import Path

CONFIG_REL=Path('Reports/Optimization_Protocol/Final_Experiment_Config/cajo_experiment_config.json')
CAND_REL=Path('Reports/Optimization_Protocol/Proposed_FP32_Candidates')
EXP_REL=Path('Reports/CAJO_Experiments')
OUT_REL=Path('Reports/CAJO_GPU_Hardware_Benchmark')
SHAPES={'CIFAR10':(1,3,32,32),'CIFAR100':(1,3,32,32),'SpeechCommands_v0.02':(1,1,40,101)}
CLASSES={'CIFAR10':10,'CIFAR100':100,'SpeechCommands_v0.02':35}
LOCKED_SHA='56361ba3ef2cb530d54a923cf4f5b3758a22bee82a9119822395cef0be9bbdc5'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def import_module(p,name):
    spec=importlib.util.spec_from_file_location(name,p)
    if spec is None or spec.loader is None:raise RuntimeError('Cannot import '+str(p))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def write_new(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError('Immutable report already exists: '+str(p))
    temp=p.with_name(p.name+'.partial')
    with temp.open('x',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,p)

def inventory(root):
    cp=root/CONFIG_REL
    if sha(cp)!=LOCKED_SHA:raise RuntimeError('Locked config SHA mismatch')
    config=read(cp)
    if config.get('status')!='LOCKED':raise RuntimeError('Configuration not LOCKED')
    items=[]
    for p in sorted((root/CAND_REL).glob('*.json')):
        c=read(p)
        if c.get('architecture')!='baseline' or c.get('precision')!='FP32' or c.get('mode')!='recovery':continue
        if c.get('pruning_ratio')!=0.25 or c.get('seed') not in (42,43,44):continue
        if c.get('strategy')=='individual' and c.get('kd') is False:pass
        elif c.get('strategy')=='sequential' and c.get('kd') is True:pass
        else:continue
        if c.get('task') not in SHAPES:continue
        out=root/EXP_REL/c['task']/c['strategy']/c['candidate_id']/('seed_'+str(c['seed']))
        hist=out/'history.json';best=out/'best_model.pt'
        items.append((p,c,hist,best))
    if len(items)!=27 or len({c['candidate_id'] for _,c,_,_ in items})!=27:
        raise RuntimeError('Expected 27 unique supported FP32 candidates, found '+str(len(items)))
    return config,items

def validate_item(root,config,item):
    p,c,hp,bp=item
    if not hp.is_file() or not bp.is_file():raise FileNotFoundError('Missing history/checkpoint for '+c['candidate_id'])
    h=read(hp)
    if h.get('status')!='COMPLETE' or h.get('test_access')!='SEALED':raise RuntimeError('Incomplete/unsealed history: '+c['candidate_id'])
    if len(h.get('history',[]))!=20 or h['history'][-1].get('epoch')!=20:raise RuntimeError('Expected 20 completed epochs')
    if h.get('candidate',{}).get('candidate_id')!=c['candidate_id']:raise RuntimeError('Candidate/history ID mismatch')
    fp=h.get('fingerprints',{})
    if fp.get('config_sha256')!=LOCKED_SHA or fp.get('candidate_sha256')!=sha(p):
        raise RuntimeError('Training provenance mismatch: '+c['candidate_id'])
    adapter=root/'Codes'/'04_compression_recovery.py'
    if fp.get('adapter_sha256')!=sha(adapter):raise RuntimeError('Adapter changed since training')
    digest=sha(bp)
    if h.get('best_checkpoint_sha256')!=digest:raise RuntimeError('Best checkpoint SHA mismatch: '+c['candidate_id'])
    return h,digest

def build_model(root,config,c,torch):
    """Reproduce exact physical channel surgery without creating ANY dataset objects."""
    adapter=import_module(root/'Codes'/'04_compression_recovery.py','cajo_step16_pruning')
    arch,unused_loader,surgery,importance=adapter.sources(root)
    teacher,baseline=adapter.frozen_teacher(root,config,c['task'],arch,torch)
    model=arch.build_architecture(c['task'],'baseline',config,root)
    model.load_state_dict(teacher.state_dict(),strict=True)
    details=adapter.physical_prune(model,c['task'],float(c['pruning_ratio']),importance,surgery,
                                   arch,c.get('pruning_policy','importance'),int(c['seed']),root)
    del teacher
    return model,baseline,details

def nvml_handle(torch):
    try:
        import pynvml as nv
        nv.nvmlInit()
        index=torch.cuda.current_device()
        handle=nv.nvmlDeviceGetHandleByIndex(index)
        return nv,handle,None
    except Exception as e:return None,None,type(e).__name__+': '+str(e)[:150]

def energy_mj(nv,handle):
    try:return float(nv.nvmlDeviceGetTotalEnergyConsumption(handle))
    except Exception:return None

def time_interval(torch,model,x,ncalls,nv,handle,execute=True):
    torch.cuda.synchronize()
    before=energy_mj(nv,handle) if nv else None
    wall=time.perf_counter()
    if execute:
        with torch.inference_mode():
            for _ in range(ncalls):model(x)
        torch.cuda.synchronize()
    else:
        # Same wall duration is matched outside this function by caller.
        pass
    elapsed=time.perf_counter()-wall
    after=energy_mj(nv,handle) if nv else None
    delta=(after-before) if before is not None and after is not None and after>=before else None
    return elapsed,delta

def benchmark(root,config,item,args):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; refusing CPU fallback')
    if args.device_index>=torch.cuda.device_count():raise RuntimeError('Invalid CUDA device index')
    torch.cuda.set_device(args.device_index)
    device=torch.device('cuda',args.device_index)
    h,checkpoint_sha=validate_item(root,config,item)
    p,c,hp,bp=item
    model,baseline,details=build_model(root,config,c,torch)
    # Trusted local checkpoint only. Safe tensor/container loader.
    blob=torch.load(bp,map_location='cpu',weights_only=True)
    if not isinstance(blob,dict) or 'model' not in blob:raise RuntimeError('Unexpected checkpoint structure')
    model.load_state_dict(blob['model'],strict=True)
    model=model.to(device).eval()
    x=torch.zeros((args.batch_size,)+SHAPES[c['task']][1:],device=device,dtype=torch.float32)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with torch.inference_mode():
        y=model(x)
        if tuple(y.shape)!=(args.batch_size,CLASSES[c['task']]) or not bool(torch.isfinite(y).all()):
            raise RuntimeError('Output shape or finite-value check failed')
        for _ in range(args.warmups):model(x)
    torch.cuda.synchronize()
    # Warmed, event-based per-call latency; no host-launch latency mislabeling.
    lat=[]
    with torch.inference_mode():
        for _ in range(args.timed_runs):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();model(x);end.record();end.synchronize()
            lat.append(start.elapsed_time(end))
    # Report allocator scope explicitly. Reserved is allocator cache, not full device RSS.
    torch.cuda.synchronize();torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    alloc_before=torch.cuda.memory_allocated(device);reserve_before=torch.cuda.memory_reserved(device)
    with torch.inference_mode():
        for _ in range(5):model(x)
    torch.cuda.synchronize()
    peak_alloc=torch.cuda.max_memory_allocated(device);peak_reserved=torch.cuda.max_memory_reserved(device)
    nv,handle,nv_error=nvml_handle(torch)
    # Device-energy counter resolution requires sufficiently long measurement windows.
    # Use repeated batches; collect raw GPU device energy, plus idle windows of equal
    # duration (no assumption that all device power belongs to this process).
    energy=[]
    for _ in range(args.energy_repeats):
        torch.cuda.synchronize()
        start_counter=energy_mj(nv,handle) if nv else None
        t0=time.perf_counter()
        calls=0
        with torch.inference_mode():
            while calls<args.max_energy_calls:
                for __ in range(args.energy_chunk):model(x)
                calls+=args.energy_chunk
                torch.cuda.synchronize()
                if time.perf_counter()-t0>=args.min_energy_seconds:break
        duration=time.perf_counter()-t0
        end_counter=energy_mj(nv,handle) if nv else None
        raw=(end_counter-start_counter) if start_counter is not None and end_counter is not None and end_counter>=start_counter else None
        # Idle is sampled AFTER the active workload; equal-duration wall window.
        torch.cuda.synchronize();idle_before=energy_mj(nv,handle) if nv else None
        idle_start=time.perf_counter();time.sleep(duration)
        idle_after=energy_mj(nv,handle) if nv else None
        idle=(idle_after-idle_before) if idle_before is not None and idle_after is not None and idle_after>=idle_before else None
        net=max(0.0,raw-idle) if raw is not None and idle is not None else None
        energy.append({'calls':calls,'active_seconds':duration,'raw_device_energy_mj':raw,
                       'idle_device_energy_mj':idle,'idle_seconds':time.perf_counter()-idle_start,
                       'idle_subtracted_mj':net,'idle_subtracted_mj_per_inference':None if net is None else net/calls})
    params=sum(z.numel() for z in model.parameters())
    buffers=sum(z.numel()*z.element_size() for z in model.buffers())
    param_bytes=sum(z.numel()*z.element_size() for z in model.parameters())
    energy_valid=[r['idle_subtracted_mj_per_inference'] for r in energy if r['idle_subtracted_mj_per_inference'] is not None]
    return {
      'schema':'cajo_gpu_hardware_benchmark_v1','created_utc':datetime.now(timezone.utc).isoformat(),
      'candidate_id':c['candidate_id'],'task':c['task'],'strategy':c['strategy'],'seed':c['seed'],
      'pruning_policy':c.get('pruning_policy','importance'),'kd':c['kd'],'precision':'FP32',
      'locked_config_sha256':LOCKED_SHA,'candidate_contract_sha256':sha(p),
      'training_adapter_sha256':h['fingerprints']['adapter_sha256'],
      'training_history_sha256':sha(hp),'best_checkpoint_sha256':checkpoint_sha,
      'baseline_checkpoint_sha256':sha(baseline),'best_validation_accuracy_percent':h['best_validation_accuracy'],
      'best_epoch':h['best_epoch'],'checkpoint_storage_bytes':bp.stat().st_size,
      'checkpoint_storage_note':'Training checkpoint container; NOT deployment inference artifact size',
      'parameter_bytes':param_bytes,'buffer_bytes':buffers,'parameter_count':params,
      'gpu':torch.cuda.get_device_name(device),'cuda_index':args.device_index,
      'cuda_version':torch.version.cuda,'pytorch_version':torch.__version__,'platform':platform.platform(),
      'batch_size':args.batch_size,'input_shape':list(x.shape),'warmups':args.warmups,
      'timed_runs':args.timed_runs,'latency_method':'CUDA event elapsed GPU time, synchronized per call',
      'latency_median_ms':statistics.median(lat),'latency_p95_ms':sorted(lat)[int(.95*(len(lat)-1))],
      'latency_samples_ms':lat,'throughput_images_per_second_estimate':args.batch_size*1000/statistics.median(lat),
      'throughput_note':'Reciprocal of per-call median; not independently measured sustained throughput',
      'memory':{'allocator_baseline_allocated_bytes':alloc_before,'allocator_baseline_reserved_bytes':reserve_before,
                'allocator_peak_allocated_bytes':peak_alloc,'allocator_peak_reserved_bytes':peak_reserved,
                'incremental_peak_allocated_bytes':max(0,peak_alloc-alloc_before),
                'total_native_runtime_workspace_bytes':None,
                'scope':'PyTorch CUDA caching allocator only; excludes untracked CUDA/cuDNN native allocations and other processes'},
      'energy':{'source':'NVML GPU total-energy counter, device-level; not whole-system or per-process',
                'nvml_error':nv_error,'repeats':energy,'idle_subtracted_mj_per_inference_median':statistics.median(energy_valid) if energy_valid else None,
                'note':'Adjacent equal-duration idle subtraction is an estimate; counter may be coarse, and other GPU activity contaminates readings'},
      'pruning_surgery_blocks':len(details),'test_access':'SEALED','dataset_samples_accessed':False,
      'model_selection_performed':False,'full_runtime_memory_feasibility_decided':False,
      'energy_feasibility_decided':False,'gpu_int8_verified':False,'joint_nas_demonstrated':False
    }

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('inspect','run-one','run-all','summarize'))
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
    a=ap.parse_args();root=a.root.resolve()
    if min(a.batch_size,a.warmups,a.timed_runs,a.energy_repeats,a.energy_chunk,a.max_energy_calls)<1 or a.min_energy_seconds<1:
        raise ValueError('All benchmark quantities must be positive')
    config,items=inventory(root)
    if a.action=='inspect':
        status=[]
        for item in items:
            p,c,h,b=item
            status.append({'candidate_id':c['candidate_id'],'history_exists':h.is_file(),
                           'checkpoint_exists':b.is_file(),'benchmark_exists':(root/OUT_REL/(c['candidate_id']+'.json')).exists()})
        print(json.dumps({'locked_config_sha256':LOCKED_SHA,'candidate_count':len(items),'items':status},indent=2));return
    if a.action=='summarize':
        rows=[]
        for _,c,_,_ in items:
            p=root/OUT_REL/(c['candidate_id']+'.json')
            if not p.is_file():continue
            d=read(p)
            if d['locked_config_sha256']!=LOCKED_SHA:raise RuntimeError('Benchmark config mismatch')
            rows.append({'candidate_id':d['candidate_id'],'task':d['task'],'strategy':d['strategy'],
                         'seed':d['seed'],'pruning_policy':d['pruning_policy'],'kd':d['kd'],
                         'validation_accuracy_percent':d['best_validation_accuracy_percent'],
                         'latency_median_ms':d['latency_median_ms'],
                         'parameter_bytes':d['parameter_bytes'],
                         'allocator_peak_reserved_bytes':d['memory']['allocator_peak_reserved_bytes'],
                         'idle_subtracted_mj_per_inference':d['energy']['idle_subtracted_mj_per_inference_median']})
        out=root/OUT_REL/'summary.csv'
        if out.exists():raise FileExistsError('Summary already exists: '+str(out))
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['candidate_id']);writer.writeheader();writer.writerows(rows)
        print('SUMMARY',len(rows),'of',len(items),out);return
    selected=[i for i in items if a.action=='run-all' or i[1]['candidate_id']==a.candidate_id]
    if not selected:raise RuntimeError('Unknown/missing --candidate-id')
    for item in selected:
        c=item[1];dest=root/OUT_REL/(c['candidate_id']+'.json')
        if dest.exists():
            d=read(dest)
            if d.get('best_checkpoint_sha256')!=sha(item[3]) or d.get('locked_config_sha256')!=LOCKED_SHA:
                raise RuntimeError('Existing benchmark provenance mismatch: '+c['candidate_id'])
            print('SKIP EXISTING',c['candidate_id'],flush=True);continue
        print('BENCHMARK',c['candidate_id'],flush=True)
        report=benchmark(root,config,item,a)
        write_new(dest,report)
        print('SAVED',dest,'latency_ms=',round(report['latency_median_ms'],4),
              'energy_mj_per_inference=',report['energy']['idle_subtracted_mj_per_inference_median'],flush=True)
    print('BENCHMARK COMMAND FINISHED. No test data accessed.')

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('CAJO STEP 16 BLOCKED:',type(e).__name__+':',e,file=sys.stderr);sys.exit(2)
