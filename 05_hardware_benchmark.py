#!/usr/bin/env python3
"""CAJO Step 5: reproducible CPU inference measurements; no training or test data.

Commands:
  python 05_hardware_benchmark.py --inspect
  python 05_hardware_benchmark.py --benchmark --model PATH --task CIFAR10 --precision FP32
  python 05_hardware_benchmark.py --benchmark --model PATH --task CIFAR10 --precision INT8

Model file must be a *trusted, locally generated* TorchScript inference artifact.
A TorchScript artifact is used to benchmark the exact exported FP32/INT8 graph;
loading arbitrary TorchScript files can execute code. Never use untrusted files.

The command measures one CPU thread, batch=1, 20 warmups, 100 timed calls,
median latency, actual artifact bytes, and CPU operator-level memory where
supported. It does not estimate device energy or claim embedded-device results.

Peak extra memory is obtained from PyTorch profiler memory-category timelines:
max of total CPU tensor memory excluding PARAMETER and OPTIMIZER_STATE. This is
CPU allocator tensor memory, not system-wide peak RSS or all native workspaces;
if the timeline is unavailable, memory is reported unavailable, not invented.
For deployment constraints requiring all native workspace memory, an external
allocator-level measurement is still necessary before declaring feasibility.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT/'Reports'/'Optimization_Protocol'/'Final_Experiment_Config'/'cajo_experiment_config.json'
TASKS = {'CIFAR100':(1,3,32,32),'CIFAR10':(1,3,32,32),'SpeechCommands_v0.02':(1,1,40,98)}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def protocol():
    if not CONFIG.is_file(): raise FileNotFoundError('Missing Step 1 config: '+str(CONFIG))
    config=json.loads(CONFIG.read_text(encoding='utf-8'))
    p=config.get('measurement',{})
    return config,p


def inspect():
    config,p=protocol()
    print('CAJO STEP 5: READ_ONLY_INSPECTION')
    print('Config SHA256:',digest(CONFIG))
    print('Tasks:',', '.join(TASKS))
    print('Inference: single-thread CPU, batch=1, FP32 or genuine exported INT8')
    print('Protocol: 20 warmups, 100 timed calls, median latency (milliseconds)')
    print('Storage: serialized inference artifact bytes; parameter storage separately when extractable')
    print('Memory: profiler CPU live non-parameter tensor allocation; unavailable if unsupported')
    print('No dataset, checkpoint, model, GPU, or result file accessed.')


def cpu_memory(model,x,torch):
    """Returns nonparameter tensor allocator peak or explicit unavailable reason."""
    try:
        from torch.profiler import profile, ProfilerActivity
        from torch.profiler._memory_profiler import Category
        with tempfile.TemporaryDirectory(prefix='cajo_cpu_memory_') as td:
            output=Path(td)/'timeline.json'
            with profile(activities=[ProfilerActivity.CPU], profile_memory=True,
                         record_shapes=True, with_stack=True) as prof:
                with torch.inference_mode(): model(x)
            prof.export_memory_timeline(str(output),device='cpu')
            raw=json.loads(output.read_text(encoding='utf-8'))
        if not (isinstance(raw,list) and len(raw)==2 and isinstance(raw[1],list) and raw[1]):
            return None,'unexpected memory timeline format'
        # Categories are ordered by their Enum integer values in PyTorch's
        # memory profiler. Exclude parameters and optimizer state explicitly.
        excluded={Category.PARAMETER.value,Category.OPTIMIZER_STATE.value}
        peaks=[]
        for snapshot in raw[1]:
            if not isinstance(snapshot,list):return None,'unexpected category snapshot'
            peaks.append(sum(max(0,int(v)) for i,v in enumerate(snapshot) if i not in excluded))
        if not peaks:return None,'empty memory timeline'
        return max(peaks),'cpu_profiler_live_nonparameter_tensors_only'
    except Exception as e:
        return None,'CPU memory timeline unavailable: '+type(e).__name__+': '+str(e)[:180]


def benchmark(args):
    config,p=protocol()
    if args.task not in config['tasks']:raise ValueError('Task not in approved config')
    if args.precision not in ('FP32','INT8'):raise ValueError('Unsupported precision')
    model_path=Path(args.model).resolve()
    if not model_path.is_file():raise FileNotFoundError(model_path)
    if model_path.suffix.lower() not in ('.pt','.ts','.torchscript'):
        raise ValueError('Expected trusted locally generated TorchScript artifact')
    import torch
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.backends.mkldnn.enabled=True
    # TorchScript may execute embedded code. Only load trusted artifacts.
    model=torch.jit.load(str(model_path),map_location='cpu').eval()
    x=torch.zeros(TASKS[args.task],dtype=torch.float32)
    with torch.inference_mode():
        y=model(x)
        if not isinstance(y,torch.Tensor) or y.ndim!=2 or y.shape[0]!=1:
            raise RuntimeError('Invalid inference output; expected [1,num_classes]')
        expected={'CIFAR100':100,'CIFAR10':10,'SpeechCommands_v0.02':35}[args.task]
        if y.shape[1]!=expected or not torch.isfinite(y).all():
            raise RuntimeError('Output shape or finite-value check failed')
        for _ in range(20):model(x)
        elapsed=[]
        for _ in range(100):
            t0=time.perf_counter_ns();model(x);elapsed.append((time.perf_counter_ns()-t0)/1e6)
    mem,mem_status=cpu_memory(model,x,torch)
    artifact_bytes=model_path.stat().st_size
    report={
      'schema':'cajo_cpu_benchmark_v1','task':args.task,'precision_declared':args.precision,
      'precision_verified':False if args.precision=='INT8' else None,
      'int8_note':'INT8 requires independent operator graph inspection; filename/declared precision is not proof' if args.precision=='INT8' else None,
      'artifact_path':str(model_path),'artifact_sha256':digest(model_path),
      'config_sha256':digest(CONFIG),'device':'local workstation CPU',
      'cpu':platform.processor(),'torch_version':torch.__version__,
      'threads':1,'batch_size':1,'warmups':20,'timed_runs':100,
      'latency_statistic':'median','latency_ms':statistics.median(elapsed),
      'latency_p95_ms':sorted(elapsed)[94],
      'artifact_storage_bytes':artifact_bytes,
      'artifact_storage_mib':artifact_bytes/(1024**2),
      'parameter_storage_bytes':None,
      'parameter_storage_note':'Serialized TorchScript artifact includes graph metadata; not equivalent to parameter-only storage',
      'peak_extra_tensor_memory_bytes':mem,
      'peak_extra_tensor_memory_mib':None if mem is None else mem/(1024**2),
      'memory_measurement_scope':mem_status,
      'total_inference_memory_mib':None,
      'total_memory_note':'Not computed: native backend workspace may be untracked and parameter storage not separately established',
      'energy_measured':False,'test_data_accessed':False,'feasibility_decided':False,
      'latency_samples_ms':elapsed
    }
    if args.output:
        out=Path(args.output).resolve()
        if out.exists():raise FileExistsError('Refusing to overwrite existing report: '+str(out))
        out.parent.mkdir(parents=True,exist_ok=True)
        tmp=out.with_name(out.name+'.partial')
        if tmp.exists():raise FileExistsError('Interrupted previous write: '+str(tmp))
        try:
            with tmp.open('x',encoding='utf-8') as f:
                json.dump(report,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
            os.replace(tmp,out)
        finally:
            if tmp.exists():tmp.unlink()
    print('CAJO STEP 5: CPU_BENCHMARK')
    print('Task:',args.task,'Precision (declared):',args.precision)
    print('Median latency (ms):',round(report['latency_ms'],6))
    print('Artifact size (MiB):',round(report['artifact_storage_mib'],6))
    print('Live nonparameter tensor peak (MiB):',report['peak_extra_tensor_memory_mib'])
    print('Memory measurement scope:',mem_status)
    print('INT8 graph verified:',report['precision_verified'])
    print('Feasibility decided: NO; no test dataset accessed')
    if args.output:print('Report:',out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--inspect',action='store_true')
    g.add_argument('--benchmark',action='store_true')
    ap.add_argument('--model',help='Trusted locally generated TorchScript inference artifact')
    ap.add_argument('--task',choices=list(TASKS))
    ap.add_argument('--precision',choices=['FP32','INT8'])
    ap.add_argument('--output',help='Optional new JSON report path; never overwritten')
    a=ap.parse_args()
    if a.inspect:inspect();return
    if not a.model or not a.task or not a.precision:ap.error('--benchmark requires --model, --task and --precision')
    benchmark(a)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('ERROR:',e,file=sys.stderr);sys.exit(2)
