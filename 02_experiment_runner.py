#!/usr/bin/env python3
"""CAJO Step 2: one reusable, fail-closed experiment runner.

Commands:
  python 02_experiment_runner.py --plan
  python 02_experiment_runner.py --run --candidate candidate.json
  python 02_experiment_runner.py --resume --candidate candidate.json

No full runs while Step 1 is unlocked, manuscript alignment unconfirmed,
or full-training authorization false. Candidate implementations are provided
by the future Steps 3-4 through ONE importable module each; this runner does
not invent architecture, pruning, quantization, or data-loader implementations.

Candidate schema (created later by Step 6):
 {"candidate_id":"...", "task":"CIFAR10", "architecture":"baseline",
  "pruning_ratio":0.25, "precision":"FP32", "kd":true,
  "strategy":"individual", "seed":42, "epochs":20,
  "mode":"recovery", "operation_order":["pruning","KD"]}

Integration contract for Steps 3-4 (in Codes/04_compression_recovery.py):
 prepare_candidate(*, root, config, candidate, device) -> dict with keys:
  student: torch.nn.Module (constructed, not trained)
  teacher: torch.nn.Module or None (frozen, eval, KD only)
  train_epoch_batches: callable(epoch:int) -> iterable of (x,y) or (x,y,trace)
  validation_batches: callable() -> iterable of (x,y)
  provenance: dict (split, source, initial weights and operation order evidence)
 The adapter MUST verify frozen split manifest/checkpoint hashes, train-only
 calibration, exact architecture/pruning/precision implementation, and a sealed
 test partition. The runner refuses to run without this adapter.

IMPORTANT: This is a training/evaluation orchestration kernel, NOT a complete
end-to-end CAJO pipeline until the later modules and protocol lock exist.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_REL = Path('Reports/Optimization_Protocol/Final_Experiment_Config/cajo_experiment_config.json')
OUT_REL = Path('Reports/CAJO_Experiments')
TASKS = ('CIFAR100', 'CIFAR10', 'SpeechCommands_v0.02')
STRATEGIES = ('individual', 'sequential', 'joint', 'screening')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    if tmp.exists():
        raise RuntimeError('Previous interrupted JSON write; inspect before retry: ' + str(tmp))
    try:
        with open(tmp, 'x', encoding='utf-8') as stream:
            json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_config(root):
    path = root / CONFIG_REL
    if not path.is_file():
        raise FileNotFoundError('Missing Step 1 configuration: ' + str(path))
    config = read_json(path)
    if config.get('schema') != 'cajo_final_experiment_config_v1':
        raise ValueError('Unsupported configuration schema')
    return config, path


def run_gate(config):
    errors = []
    approval = config.get('approval', {})
    if config.get('status') != 'LOCKED':
        errors.append('Step 1 configuration not LOCKED')
    if approval.get('human_approved') is not True:
        errors.append('Human approval missing')
    if approval.get('manuscript_energy_alignment_confirmed') is not True:
        errors.append('Manuscript energy alignment pending')
    if approval.get('full_training_authorized') is not True:
        errors.append('Full training not authorized')
    if config.get('claims_scope', {}).get('energy_policy') != 'MEASURED_DEVICE_LEVEL':
        errors.append('Measured device-level energy policy required')
    if not config.get('measurement', {}).get('energy_metric'):
        errors.append('Energy measurement method not specified')
    if 'minimize_energy' not in config.get('selection', {}).get('objectives', []):
        errors.append('Energy objective not registered')
    if config.get('selection', {}).get('test_access') != 'sealed_until_selection_frozen':
        errors.append('Test-set firewall configuration drift')
    if config.get('provenance', {}).get('baseline_retraining_permitted') is not False:
        errors.append('Baseline retraining prohibition missing')
    return errors


def candidate_contract(candidate, config):
    required = ('candidate_id','task','architecture','pruning_ratio','precision','kd','strategy','seed','epochs','mode','operation_order')
    missing = [k for k in required if k not in candidate]
    if missing:
        raise ValueError('Candidate missing: ' + ', '.join(missing))
    cid = candidate['candidate_id']
    if not isinstance(cid, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', cid):
        raise ValueError('Unsafe candidate_id')
    task = candidate['task']
    if task not in TASKS or task not in config['tasks']:
        raise ValueError('Unknown task')
    if candidate['strategy'] not in STRATEGIES:
        raise ValueError('Unknown strategy')
    architectures = {a['id'] for a in config['tasks'][task]['architectures']}
    if candidate['architecture'] not in architectures:
        raise ValueError('Architecture not declared in Step 1')
    if candidate['pruning_ratio'] not in config['techniques']['structured_pruning_ratios']:
        raise ValueError('Pruning ratio not declared in Step 1')
    if candidate['precision'] not in config['techniques']['precision_options']:
        raise ValueError('Precision not declared in Step 1')
    if type(candidate['kd']) is not bool or candidate['kd'] not in config['techniques']['kd_options']:
        raise ValueError('Invalid KD setting')
    if type(candidate['seed']) is not int or candidate['seed'] not in config['recovery']['seeds']:
        raise ValueError('Seed not in prespecified set')
    if type(candidate['epochs']) is not int or candidate['epochs'] < 1:
        raise ValueError('Invalid epochs')
    if candidate['strategy'] == 'screening':
        if candidate['epochs'] > 5:
            raise ValueError('Screening exceeds prespecified 5-epoch maximum')
    elif candidate['epochs'] != config['recovery']['epochs']:
        raise ValueError('Final candidate must use fixed recovery epochs')
    if candidate['mode'] not in ('recovery','architecture_training'):
        raise ValueError('Unknown training mode')
    if not isinstance(candidate['operation_order'], list) or not all(isinstance(x,str) for x in candidate['operation_order']):
        raise ValueError('Invalid operation order')
    if candidate['mode'] == 'architecture_training' and candidate['architecture'] == 'baseline':
        raise ValueError('Frozen baseline retraining prohibited')
    # Quantized inference candidates are not trainable in this kernel. PTQ
    # belongs to the compression module after FP32 checkpoint recovery.
    if candidate['precision'] != 'FP32':
        raise ValueError('INT8 candidates require Step 4 post-training quantization; do not train INT8 here')
    return cid


def seed_everything(seed, torch):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def import_adapter(root):
    path = root / 'Codes' / '04_compression_recovery.py'
    if not path.is_file():
        raise FileNotFoundError('Step 4 implementation not available: ' + str(path))
    spec = importlib.util.spec_from_file_location('cajo_step4_adapter', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Unable to import Step 4 adapter')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'prepare_candidate', None)):
        raise RuntimeError('Step 4 adapter must export prepare_candidate(...)')
    return module, path


def safe_torch_load(torch, path, device):
    # Checkpoint is generated by this script only, with simple containers and tensors.
    return torch.load(path, map_location=device, weights_only=True)


def unpack_batch(batch):
    if not isinstance(batch, (tuple,list)) or len(batch) not in (2,3):
        raise RuntimeError('Batch must be (inputs, labels[, trace])')
    return batch[0],batch[1]


def evaluate(model, batches, device, torch):
    correct = total = 0
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            x,y = unpack_batch(batch)
            x,y = x.to(device),y.to(device)
            logits = model(x)
            if logits.ndim != 2 or logits.shape[0] != y.numel():
                raise RuntimeError('Unexpected model logits/label shape')
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    if total == 0:
        raise RuntimeError('Empty validation partition')
    return {'accuracy_percent':100.0*correct/total, 'correct':correct, 'samples':total}


def fit(root, config, config_path, candidate, candidate_path, resume=False):
    # No torch import or GPU initialization before all approval gates.
    blockers = run_gate(config)
    if blockers:
        raise PermissionError('RUN BLOCKED: ' + '; '.join(blockers))
    cid = candidate_contract(candidate, config)
    adapter, adapter_path = import_adapter(root)
    import torch
    import torch.nn.functional as F
    seed = candidate['seed']
    seed_everything(seed, torch)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result = adapter.prepare_candidate(root=root, config=config, candidate=candidate, device=device)
    if not isinstance(result, dict):
        raise RuntimeError('Step 4 adapter must return dict')
    for k in ('student','train_epoch_batches','validation_batches','provenance'):
        if k not in result:
            raise RuntimeError('Step 4 adapter missing ' + k)
    if not callable(result['train_epoch_batches']) or not callable(result['validation_batches']):
        raise RuntimeError('Step 4 batch factories must be callable')
    provenance = result['provenance']
    if not isinstance(provenance,dict) or provenance.get('test_access') != 'sealed':
        raise PermissionError('Step 4 adapter must attest sealed test access')
    if provenance.get('frozen_manifest_sha256') != config['tasks'][candidate['task']]['split_manifest_sha256']:
        raise RuntimeError('Dataset manifest provenance mismatch')
    if provenance.get('frozen_teacher_checkpoint_sha256') != config['tasks'][candidate['task']]['checkpoint_sha256']:
        raise RuntimeError('Frozen baseline provenance mismatch')
    if provenance.get('architecture') != candidate['architecture'] or provenance.get('operation_order') != candidate['operation_order']:
        raise RuntimeError('Candidate construction provenance mismatch')
    model = result['student'].to(device)
    teacher = result.get('teacher')
    if candidate['kd']:
        if teacher is None:
            raise RuntimeError('KD candidate missing frozen teacher')
        teacher = teacher.to(device).eval()
        teacher.requires_grad_(False)
    elif teacher is not None:
        raise RuntimeError('CE candidate must not use teacher')
    recovery = config['recovery']
    opt = torch.optim.SGD(model.parameters(),lr=recovery['learning_rate'],momentum=recovery['momentum'],weight_decay=recovery['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=recovery['epochs'],eta_min=0)
    out = root / OUT_REL / candidate['task'] / candidate['strategy'] / cid / ('seed_'+str(seed))
    out.mkdir(parents=True,exist_ok=True)
    best_path = out / 'best_model.pt'
    state_path = out / 'resume_state.pt'
    history_path = out / 'history.json'
    fingerprints = {'config_sha256':sha256(config_path),'candidate_sha256':sha256(candidate_path),
                    'adapter_sha256':sha256(adapter_path)}
    history = []
    start_epoch = 1
    best_accuracy = -1.0
    best_epoch = None
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError('Resume checkpoint missing')
        state = safe_torch_load(torch,state_path,device)
        if state['fingerprints'] != fingerprints:
            raise RuntimeError('Resume provenance drift')
        model.load_state_dict(state['model'],strict=True)
        opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['cpu_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        if state.get('numpy_rng') is not None:
            import numpy as np
            np.random.set_state((state['numpy_rng'][0], np.asarray(state['numpy_rng'][1], dtype='uint32'), state['numpy_rng'][2], state['numpy_rng'][3], state['numpy_rng'][4]))
        history = state['history']
        best_accuracy = state['best_accuracy']
        best_epoch = state['best_epoch']
        start_epoch = state['last_epoch'] + 1
    else:
        if any(p.exists() for p in (state_path,best_path,history_path)):
            raise FileExistsError('Existing outputs; use --resume or a distinct candidate ID')
    epochs = candidate['epochs']
    if start_epoch > epochs:
        print('ALREADY COMPLETE:',cid,'seed',seed)
        return
    started = time.perf_counter()
    for epoch in range(start_epoch,epochs+1):
        model.train()
        loss_total = 0.0
        seen = 0
        lr = float(opt.param_groups[0]['lr'])
        for batch in result['train_epoch_batches'](epoch):
            x,y = unpack_batch(batch)
            x,y = x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            ce = F.cross_entropy(logits,y)
            if candidate['kd']:
                with torch.no_grad():
                    teacher_logits = teacher(x)
                T = float(config['techniques']['kd_temperature'])
                alpha = float(config['techniques']['kd_alpha'])
                kd = F.kl_div(F.log_softmax(logits/T,dim=1),F.softmax(teacher_logits/T,dim=1),reduction='batchmean') * (T*T)
                loss = (1-alpha)*ce + alpha*kd
            else:
                loss = ce
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError('Nonfinite loss at epoch '+str(epoch))
            loss.backward()
            opt.step()
            n = int(y.numel())
            loss_total += float(loss.detach().item())*n
            seen += n
        if seen == 0:
            raise RuntimeError('Empty training epoch')
        validation = evaluate(model,result['validation_batches'](),device,torch)
        accuracy = validation['accuracy_percent']
        # Strict > preserves earliest epoch on ties.
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save({'model':model.state_dict(),'epoch':epoch,'validation':validation,
                        'fingerprints':fingerprints,'candidate':candidate},best_path)
        history.append({'epoch':epoch,'lr':lr,'train_loss':loss_total/seen,
                        'train_samples':seen,'validation':validation,'best_epoch':best_epoch})
        scheduler.step()  # exactly once, at epoch end
        np_rng = None
        if 'numpy' in sys.modules:
            import numpy as np
            raw_np_rng = np.random.get_state()
            np_rng = (raw_np_rng[0], raw_np_rng[1].tolist(), raw_np_rng[2], raw_np_rng[3], raw_np_rng[4])
        state = {'last_epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),
                 'scheduler':scheduler.state_dict(),'cpu_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                 'python_rng':random.getstate(),'numpy_rng':np_rng,'history':history,
                 'best_accuracy':best_accuracy,'best_epoch':best_epoch,'fingerprints':fingerprints}
        torch.save(state,state_path)
        atomic_json(history_path,{'candidate':candidate,'fingerprints':fingerprints,
                                  'history':history,'best_epoch':best_epoch,'best_validation_accuracy':best_accuracy,
                                  'best_checkpoint_sha256':sha256(best_path),
                                  'runtime_seconds_since_start':time.perf_counter()-started,
                                  'test_access':'SEALED','status':'COMPLETE' if epoch==epochs else 'IN_PROGRESS'})
        print(f'{cid} seed={seed} epoch={epoch}/{epochs} train_loss={loss_total/seen:.6f} val={accuracy:.4f}% best={best_accuracy:.4f}%')
    print('COMPLETE:',history_path)


def main():
    parser=argparse.ArgumentParser(description='CAJO Step 2 unified experiment runner')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true',help='Read-only readiness and methods/results mapping')
    mode.add_argument('--run',action='store_true',help='Run one authorized candidate')
    mode.add_argument('--resume',action='store_true',help='Resume one authorized candidate')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--candidate',type=Path)
    args=parser.parse_args()
    config,config_path=load_config(args.root)
    if args.plan:
        print('CAJO STEP 2: READ_ONLY_PLAN')
        print('Config:',config_path)
        print('Config SHA256:',sha256(config_path))
        print('Methodology: 2.1 constraints; 2.2 candidate/KD; 2.3 evaluation, feasibility, Pareto')
        print('Outputs: baseline, compression ablation, sequential-vs-joint, hardware, Pareto, frozen selection, held-out')
        blockers=run_gate(config)
        print('Full-run status:', 'GATE_OPEN' if not blockers else 'BLOCKED')
        for b in blockers:print(' -',b)
        print('No datasets, GPU, training, checkpoints, or result files accessed/written.')
        return
    if args.candidate is None or not args.candidate.is_file():
        parser.error('--run/--resume requires --candidate existing JSON file')
    # Fail closed before loading candidate or importing GPU stack.
    blockers=run_gate(config)
    if blockers:
        raise SystemExit('RUN BLOCKED: '+'; '.join(blockers))
    candidate=read_json(args.candidate)
    candidate_contract(candidate,config)
    fit(args.root,config,config_path,candidate,args.candidate,resume=args.resume)

if __name__=='__main__':
    main()
