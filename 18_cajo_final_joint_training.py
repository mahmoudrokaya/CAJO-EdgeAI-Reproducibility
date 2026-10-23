#!/usr/bin/env python3
"""CAJO Step 18: resumable, sealed-test FP32 multi-architecture training.

Commands (from Codes/):
  python 18_cajo_final_joint_training.py plan --root ..
  python 18_cajo_final_joint_training.py authorize --root .. --confirm APPROVE_ALTERNATIVE_FP32_TRAINING
  python 18_cajo_final_joint_training.py run-all --root .. --confirm START_ALTERNATIVE_FP32_TRAINING

Scope: train each declared alternative from scratch (P0 CE), then physically
prune its validation-selected trained weights at P25/P50, recover with CE/KD.
Three seeds, 20 epochs each. No INT8, no sealed test, no claims of complete
matched-budget joint NAS or Pareto feasibility. Frozen baselines unchanged.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path

CONFIG=Path('Reports/Optimization_Protocol/Final_Experiment_Config/cajo_experiment_config.json')
RUNNER=Path('Codes/02_experiment_runner.py')
ADAPTER=Path('Codes/04_compression_recovery.py')
ARCH=Path('Codes/03_architecture_search_space.py')
BASE=Path('Reports/CAJO_Experiments')
OUT=Path('Reports/Optimization_Protocol/Alternative_FP32_Training')
LOCKED='56361ba3ef2cb530d54a923cf4f5b3758a22bee82a9119822395cef0be9bbdc5'
ALT={'CIFAR100':'ResNet18_width075','CIFAR10':'ResNet8_width075','SpeechCommands_v0.02':'DSCNN_width075'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p)
    if not spec or not spec.loader:raise RuntimeError('Cannot import '+str(p))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def write_once(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if read(p)!=obj:raise RuntimeError('Immutable file differs: '+str(p))
        return
    tmp=p.with_name(p.name+'.partial')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def stages(config):
    result=[]
    for task in sorted(ALT):
        arch=ALT[task]
        if arch not in [x['id'] for x in config['tasks'][task]['architectures']]:raise RuntimeError('Alternative architecture not declared: '+arch)
        for seed in config['recovery']['seeds']:
            for ratio,kd in [(0.0,False),(0.25,False),(0.25,True),(0.5,False),(0.5,True)]:
                pre=(ratio==0)
                cid=f'{task}__{arch}__'+('pretrain_P0_CE' if pre else f'joint_P{int(ratio*100)}_'+('KD' if kd else 'CE'))+f'__s{seed}'
                result.append(dict(candidate_id=cid,task=task,architecture=arch,pruning_ratio=ratio,
                    precision='FP32',kd=kd,strategy='joint',seed=seed,epochs=config['recovery']['epochs'],
                    mode='architecture_training',operation_order=['architecture_training'] if pre else ['architecture_training','pruning']+(['KD'] if kd else ['CE']),
                    pruning_policy='importance'))
    return result

def state(root):
    cp=root/CONFIG
    if sha(cp)!=LOCKED:raise RuntimeError('Locked config SHA256 mismatch')
    config=read(cp)
    if config['status']!='LOCKED' or config['approval'].get('human_approved') is not True:raise RuntimeError('Config not approved/locked')
    if config['approval'].get('full_training_authorized') is not False:raise RuntimeError('Expected unmodified locked config')
    if config['selection'].get('test_access')!='sealed_until_selection_frozen':raise RuntimeError('Test firewall changed')
    if config['provenance'].get('baseline_retraining_permitted') is not False:raise RuntimeError('Baseline protection missing')
    if config['claims_scope'].get('energy_policy')!='MEASURED_DEVICE_LEVEL':raise RuntimeError('Energy policy drift')
    for p in (root/RUNNER,root/ADAPTER,root/ARCH):
        if not p.is_file():raise FileNotFoundError(p)
    runner=load(root/RUNNER,'cajo_step18_runner')
    allowed={'Manuscript energy alignment pending','Full training not authorized'}
    remaining=[e for e in runner.run_gate(config) if e not in allowed]
    if remaining:raise RuntimeError('Runner gates: '+'; '.join(remaining))
    candidates=stages(config)
    for c in candidates:runner.candidate_contract(c,config)
    return config,runner,candidates

def authorization(root,candidates):
    return dict(schema='cajo_alternative_fp32_training_scope_v1',locked_config_sha256=LOCKED,
        source_sha256={str(p):sha(root/p) for p in (RUNNER,ADAPTER,ARCH)},
        scope='ALTERNATIVE_ARCHITECTURE_PRETRAIN_AND_P25_P50_FP32_CE_KD_ONLY',
        no_test_access=True,baseline_retraining=False,quantization=False,
        full_cajo_feasibility_claim=False,
        limitations=['No INT8 training/deployment in this step','No full-runtime-memory feasibility',
                     'Not a matched-budget 5-epoch screening search','Not a complete four-dimensional joint NAS'],
        candidate_ids=[c['candidate_id'] for c in candidates])

def verify_auth(root,candidates):
    p=root/OUT/'authorization.json'
    if not p.is_file() or read(p)!=authorization(root,candidates):raise RuntimeError('Authorization missing or source drift; use authorize after review')

def path_for(root,c):
    return root/BASE/c['task']/c['strategy']/c['candidate_id']/('seed_'+str(c['seed']))

def run_candidate(root,config,runner,c,all_candidates):
    verify_auth(root,all_candidates)
    out=path_for(root,c);history=out/'history.json';best=out/'best_model.pt'
    if history.is_file():
        h=read(history)
        if h.get('status')=='COMPLETE':
            if h.get('test_access')!='SEALED' or h.get('candidate')!=c or not best.is_file() or h.get('best_checkpoint_sha256')!=sha(best):
                raise RuntimeError('Completed experiment provenance invalid: '+c['candidate_id'])
            print('SKIP COMPLETE:',c['candidate_id'],flush=True);return
    candidate_path=root/OUT/'candidates'/(c['candidate_id']+'.json')
    write_once(candidate_path,c)
    old_gate=runner.run_gate
    old_import=runner.import_adapter
    def scoped_gate(d):
        if d is not config:raise RuntimeError('Different configuration object')
        verify_auth(root,all_candidates)
        other=[x for x in old_gate(d) if x not in ('Manuscript energy alignment pending','Full training not authorized')]
        if other:raise RuntimeError('Runner gate: '+'; '.join(other))
        return []
    runner.run_gate=scoped_gate
    if c['pruning_ratio']>0:
        prior=next(x for x in all_candidates if x['task']==c['task'] and x['seed']==c['seed'] and x['pruning_ratio']==0)
        prior_out=path_for(root,prior);prior_history=prior_out/'history.json';prior_best=prior_out/'best_model.pt'
        if not prior_history.is_file() or read(prior_history).get('status')!='COMPLETE':raise RuntimeError('Pretrained source not complete')
        if read(prior_history).get('best_checkpoint_sha256')!=sha(prior_best):raise RuntimeError('Pretrained checkpoint digest mismatch')
        adapter=load(root/ADAPTER,'cajo_step18_adapter')
        def prepared(*,root,config,candidate,device):
            import torch
            source_arch,data,surgery,importance=adapter.sources(root)
            splits,manifest=adapter.verified_manifest(root,config,candidate['task'],data)
            teacher,baseline=adapter.frozen_teacher(root,config,candidate['task'],source_arch,torch)
            student=source_arch.build_architecture(candidate['task'],candidate['architecture'],config,root)
            blob=torch.load(prior_best,map_location='cpu',weights_only=True)
            if blob.get('candidate')!=prior:raise RuntimeError('Source checkpoint candidate mismatch')
            student.load_state_dict(blob['model'],strict=True)
            details=adapter.physical_prune(student,candidate['task'],candidate['pruning_ratio'],importance,surgery,
                source_arch,'importance',candidate['seed'],root)
            train,val=adapter.make_data(root,candidate['task'],splits,config,candidate['seed'],data)
            provenance=dict(test_access='sealed',frozen_manifest_sha256=sha(manifest),
                frozen_teacher_checkpoint_sha256=sha(baseline),architecture=candidate['architecture'],
                operation_order=candidate['operation_order'],initialization='validation_selected_trained_alternative',
                pretrained_checkpoint_sha256=sha(prior_best),pruning_details=details,
                student_parameters=sum(p.numel() for p in student.parameters()),precision_training='FP32')
            return dict(student=student,teacher=teacher if candidate['kd'] else None,
                train_epoch_batches=train,validation_batches=val,provenance=provenance)
        class AdapterProxy:
            prepare_candidate=staticmethod(prepared)
        runner.import_adapter=lambda root:(AdapterProxy,root/ADAPTER)
    try:
        resume=(out/'resume_state.pt').is_file()
        print('TRAIN',c['candidate_id'],'RESUME' if resume else 'NEW',flush=True)
        runner.fit(root,config,root/CONFIG,c,candidate_path,resume=resume)
    finally:
        runner.run_gate=old_gate;runner.import_adapter=old_import

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['plan','authorize','run-one','run-all'])
    ap.add_argument('--root',type=Path,default=Path('..'))
    ap.add_argument('--confirm');ap.add_argument('--candidate-id')
    a=ap.parse_args();root=a.root.resolve()
    config,runner,candidates=state(root)
    if a.action=='plan':
        print(json.dumps(dict(config_sha256=LOCKED,candidate_count=len(candidates),
            pretrained_architectures=3,pruned_recovery_runs=36,pretraining_runs=9,
            candidates=[dict(id=c['candidate_id'],complete=(path_for(root,c)/'history.json').is_file()) for c in candidates],
            limitations=authorization(root,candidates)['limitations'],training_started=False),indent=2));return
    if a.action=='authorize':
        if a.confirm!='APPROVE_ALTERNATIVE_FP32_TRAINING':raise RuntimeError('Authorization confirmation missing')
        write_once(root/OUT/'authorization.json',authorization(root,candidates))
        print('SCOPE AUTHORIZED. No training started.');return
    if a.confirm!='START_ALTERNATIVE_FP32_TRAINING':raise RuntimeError('Training confirmation missing')
    verify_auth(root,candidates)
    if a.action=='run-one':
        matches=[c for c in candidates if c['candidate_id']==a.candidate_id]
        if len(matches)!=1:raise RuntimeError('Unknown --candidate-id')
        run_candidate(root,config,runner,matches[0],candidates);return
    for c in candidates:
        # Fresh runner for each experiment, preserving original module behavior.
        config,runner,current=state(root)
        run_candidate(root,config,runner,c,current)
    print('ALTERNATIVE FP32 TRAINING FINISHED; test remains sealed.')

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('CAJO STOPPED:',type(e).__name__+':',e,file=sys.stderr);sys.exit(2)
