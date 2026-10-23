#!/usr/bin/env python3
"""CAJO Step 4: physical pruning, frozen-teacher/data adapter, train-only INT8 PTQ.

Read-only: python 04_compression_recovery.py --inspect
Runner interface: prepare_candidate(root=..., config=..., candidate=..., device=...)
PTQ interface: quantize_recovered_model(model, *, task, root, config, seed=42)

This module NEVER trains, accesses test examples, writes checkpoints, or changes
frozen baselines. The Step-2 runner owns CE/KD training and checkpoint selection.
Quantization must be called AFTER FP32 recovery, never as trainable INT8.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
MANIFEST={'CIFAR100':'cifar100_split_manifest.csv','CIFAR10':'cifar10_split_manifest.csv',
          'SpeechCommands_v0.02':'speech_commands_split_manifest.csv'}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def import_file(path,name):
    if not path.is_file():raise FileNotFoundError(path)
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise RuntimeError('Cannot import '+str(path))
    module=importlib.util.module_from_spec(spec)
    old_cuda=os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        spec.loader.exec_module(module)
    finally:
        if old_cuda is None:os.environ.pop("CUDA_VISIBLE_DEVICES",None)
        else:os.environ["CUDA_VISIBLE_DEVICES"]=old_cuda
    return module


def sources(root):
    root=Path(root)
    return (import_file(root/'Codes'/'03_architecture_search_space.py','cajo_step3'),
            import_file(root/'Codes'/'03_train_baseline_models.py','cajo_baseline_data'),
            import_file(root/'Codes'/'06e_verify_structured_pruning.py','cajo_channel_surgery'),
            import_file(root/'Codes'/'06f3_verify_importance_pruning.py','cajo_gamma_pruning'))


def verified_manifest(root,config,task,loader):
    path=Path(root)/'Reports'/'Dataset_Verification'/MANIFEST[task]
    expected=config['tasks'][task]['split_manifest_sha256']
    if sha256(path)!=expected:raise RuntimeError('Frozen split manifest SHA256 mismatch: '+task)
    rows=loader.read_manifest(path)
    splits={'train':[],'validation':[]}
    for row in rows:
        if row['split'] in splits:splits[row['split']].append(row)
        elif row['split']!='test':raise RuntimeError('Unexpected manifest split')
    if not all(splits.values()):raise RuntimeError('Missing train/validation partition')
    return splits,path


def frozen_teacher(root,config,task,architecture_module,torch):
    entry=config['tasks'][task]
    checkpoint=Path(root)/Path(entry['frozen_checkpoint'].replace('\\','/'))
    if sha256(checkpoint)!=entry['checkpoint_sha256']:
        raise RuntimeError('Frozen checkpoint SHA256 mismatch: '+task)
    # Only load locally generated trusted checkpoints. TorchVersion is present in
    # original PyTorch metadata and is allowlisted for weights_only=True.
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        blob=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not isinstance(blob,dict) or 'model' not in blob:
        raise RuntimeError('Unexpected frozen checkpoint structure')
    model=architecture_module.build_architecture(task,'baseline',config,root)
    model.load_state_dict(blob['model'],strict=True)
    model.eval().requires_grad_(False)
    return model,checkpoint


def physical_prune(model,task,ratio,importance,surgery,architecture_module,policy='importance',seed=42,root=ROOT):
    """Remove real channels; never mask weights. Random control is deterministic."""
    if ratio==0:return []
    if ratio not in (0.25,0.5):raise ValueError('Undeclared pruning ratio')
    if policy not in ('importance','random'):raise ValueError('Unsupported pruning policy')
    import torch
    import torch.nn as nn
    if policy=='importance':
        # Existing audited pruning supports frozen baseline BasicBlocks and DS-CNN.
        if task=='SpeechCommands_v0.02':
            details=importance.prune_dscnn(model,ratio,surgery)
        else:
            # Same audited BN-gamma ranking for both original and Step-3
            # alternative residual blocks (which use a different Python class).
            details=[]
            for name,b in model.named_modules():
                if not (hasattr(b,'conv1') and hasattr(b,'bn1') and hasattr(b,'conv2')
                        and hasattr(b,'bn2') and isinstance(b.conv1,nn.Conv2d)
                        and isinstance(b.conv2,nn.Conv2d)):
                    continue
                old=b.conv1.out_channels
                scores=importance.normalized_gamma(b.bn1)
                keep=max(1,int(round(old*(1-ratio))))
                idx=sorted(importance.ranking(scores)[:keep])
                b.conv1=surgery.conv_subset(b.conv1,out_idx=idx)
                b.bn1=surgery.bn_subset(b.bn1,idx)
                b.conv2=surgery.conv_subset(b.conv2,in_idx=idx)
                details.append({'block':name,'channels_before':old,
                                'channels_after':keep,'retained_indices':idx,
                                'score_rule':'abs(bn1.gamma)/mean(abs(bn1.gamma))',
                                'residual_output_channels_unchanged':b.conv2.out_channels})
            if not details:raise RuntimeError('No prunable residual blocks found')
        return details
    generator=torch.Generator().manual_seed(int(seed))
    def pick(n):
        keep=max(1,int(round(n*(1-ratio))))
        return sorted(torch.randperm(n,generator=generator)[:keep].tolist())
    if task=='SpeechCommands_v0.02':
        width=model.features[0].out_channels
        idx=pick(width)
        model.features[0]=surgery.conv_subset(model.features[0],out_idx=idx)
        model.features[1]=surgery.bn_subset(model.features[1],idx)
        for i in range(3,7):
            block=model.features[i].layers
            block[0]=surgery.conv_subset(block[0],in_idx=idx,out_idx=idx,depthwise=True)
            block[1]=surgery.bn_subset(block[1],idx)
            block[3]=surgery.conv_subset(block[3],in_idx=idx,out_idx=idx)
            block[4]=surgery.bn_subset(block[4],idx)
        model.fc=surgery.linear_subset(model.fc,idx)
        return [{'group':'linked_dscnn_channels','retained_indices':idx,'policy':'random'}]
    details=[]
    for name,b in model.named_modules():
        if not (hasattr(b,'conv1') and hasattr(b,'bn1') and hasattr(b,'conv2') and hasattr(b,'bn2')
                and isinstance(b.conv1,nn.Conv2d) and isinstance(b.conv2,nn.Conv2d)):
            continue
        idx=pick(b.conv1.out_channels)
        b.conv1=surgery.conv_subset(b.conv1,out_idx=idx)
        b.bn1=surgery.bn_subset(b.bn1,idx)
        b.conv2=surgery.conv_subset(b.conv2,in_idx=idx)
        details.append({'block':name,'retained_indices':idx,'policy':'random'})
    if not details:raise RuntimeError('No prunable residual blocks found')
    return details


def make_data(root,task,splits,config,seed,baseline_loader):
    """Reuse frozen baseline transforms and labels; never instantiate test partition."""
    import torch
    data_root=Path(root)/'Data'
    if task=='SpeechCommands_v0.02':
        report=json.loads((Path(root)/'Reports'/'Dataset_Verification'/'dataset_verification_report.json').read_text(encoding='utf-8'))
        labels=report['datasets'][task]['class_names']
        train=baseline_loader.SpeechManifestDataset(splits['train'],data_root,labels)
        val=baseline_loader.SpeechManifestDataset(splits['validation'],data_root,labels)
    else:
        train=baseline_loader.CIFARManifestDataset(splits['train'],data_root,task,True,seed)
        val=baseline_loader.CIFARManifestDataset(splits['validation'],data_root,task,False,seed)
    batch=int(config['recovery']['batch_size'])
    def train_epoch_batches(epoch):
        # Deterministic per-epoch shuffle. Workers=0 keeps augmentation RNG local.
        generator=torch.Generator().manual_seed(int(seed)+int(epoch)*1009)
        return torch.utils.data.DataLoader(train,batch_size=batch,shuffle=True,
                                           num_workers=0,generator=generator)
    def validation_batches():
        return torch.utils.data.DataLoader(val,batch_size=batch,shuffle=False,num_workers=0)
    return train_epoch_batches,validation_batches


def prepare_candidate(*,root,config,candidate,device):
    """Step-2 adapter: build FP32 student, verify teacher and manifest, seal test."""
    import torch
    root=Path(root)
    task=candidate['task'];architecture=candidate['architecture']
    ratio=float(candidate['pruning_ratio']);seed=int(candidate['seed'])
    if candidate['precision']!='FP32':
        raise ValueError('INT8 is post-recovery inference only; FP32 training required')
    if candidate['mode']=='architecture_training' and architecture=='baseline':
        raise PermissionError('Frozen baseline retraining prohibited')
    arch,data,surgery,importance=sources(root)
    splits,manifest_path=verified_manifest(root,config,task,data)
    teacher,checkpoint=frozen_teacher(root,config,task,arch,torch)
    student=arch.build_architecture(task,architecture,config,root)
    if architecture=='baseline':
        student.load_state_dict(teacher.state_dict(),strict=True)
        initialization='frozen_checkpoint_copy'
    else:
        initialization='random_initialization_new_architecture'
        if candidate['mode']!='architecture_training':
            raise ValueError('Alternative architecture must be trained from scratch before recovery; no compatible pretrained checkpoint')
    policy=candidate.get('pruning_policy','importance')
    if policy=='random' and not config['techniques'].get('random_pruning_control',False):
        raise ValueError('Random-pruning control disabled by configuration')
    if architecture!='baseline' and ratio>0:
        raise RuntimeError('Alternate architecture requires an independently trained FP32 source checkpoint before meaningful pruning; Step 4 refuses pruning random initial weights')
    details=physical_prune(student,task,ratio,importance,surgery,arch,policy,seed,root)
    student.train()
    train_factory,val_factory=make_data(root,task,splits,config,seed,data)
    provenance={
        'test_access':'sealed',
        'frozen_manifest_sha256':sha256(manifest_path),
        'frozen_teacher_checkpoint_sha256':sha256(checkpoint),
        'architecture':architecture,
        'operation_order':candidate['operation_order'],
        'initialization':initialization,
        'pruning_policy':policy,
        'pruning_ratio':ratio,
        'pruning_details':details,
        'student_parameters':sum(p.numel() for p in student.parameters()),
        'teacher_frozen':True,
        'data_splits_loaded':['train','validation'],
        'precision_training':'FP32',
    }
    return {'student':student,'teacher':teacher if candidate['kd'] else None,
            'train_epoch_batches':train_factory,'validation_batches':val_factory,
            'provenance':provenance}


def quantize_recovered_model(model,*,task,root,config,seed=42):
    """Real train-only static INT8 PTQ on CPU; returns deployable model + provenance.

    Call ONLY on a trained, recovered FP32 checkpoint selected by validation.
    This function never reads validation/test examples and never benchmarks latency.
    """
    import torch
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx,convert_fx
    from torch.ao.quantization.backend_config import get_onednn_backend_config
    if 'onednn' not in torch.backends.quantized.supported_engines:
        raise RuntimeError('oneDNN INT8 engine unsupported')
    arch,data,_,_=sources(Path(root))
    splits,manifest=verified_manifest(root,config,task,data)
    count=int(config['techniques']['quantization_calibration_train_only_count'])
    if count!=256:raise RuntimeError('Calibration count changed from approved 256')
    import random
    rng=random.Random(42)
    selected=rng.sample(splits['train'],count)
    if task=='SpeechCommands_v0.02':
        report=json.loads((Path(root)/'Reports'/'Dataset_Verification'/'dataset_verification_report.json').read_text(encoding='utf-8'))
        ds=data.SpeechManifestDataset(selected,Path(root)/'Data',report['datasets'][task]['class_names'])
    else:
        ds=data.CIFARManifestDataset(selected,Path(root)/'Data',task,False,seed)
    model=model.cpu().eval()
    torch.backends.quantized.engine='onednn'
    shape=arch.SHAPES[task]
    example=torch.zeros(shape)
    qconfig=get_default_qconfig_mapping('onednn')
    prepared=prepare_fx(model,qconfig,(example,),backend_config=get_onednn_backend_config())
    prepared.eval()
    with torch.inference_mode():
        for x,_ in torch.utils.data.DataLoader(ds,batch_size=1,shuffle=False,num_workers=0):
            prepared(x)
    quantized=convert_fx(prepared,backend_config=get_onednn_backend_config()).eval()
    with torch.inference_mode():
        y=quantized(example)
    if y.shape!=(1,arch.CLASSES[task]) or not bool(torch.isfinite(y).all()):
        raise RuntimeError('Quantized candidate failed compatibility check')
    return quantized,{'precision':'INT8','engine':'onednn','calibration_split':'train',
                      'calibration_examples':count,'calibration_seed':42,
                      'manifest_sha256':sha256(manifest),'test_access':'sealed',
                      'runtime_compatibility_probe':'single_forward_pass_only'}


def main():
    ap=argparse.ArgumentParser(description='CAJO Step 4 compression/recovery adapter')
    ap.add_argument('--inspect',action='store_true',required=True)
    ap.add_argument('--root',type=Path,default=ROOT)
    args=ap.parse_args()
    config_path=args.root/'Reports'/'Optimization_Protocol'/'Final_Experiment_Config'/'cajo_experiment_config.json'
    if not config_path.is_file():raise SystemExit('Missing approved draft configuration: '+str(config_path))
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('schema')!='cajo_final_experiment_config_v1':raise SystemExit('Unexpected config schema')
    print('CAJO STEP 4: READ_ONLY_INSPECTION')
    print('Config SHA256:',sha256(config_path))
    print('Methods: 2.2 candidate (architecture, pruning, precision, KD), train-only PTQ, ordered strategies')
    print('Adapter: prepare_candidate; INT8: quantize_recovered_model')
    print('Training: Step 2 CE/KD runner; hardware: Step 5; search: Step 6; selection: Step 7')
    print('Pruning: audited BN-gamma ranking and physical channel surgery, including alternate residual blocks')
    print('Full-run authorization: unchanged; no dataset, checkpoint, GPU, or model accessed')

if __name__=='__main__':main()
