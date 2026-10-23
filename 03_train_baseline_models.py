#!/usr/bin/env python3
"""CAJO Phase 3: faithful baseline training for the Phase-2 model definitions.

Examples (Windows PowerShell, from New_Experiments/Codes):
 python 03_train_baseline_models.py --dataset CIFAR100 --mode smoke
 python 03_train_baseline_models.py --dataset CIFAR100 --mode train --epochs 100
 python 03_train_baseline_models.py --dataset CIFAR100 --mode train --epochs 100 --resume
 python 03_train_baseline_models.py --dataset CIFAR100 --mode test

Training and model selection use only train/validation. Test requires a complete
training run and is a separate explicit command. No deployment metrics are claimed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import random
import sys
import time
import wave as wave_module
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

BASE = Path(r'D:\48\481\New Papers\Toward Lightweight Edge Intelligence\Related_Files\New_Experiments')
DATASETS = {'CIFAR100': ('cifar100_split_manifest.csv',100), 'CIFAR10':('cifar10_split_manifest.csv',10), 'SpeechCommands_v0.02':('speech_commands_split_manifest.csv',35)}
NORMALIZATION = {
 'CIFAR100': ((.5071,.4867,.4408),(.2675,.2565,.2761)),
 'CIFAR10': ((.4914,.4822,.4465),(.247,.2435,.2616))}


def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def json_write(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    tmp.replace(path)


def read_manifest(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def seed_worker(_):
    s=torch.initial_seed()%2**32
    np.random.seed(s); random.seed(s)


def phase2_module(script_dir):
    path=script_dir/'02_prepare_baseline_models.py'
    if not path.exists():
        alternative=script_dir/'02_prepare_baseline_models_regenerated.py'
        if alternative.exists(): path=alternative
        else: raise FileNotFoundError('Phase-2 architecture source missing beside this script: '+str(path))
    spec=importlib.util.spec_from_file_location('cajo_phase2_architectures',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module,path


def build_model(module,dataset):
    if dataset=='CIFAR100': return module.CIFARResNet((2,2,2,2),100)
    if dataset=='CIFAR10': return module.ResNet8(10)
    return module.DSCNN(num_classes=35,width=64)


class CIFARManifestDataset(Dataset):
    """Load trusted local CIFAR pickle files, indexed by exact Phase-1 manifest records."""
    def __init__(self,rows,data_root,name,train,seed):
        self.rows=rows; self.train=train; self.seed=seed
        self.mean=torch.tensor(NORMALIZATION[name][0],dtype=torch.float32).view(3,1,1)
        self.std=torch.tensor(NORMALIZATION[name][1],dtype=torch.float32).view(3,1,1)
        sources={}
        for rel in sorted({r['source_file_relative_to_data'] for r in rows}):
            path=data_root/rel
            if not path.is_file(): raise FileNotFoundError(path)
            # CIFAR's official serialized archives use pickle. Load ONLY trusted original files.
            with path.open('rb') as f: obj=pickle.load(f,encoding='bytes')
            arr=obj[b'data']; labels=obj[b'fine_labels'] if name=='CIFAR100' else obj[b'labels']
            if arr.ndim!=2 or arr.shape[1]!=3072 or len(labels)!=len(arr):
                raise ValueError('Unexpected CIFAR source format: '+str(path))
            sources[rel]=(arr,np.asarray(labels))
        self.images=np.empty((len(rows),3,32,32),dtype=np.uint8)
        self.labels=np.empty(len(rows),dtype=np.int64)
        for i,r in enumerate(rows):
            rel=r['source_file_relative_to_data']; idx=int(r['record_index_in_file'])
            arr,labs=sources[rel]
            if not 0<=idx<len(labs): raise ValueError('Invalid manifest record index')
            label=int(r['class_id'])
            if int(labs[idx])!=label: raise ValueError('Manifest label mismatch at '+rel+':'+str(idx))
            self.images[i]=arr[idx].reshape(3,32,32)
            self.labels[i]=label
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        x=torch.from_numpy(self.images[i].copy()).float().div_(255)
        if self.train:
            x=torch.nn.functional.pad(x,(4,4,4,4))
            top=int(torch.randint(0,9,(1,)).item()); left=int(torch.randint(0,9,(1,)).item())
            x=x[:,top:top+32,left:left+32]
            if bool(torch.randint(0,2,(1,)).item()): x=torch.flip(x,(-1,))
        return (x-self.mean)/self.std, int(self.labels[i])


class SpeechManifestDataset(Dataset):
    """35-way speech task: mono 16-kHz, 1 s, 40 log-mel bins, no silence remapping."""
    def __init__(self,rows,data_root,labels):
        import torchaudio
        self.rows=rows; self.root=data_root; self.label_to_id={v:i for i,v in enumerate(labels)}
        self.mel=torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=400,
                 win_length=400,hop_length=160,n_mels=40,center=True,power=2.0)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; path=self.root/r['wav_path_relative_to_data']
        # Speech Commands v0.02 is expected to contain mono 16-kHz PCM16 WAV.
        # Decode with the Python standard library to avoid TorchCodec native DLLs.
        with wave_module.open(str(path), 'rb') as f:
            sr=f.getframerate()
            if (f.getnchannels()!=1 or f.getsampwidth()!=2 or sr!=16000
                    or f.getcomptype()!='NONE'):
                raise ValueError('Unsupported Speech Commands WAV format: '+str(path))
            pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').copy()
        wave=torch.from_numpy(pcm.astype(np.float32)/32768.0).unsqueeze(0)
        wave=wave.mean(dim=0,keepdim=True)
        wave=wave[:,:16000]
        if wave.shape[1]<16000: wave=torch.nn.functional.pad(wave,(0,16000-wave.shape[1]))
        feat=torch.log(self.mel(wave).clamp_min(1e-6))
        return feat,self.label_to_id[r['original_word_label']]


def build_dataloader(ds,split,batch_size,workers,seed,limit=0):
    if limit: ds=torch.utils.data.Subset(ds,range(min(limit,len(ds))))
    generator=torch.Generator().manual_seed(seed+{'train':0,'validation':1,'test':2}[split])
    kwargs=dict(dataset=ds,batch_size=batch_size,shuffle=(split=='train'),num_workers=workers,
                worker_init_fn=seed_worker,generator=generator,pin_memory=torch.cuda.is_available())
    if workers: kwargs['persistent_workers']=False
    return DataLoader(**kwargs)


def epoch_run(model,loader,optimizer,scaler,device,amp,train,limit_batches=0):
    model.train(train)
    total=0; correct=0; loss_sum=0.
    start=time.perf_counter()
    for b,(x,y) in enumerate(loader):
        if limit_batches and b>=limit_batches: break
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type,enabled=amp,dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
                pred=model(x); loss=nn.functional.cross_entropy(pred,y)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
        n=y.numel(); total+=n; correct+=(pred.argmax(1)==y).sum().item(); loss_sum+=loss.item()*n
    if device.type=='cuda': torch.cuda.synchronize()
    return {'loss':loss_sum/total if total else None,'accuracy':correct/total if total else None,
            'samples':total,'elapsed_seconds':round(time.perf_counter()-start,3)}


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])
    if s['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(s['cuda'])


def save_checkpoint(path,blob):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    torch.save(blob,tmp); tmp.replace(path)


def load_checkpoint(path,device):
    # Checkpoints must be generated locally by this project; never load untrusted files.
    return torch.load(path,map_location=device,weights_only=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=DATASETS,default='CIFAR100')
    ap.add_argument('--mode',choices=['smoke','train','test'],default='smoke')
    ap.add_argument('--base',type=Path,default=BASE)
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--lr',type=float,default=.1)
    ap.add_argument('--momentum',type=float,default=.9)
    ap.add_argument('--weight-decay',type=float,default=.0005)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--smoke-batches',type=int,default=2)
    ap.add_argument('--allow-cpu',action='store_true')
    args=ap.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.workers<0: ap.error('epochs and batch-size must be positive; workers nonnegative')
    if args.mode!='train' and args.resume: ap.error('--resume applies only to --mode train')
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError('CUDA unavailable. Check CUDA PyTorch installation, or pass --allow-cpu intentionally.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all(args.seed)
    report_root=args.base/'Reports'/'Dataset_Verification'
    phase1=json.loads((report_root/'dataset_verification_report.json').read_text(encoding='utf-8'))
    phase2=json.loads((args.base/'Reports'/'Baseline_Preparation'/'baseline_preparation_report.json').read_text(encoding='utf-8'))
    if phase1.get('overall_status')!='PASS' or phase2.get('status')!='PASS': raise ValueError('Phase 1/2 reports must PASS')
    manifest_name,nclasses=DATASETS[args.dataset]
    manifest=report_root/manifest_name
    digest=hash_file(manifest)
    if digest!=phase1['datasets'][args.dataset]['manifest_sha256'] or digest!=phase2['datasets'][args.dataset]['sha256']:
        raise ValueError('Phase-1/2 manifest hash mismatch; stop to avoid data drift')
    rows=read_manifest(manifest)
    expected=phase1['datasets'][args.dataset]['split_counts']
    actual=Counter(r['split'] for r in rows)
    if dict(actual)!=expected: raise ValueError(f'Split counts changed: {actual} vs {expected}')
    script_dir=Path(__file__).resolve().parent
    module,arch_path=phase2_module(script_dir)
    model=build_model(module,args.dataset).to(device)
    model_key={'CIFAR100':'CIFAR100_ResNet18','CIFAR10':'CIFAR10_ResNet8','SpeechCommands_v0.02':'SpeechCommands_DSCNN35'}[args.dataset]
    if sum(p.numel() for p in model.parameters())!=phase2['models'][model_key]['parameter_count']:
        raise ValueError('Architecture parameter count differs from Phase 2')
    arch_hash=hash_file(arch_path)
    amp=device.type=='cuda'
    out=args.base/'Reports'/'Baseline_Training'/args.dataset
    out.mkdir(parents=True,exist_ok=True)
    config={'dataset':args.dataset,'manifest_sha256':digest,'architecture_sha256':arch_hash,
            'seed':args.seed,'epochs':args.epochs,'batch_size':args.batch_size,'workers':args.workers,
            'optimizer':'SGD','lr':args.lr,'momentum':args.momentum,'weight_decay':args.weight_decay,
            'scheduler':'CosineAnnealingLR(epoch-level)','amp':amp,'device':str(device),
            'torch':torch.__version__,'cuda_version':torch.version.cuda,'num_classes':nclasses,
            'label_policy':'35 original words' if args.dataset.startswith('Speech') else 'original labels',
            'selection':'max validation accuracy, tie break min validation loss',
            'test_policy':'separate explicit test command only after complete training',
            'training_preprocessing':'CIFAR pad4 random crop32 horizontal flip + normalization' if not args.dataset.startswith('Speech') else '16kHz mono 1s 40 log-mel, no augmentation',
            'evaluation_preprocessing':'deterministic, no augmentation'}
    # Workers do not change the scientific hyperparameters but are recorded.
    config_id=hashlib.sha256(json.dumps({k:v for k,v in config.items() if k not in ('workers',)},sort_keys=True).encode()).hexdigest()
    config['config_id']=config_id
    if args.mode=='smoke':
        print('SMOKE ONLY: two short train/validation batches; no checkpoints or test evaluation')
        split_rows={s:[r for r in rows if r['split']==s][:args.batch_size*args.smoke_batches] for s in ('train','validation')}
    elif args.mode=='test':
        split_rows={'test':[r for r in rows if r['split']=='test']}
    else:
        split_rows={s:[r for r in rows if r['split']==s] for s in ('train','validation')}
    if args.dataset.startswith('Speech'):
        labels=phase1['datasets'][args.dataset]['class_names']
        ds={s:SpeechManifestDataset(rr,args.base/'Data',labels) for s,rr in split_rows.items()}
    else:
        ds={s:CIFARManifestDataset(rr,args.base/'Data',args.dataset,s=='train',args.seed) for s,rr in split_rows.items()}
    loader={s:build_dataloader(d,s,args.batch_size,args.workers,args.seed) for s,d in ds.items()}
    optimizer=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    scaler=torch.amp.GradScaler('cuda') if amp else None
    print(f'CAJO Phase 3 | {args.dataset} | {args.mode} | {device} | {torch.__version__} | params={sum(p.numel() for p in model.parameters()):,}')
    if args.mode=='smoke':
        a=epoch_run(model,loader['train'],optimizer,scaler,device,amp,True,args.smoke_batches)
        b=epoch_run(model,loader['validation'],optimizer,scaler,device,amp,False,args.smoke_batches)
        result={'status':'SMOKE_PASS','train':a,'validation':b,'configuration':config,
                'warning':'Smoke results are NOT experimental results and must not be reported as accuracy.'}
        json_write(out/'smoke_test_report.json',result)
        print('Smoke passed:',result['train'],result['validation']); print('Report:',out/'smoke_test_report.json')
        return
    best_path=out/'best_model.pt'; last_path=out/'last_checkpoint.pt'
    if args.mode=='test':
        if not best_path.exists() or not last_path.exists(): raise FileNotFoundError('Complete training checkpoints not found; run --mode train first')
        last=load_checkpoint(last_path,device)
        if last['epoch']!=last['config']['epochs'] or not last.get('training_complete',False):
            raise RuntimeError('Training not complete. Refusing to evaluate held-out test set.')
        best=load_checkpoint(best_path,device)
        if best['config']['config_id']!=last['config']['config_id']: raise ValueError('Checkpoint mismatch')
        if best['config']['manifest_sha256']!=digest or best['config']['architecture_sha256']!=arch_hash:
            raise ValueError('Data or architecture drift')
        test_report=out/'heldout_test_report.json'
        if test_report.exists(): raise FileExistsError('Held-out test report already exists. Refusing repeat evaluation: '+str(test_report))
        model.load_state_dict(best['model'])
        result=epoch_run(model,loader['test'],None,None,device,amp,False)
        json_write(test_report,{'dataset':args.dataset,'best_epoch':best['epoch'],'best_validation_accuracy':best['best_accuracy'],
              'test':result,'config':best['config'],'model_checkpoint_sha256':hash_file(best_path),
              'warning':'Measured on training GPU, not edge hardware. No edge latency/energy inferred.'})
        print('Held-out test:',result,'\nReport:',test_report)
        return
    start_epoch=1; best_acc=-1.; best_loss=float('inf'); best_epoch=None
    history_path=out/'epoch_history.json'
    history=[]
    if args.resume:
        if not last_path.exists(): raise FileNotFoundError('No last checkpoint to resume')
        saved=load_checkpoint(last_path,device)
        if saved['config']['config_id']!=config_id: raise ValueError('Resume configuration mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        if scaler is not None and saved.get('scaler') is not None: scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng'])
        start_epoch=saved['epoch']+1; best_acc=saved['best_accuracy']; best_loss=saved['best_loss']; best_epoch=saved['best_epoch']
        history=json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
        if len(history)!=saved['epoch']: raise ValueError('History length mismatch on resume')
        print('Resuming at epoch',start_epoch)
    elif last_path.exists() or best_path.exists():
        raise FileExistsError('Existing training checkpoints detected. Use --resume or move previous run before starting.')
    for epoch in range(start_epoch,args.epochs+1):
        # Deterministic epoch-level order/augmentations, including after resumption.
        seed_all(args.seed+epoch)
        train_loader=build_dataloader(ds['train'],'train',args.batch_size,args.workers,args.seed+epoch)
        train_result=epoch_run(model,train_loader,optimizer,scaler,device,amp,True)
        val_result=epoch_run(model,loader['validation'],None,None,device,amp,False)
        lr=optimizer.param_groups[0]['lr']; scheduler.step()
        improved=val_result['accuracy']>best_acc or (val_result['accuracy']==best_acc and val_result['loss']<best_loss)
        if improved:
            best_acc=val_result['accuracy']; best_loss=val_result['loss']; best_epoch=epoch
            save_checkpoint(best_path,{'model':model.state_dict(),'epoch':epoch,'best_accuracy':best_acc,
                            'best_loss':best_loss,'config':config})
        history.append({'epoch':epoch,'learning_rate':lr,'train':train_result,'validation':val_result,'new_best':improved})
        json_write(history_path,history)
        save_checkpoint(last_path,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                         'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict() if scaler else None,
                         'epoch':epoch,'best_epoch':best_epoch,'best_accuracy':best_acc,'best_loss':best_loss,
                         'rng':rng_state(),'config':config,'training_complete':epoch==args.epochs})
        print(f'Epoch {epoch:03d}/{args.epochs} | train {train_result["accuracy"]:.4f} | val {val_result["accuracy"]:.4f} | best {best_acc:.4f} (epoch {best_epoch}) | {train_result["elapsed_seconds"]:.1f}s')
    json_write(out/'training_summary.json',{'status':'TRAINING_COMPLETE','best_epoch':best_epoch,
        'best_validation_accuracy':best_acc,'best_validation_loss':best_loss,'config':config,
        'model_parameter_count':sum(p.numel() for p in model.parameters()),
        'fp32_parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
        'best_checkpoint':str(best_path),'last_checkpoint':str(last_path),
        'test_evaluated':False,'deployment_latency_measured':False,'deployment_energy_measured':False,
        'peak_activation_memory_measured':False})
    print('Training complete. Best epoch:',best_epoch,'validation accuracy:',best_acc)
    print('Only now, if ready for held-out evaluation, run --mode test once.')


if __name__=='__main__':
    main()
