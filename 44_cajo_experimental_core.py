#!/usr/bin/env python3
"""Step44: NEW controlled CAJO training experiments (validation only).

Runs two genuinely distinct training orders and CE/KD, P0/P25 controls.
JOINT: prune source before all recovery epochs; SEQUENTIAL: recover unpruned
source for half the epochs, prune physically, recover for remaining epochs.
Both arms use identical seed, batches, total optimizer updates, source, KD,
pruning ratio and final architecture; optimizer reset at the phase boundary
in BOTH arms. This is a controlled operation-order study, NOT simultaneous
QAT or full joint pruning/quantization/NAS. INT8 PTQ is deliberately not
represented as jointly trainable.

Commands:
 python 44_cajo_experimental_core.py run --root .. --task CIFAR10 --seed 42 --epochs 20
 python 44_cajo_experimental_core.py run --root .. --task all --seed all --epochs 20
 python 44_cajo_experimental_core.py summarize --root ..

No test data, frozen checkpoints never modified. Each completed arm gets
SHA256-bound checkpoint and JSON; partial runs cannot be mistaken for success.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,random,sys,time,traceback
from pathlib import Path
LOCK='56361ba3ef2cb530d54a923cf4f5b3758a22bee82a9119822395cef0be9bbdc5'
TASKS=('CIFAR10','CIFAR100','SpeechCommands_v0.02')
OUT=Path('Reports/CAJO_Step44_Controlled_Training')

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def mod(p,n):
 s=importlib.util.spec_from_file_location(n,str(p));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.partial')
 t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');os.replace(t,p)
def mainrun(root,task,seed,epochs,prune,kd,order,device):
 import torch
 import torch.nn.functional as F
 cfgpath=root/'Reports/Optimization_Protocol/Final_Experiment_Config/cajo_experiment_config.json'
 if digest(cfgpath)!=LOCK:raise RuntimeError('LOCKED CONFIG DRIFT')
 cfg=json.loads(cfgpath.read_text(encoding='utf8'))
 if not (2<=epochs<=100) or prune not in (0,.25):raise ValueError('Invalid training configuration')
 if order=='sequential' and prune==0:order='joint' # identical P0 control is trained once
 name=f'{task}__s{seed}__P{int(prune*100)}__{("KD" if kd else "CE")}__{order}__e{epochs}'
 dest=root/OUT/name;result=dest/'result.json';checkpoint=dest/'best_model.pt'
 if result.exists():
  r=json.loads(result.read_text(encoding='utf8'))
  if r['checkpoint_sha256']!=digest(checkpoint) or r['config_sha256']!=LOCK:raise RuntimeError('Result/checkpoint drift '+name)
  print('SKIP VERIFIED',name,flush=True);return
 if dest.exists() and any(dest.iterdir()):raise RuntimeError('Partial experiment requires manual inspection: '+str(dest))
 dest.mkdir(parents=True,exist_ok=True)
 adapter=mod(root/'Codes/04_compression_recovery.py','cajo44_adapter')
 arch,data,surgery,importance=adapter.sources(root)
 splits,manifest=adapter.verified_manifest(root,cfg,task,data)
 teacher,source=adapter.frozen_teacher(root,cfg,task,arch,torch)
 sourcehash=digest(source)
 # Always initialize from the SAME frozen source, never alter it.
 student=arch.build_architecture(task,'baseline',cfg,root)
 student.load_state_dict(teacher.state_dict(),strict=True)
 if digest(source)!=sourcehash:raise RuntimeError('Source drift')
 tr,val=adapter.make_data(root,task,splits,cfg,seed,data)
 random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 d=torch.device(device);teacher=teacher.to(d).eval()
 teacher.requires_grad_(False)
 details=[]
 if prune and order=='joint':
  details=adapter.physical_prune(student,task,prune,importance,surgery,arch,'importance',seed,root)
 student=student.to(d)
 rec=cfg['recovery'];lr=float(rec['learning_rate']);mom=float(rec['momentum']);wd=float(rec['weight_decay'])
 midpoint=epochs//2
 def optimizer():return torch.optim.SGD(student.parameters(),lr=lr,momentum=mom,weight_decay=wd)
 opt=optimizer();history=[];best=-1.;best_state=None;best_epoch=None
 started=time.time()
 for epoch in range(1,epochs+1):
  # Sequential order: first learn unpruned, then physically prune at same
  # boundary where joint arm resets optimizer. Both have equal epoch counts.
  if epoch==midpoint+1:
   if prune and order=='sequential':
    student=student.cpu()
    details=adapter.physical_prune(student,task,prune,importance,surgery,arch,'importance',seed,root)
    student=student.to(d)
   opt=optimizer()
  student.train();total=0;loss_sum=0.
  for x,y in tr(epoch):
   x=x.to(d);y=y.to(d);opt.zero_grad(set_to_none=True)
   logits=student(x);loss=F.cross_entropy(logits,y)
   if kd:
    with torch.no_grad():tlogits=teacher(x)
    T=float(cfg['techniques']['kd_temperature']);alpha=float(cfg['techniques']['kd_alpha'])
    loss=(1-alpha)*loss+alpha*T*T*F.kl_div(F.log_softmax(logits/T,dim=1),F.softmax(tlogits/T,dim=1),reduction='batchmean')
   loss.backward();opt.step();total+=y.numel();loss_sum+=float(loss.item())*y.numel()
  student.eval();correct=n=0
  with torch.inference_mode():
   for x,y in val():
    pred=student(x.to(d)).argmax(1).cpu();correct+=int((pred==y).sum());n+=len(y)
  accuracy=100*correct/n
  # Sequential pre-prune validation is not eligible for final selection.
  eligible=(not prune or epoch>midpoint)
  if eligible and accuracy>best:
   best=accuracy;best_epoch=epoch
   best_state={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
  history.append(dict(epoch=epoch,train_loss=loss_sum/total,validation_accuracy_pct=accuracy,selection_eligible=eligible))
  print(name,'epoch',epoch,'val',round(accuracy,3),flush=True)
 if best_state is None:raise RuntimeError('No eligible checkpoint')
 torch.save({'model':best_state,'task':task,'seed':seed,'experiment':name,'best_epoch':best_epoch},checkpoint)
 atomic(result,dict(schema='cajo44_controlled_training_v1',experiment=name,task=task,seed=seed,
   pruning_ratio=prune,kd=kd,operation_order=order,epochs=epochs,best_epoch=best_epoch,
   best_validation_accuracy_pct=best,history=history,pruning_details=details,
   source_checkpoint_sha256=sourcehash,split_manifest_sha256=digest(manifest),
   checkpoint_sha256=digest(checkpoint),config_sha256=LOCK,elapsed_seconds=time.time()-started,
   protocol='two-phase equal epochs; both arms optimizer reset at midpoint; FP32 only',
   test_access='SEALED',native_memory_certified=False,algorithm_comparison_scope='physical_pruning_order'))

def summarize(root):
 import csv
 allrows=[]
 for p in sorted((root/OUT).glob('*/result.json')):
  r=json.loads(p.read_text(encoding='utf8'));c=p.parent/'best_model.pt'
  if digest(c)!=r['checkpoint_sha256']:raise RuntimeError('Checkpoint drift '+str(c))
  allrows.append({k:r[k] for k in ('experiment','task','seed','pruning_ratio','kd','operation_order','epochs','best_validation_accuracy_pct','best_epoch','checkpoint_sha256')})
 target=root/OUT/'completed_training.csv';target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=['experiment','task','seed','pruning_ratio','kd','operation_order','epochs','best_validation_accuracy_pct','best_epoch','checkpoint_sha256']);w.writeheader();w.writerows(allrows)
 # Comparisons only when BOTH sides genuinely exist, never fill missing cells.
 lookup={(r['task'],r['seed'],r['pruning_ratio'],r['kd'],r['epochs'],r['operation_order']):r for r in allrows}
 pairs=[]
 for key,r in lookup.items():
  task,seed,prune,kd,epochs,order=key
  if order!='joint' or not prune:continue
  other=lookup.get((task,seed,prune,kd,epochs,'sequential'))
  if other:pairs.append(dict(task=task,seed=seed,pruning_ratio=prune,kd=kd,epochs=epochs,
     joint_accuracy_pct=r['best_validation_accuracy_pct'],sequential_accuracy_pct=other['best_validation_accuracy_pct'],
     joint_minus_sequential_pp=r['best_validation_accuracy_pct']-other['best_validation_accuracy_pct']))
 p=root/OUT/'matched_order_comparison.csv'
 with p.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=['task','seed','pruning_ratio','kd','epochs','joint_accuracy_pct','sequential_accuracy_pct','joint_minus_sequential_pp']);w.writeheader();w.writerows(pairs)
 print(json.dumps(dict(completed_training_arms=len(allrows),matched_joint_sequential_pairs=len(pairs),
  comparison=str(p),sealed_test='UNTOUCHED'),indent=2))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('run','summarize'))
 p.add_argument('--root',type=Path,default=Path('..'));p.add_argument('--task',choices=(*TASKS,'all'),default='CIFAR10')
 p.add_argument('--seed',choices=('42','43','44','all'),default='42');p.add_argument('--epochs',type=int,default=20)
 p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
 a=p.parse_args();root=a.root.resolve()
 if a.action=='summarize':summarize(root);return
 for task in (TASKS if a.task=='all' else (a.task,)):
  for seed in ((42,43,44) if a.seed=='all' else (int(a.seed),)):
   # Five distinct experimental arms per task/seed: P0 CE, P0 KD,
   # P25 CE joint, P25 KD joint, P25 KD sequential, P25 CE sequential.
   for prune,kd,order in ((0,False,'joint'),(0,True,'joint'),(.25,False,'joint'),(.25,True,'joint'),(.25,False,'sequential'),(.25,True,'sequential')):
    mainrun(root,task,seed,a.epochs,prune,kd,order,a.device)
if __name__=='__main__':
 try:main()
 except Exception as e:print('ERROR:',repr(e),file=sys.stderr);traceback.print_exc();sys.exit(2)
