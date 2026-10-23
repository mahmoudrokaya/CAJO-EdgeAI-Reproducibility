#!/usr/bin/env python3
"""CAJO Step 3: architecture definitions and read-only inspection.

One module for Step 3. Does not train, read datasets, evaluate test sets, or
modify frozen checkpoints. Existing baseline definitions are imported from
02_prepare_baseline_models.py; alternative architectures are initialized from
scratch. They are architecture alternatives, not pruned baseline checkpoints.

Commands (from Codes):
 python 03_architecture_search_space.py --inspect
 python 03_architecture_search_space.py --check

Step 4 may import build_architecture(task, architecture_id, config, root).
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ('CIFAR100', 'CIFAR10', 'SpeechCommands_v0.02')
SHAPES = {'CIFAR100': (1,3,32,32), 'CIFAR10': (1,3,32,32), 'SpeechCommands_v0.02': (1,1,40,101)}
CLASSES = {'CIFAR100':100, 'CIFAR10':10, 'SpeechCommands_v0.02':35}
ALTERNATIVES = {'CIFAR100':'ResNet18_width075', 'CIFAR10':'ResNet8_width075', 'SpeechCommands_v0.02':'DSCNN_width075'}


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def config_at(root):
    path=root/'Reports'/'Optimization_Protocol'/'Final_Experiment_Config'/'cajo_experiment_config.json'
    if not path.is_file(): raise FileNotFoundError('Missing Step 1 JSON: '+str(path))
    with path.open(encoding='utf-8') as f: config=json.load(f)
    if config.get('schema')!='cajo_final_experiment_config_v1': raise ValueError('Unexpected configuration schema')
    return config,path


def baseline_source(root):
    codes=root/'Codes'
    for name in ('02_prepare_baseline_models.py','02_prepare_baseline_models_regenerated.py'):
        path=codes/name
        if path.is_file():
            spec=importlib.util.spec_from_file_location('cajo_frozen_baseline_arch',path)
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module,path
    raise FileNotFoundError('Original frozen baseline architecture source not found in Codes; do not substitute a new baseline definition')


def build_baseline(task, source):
    if task=='CIFAR100': return source.CIFARResNet((2,2,2,2),100)
    if task=='CIFAR10': return source.ResNet8(10)
    if task=='SpeechCommands_v0.02': return source.DSCNN(num_classes=35,width=64)
    raise ValueError('Unknown task: '+str(task))


def build_cifar_resnet18_width075():
    """CIFAR stem and 4 BasicBlock stages, widths 48/96/192/384 (75% of 64/128/256/512)."""
    import torch.nn as nn
    class Block(nn.Module):
        def __init__(self, channels_in, channels_out, stride):
            super().__init__()
            self.conv1=nn.Conv2d(channels_in,channels_out,3,stride,1,bias=False)
            self.bn1=nn.BatchNorm2d(channels_out)
            self.relu=nn.ReLU(inplace=True)
            self.conv2=nn.Conv2d(channels_out,channels_out,3,1,1,bias=False)
            self.bn2=nn.BatchNorm2d(channels_out)
            self.shortcut=(nn.Identity() if stride==1 and channels_in==channels_out else
                nn.Sequential(nn.Conv2d(channels_in,channels_out,1,stride,bias=False),nn.BatchNorm2d(channels_out)))
        def forward(self,x):
            out=self.relu(self.bn1(self.conv1(x)))
            return self.relu(self.bn2(self.conv2(out))+self.shortcut(x))
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1=nn.Conv2d(3,48,3,1,1,bias=False)
            self.bn1=nn.BatchNorm2d(48)
            self.relu=nn.ReLU(inplace=True)
            in_ch=48
            layers=[]
            for stage,width in enumerate((48,96,192,384)):
                blocks=[]
                for index in range(2):
                    stride=2 if stage>0 and index==0 else 1
                    blocks.append(Block(in_ch,width,stride));in_ch=width
                layers.append(nn.Sequential(*blocks))
            self.layer1,self.layer2,self.layer3,self.layer4=layers
            self.pool=nn.AdaptiveAvgPool2d((1,1))
            self.fc=nn.Linear(384,100)
        def forward(self,x):
            x=self.relu(self.bn1(self.conv1(x)))
            for layer in (self.layer1,self.layer2,self.layer3,self.layer4):x=layer(x)
            return self.fc(self.pool(x).flatten(1))
    return Net()


def build_cifar_resnet8_width075():
    """Three-stage CIFAR ResNet8, widths 12/24/48 (75% of 16/32/64)."""
    import torch.nn as nn
    class Block(nn.Module):
        def __init__(self,in_ch,out_ch,stride):
            super().__init__()
            self.conv1=nn.Conv2d(in_ch,out_ch,3,stride,1,bias=False)
            self.bn1=nn.BatchNorm2d(out_ch)
            self.conv2=nn.Conv2d(out_ch,out_ch,3,1,1,bias=False)
            self.bn2=nn.BatchNorm2d(out_ch)
            self.shortcut=nn.Identity() if in_ch==out_ch and stride==1 else nn.Sequential(nn.Conv2d(in_ch,out_ch,1,stride,bias=False),nn.BatchNorm2d(out_ch))
            self.relu=nn.ReLU(inplace=True)
        def forward(self,x):
            z=self.relu(self.bn1(self.conv1(x)))
            return self.relu(self.bn2(self.conv2(z))+self.shortcut(x))
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1=nn.Conv2d(3,12,3,1,1,bias=False)
            self.bn1=nn.BatchNorm2d(12)
            self.layer1=Block(12,12,1)
            self.layer2=Block(12,24,2)
            self.layer3=Block(24,48,2)
            self.pool=nn.AdaptiveAvgPool2d(1)
            self.fc=nn.Linear(48,10)
            self.relu=nn.ReLU(inplace=True)
        def forward(self,x):
            x=self.relu(self.bn1(self.conv1(x)))
            for layer in (self.layer1,self.layer2,self.layer3):x=layer(x)
            return self.fc(self.pool(x).flatten(1))
    return Net()


def build_architecture(task, architecture_id, config, root=ROOT):
    """Construct a declared model. Never loads checkpoints or datasets."""
    if task not in TASKS:raise ValueError('Unknown task: '+str(task))
    declared={entry['id'] for entry in config['tasks'][task]['architectures']}
    if architecture_id not in declared:raise ValueError('Architecture not declared in Step 1: '+str(architecture_id))
    if architecture_id=='baseline':
        source,_=baseline_source(Path(root))
        return build_baseline(task,source)
    if architecture_id!=ALTERNATIVES[task]:raise ValueError('No exact implementation for architecture: '+str(architecture_id))
    if task=='CIFAR100':return build_cifar_resnet18_width075()
    if task=='CIFAR10':return build_cifar_resnet8_width075()
    source,_=baseline_source(Path(root))
    return source.DSCNN(num_classes=35,width=48)


def main():
    ap=argparse.ArgumentParser(description='Step 3: read-only architecture inspection')
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inspect',action='store_true',help='No torch model construction')
    mode.add_argument('--check',action='store_true',help='CPU forward/shape/parameter checks; no datasets')
    ap.add_argument('--root',type=Path,default=ROOT)
    args=ap.parse_args()
    root=args.root.resolve()
    config,path=config_at(root)
    print('CAJO STEP 3:', 'ARCHITECTURE_CHECK' if args.check else 'READ_ONLY_INSPECTION')
    print('Config SHA256:',sha256(path))
    for task in TASKS:
        ids=[entry['id'] for entry in config['tasks'][task]['architectures']]
        if ids!=['baseline',ALTERNATIVES[task]]:raise ValueError('Unexpected declared architecture choices for '+task+': '+repr(ids))
        print(task+': '+', '.join(ids))
    if args.check:
        import torch
        torch.set_num_threads(1)
        source,source_path=baseline_source(root)
        print('Original baseline source SHA256:',sha256(source_path))
        for task in TASKS:
            for architecture_id in ('baseline',ALTERNATIVES[task]):
                model=build_architecture(task,architecture_id,config,root).cpu().eval()
                params=sum(p.numel() for p in model.parameters())
                if architecture_id=='baseline' and params!=config['tasks'][task]['baseline_parameters']:
                    raise AssertionError(f'{task}: baseline architecture changed ({params} parameters)')
                with torch.inference_mode():output=model(torch.zeros(SHAPES[task]))
                if tuple(output.shape)!=(1,CLASSES[task]) or not torch.isfinite(output).all():
                    raise AssertionError(f'{task}/{architecture_id}: invalid output')
                if architecture_id!= 'baseline':
                    baseline_params=config['tasks'][task]['baseline_parameters']
                    if not 0<params<baseline_params:raise AssertionError(f'{task}: alternate not smaller than baseline')
                print(f'  {architecture_id}: params={params}, output={tuple(output.shape)} PASS')
    print('No datasets, training, frozen checkpoint writes, GPU or final results.')

if __name__=='__main__':main()
