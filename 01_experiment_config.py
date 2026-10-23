#!/usr/bin/env python3
"""CAJO Step 1: create, validate, and explicitly lock ONE experiment manifest.
No training, GPU execution, dataset access, or baseline modification.

Usage:
 python 01_experiment_config.py --init
 python 01_experiment_config.py --validate
 python 01_experiment_config.py --lock --approve

Edit the generated JSON after --init. Lock requires complete, explicit values.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / 'Reports' / 'Optimization_Protocol' / 'Final_Experiment_Config' / 'cajo_experiment_config.json'
CHECKPOINTS = {
 'CIFAR100': ('Baseline_Training/CIFAR100/best_model.pt','0dda6302fb9fbb23abe76851ee40a93e98e47389a6107fca94c8c9cd73c2bbdd',11220132),
 'CIFAR10': ('Baseline_Training/CIFAR10/best_model.pt','9c428a089c60116aa70bf582dd2d8612b38372f87994250cc06c9fa6b313437b',308650),
 'SpeechCommands_v0.02': ('Baseline_Training/SpeechCommands_v0.02/best_model.pt','854bb24e924aa3a5d1e70f92a669f52a56bf90828146caba5f153619cbf8bcdc',22691),
}
SPLITS = {'CIFAR100':'f95b778d146c19d9d6252d3d30884cfd46981ff5ded59600a06f6192cec4ad71',
 'CIFAR10':'e5f6ee1f7cb0b9fde92b739de2cb3112a77cfb4d5a3f8a59feef1b20eb950c19',
 'SpeechCommands_v0.02':'9ec3cd9f2d3e2907639fc3e340b9e9bdfbbc4e860298060a9d9093887bfc286c'}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def initial():
 tasks={}
 for name,(rel,digest,params) in CHECKPOINTS.items():
  tasks[name]={'split_manifest_sha256':SPLITS[name], 'frozen_checkpoint':str(Path('Reports')/rel),
               'checkpoint_sha256':digest,'baseline_parameters':params,
               'architectures':[{'id':'baseline','definition':'frozen baseline architecture'}],
               'accuracy_floor_percent':None,'max_total_inference_memory_mib':None,
               'max_latency_ms':None,'max_storage_mib':None}
 return {
  'schema':'cajo_final_experiment_config_v1','status':'DRAFT_NOT_LOCKED',
  'methodology_sections':['2.1','2.2','2.3'],
  'claims_scope':{'deployment_target':'local_workstation_only',
    'target_cpu':'Intel Core Ultra 7 265KF','target_gpu':'NVIDIA RTX 5070 12GB',
    'ram_gib':32,'inference_backend':None,'inference_device':None,
    'embedded_or_ARM_claims':False,
    'energy_policy':'EXCLUDED_PENDING_MANUSCRIPT_REVISION',
    'energy_note':'Energy is not measured; revise equations/objectives/feasibility and related claims before locking, or provide device-level instrumentation.'},
  'tasks':tasks,
  'techniques':{'structured_pruning_ratios':[0.0,0.25,0.50],
    'pruning_importance':'BN gamma magnitude; deterministic index tie-break (proposed)',
    'random_pruning_control':True,'precision_options':['FP32','INT8'],
    'quantization_backend':None,'quantization_calibration_train_only_count':None,
    'quantization_calibration_policy':None,'kd_options':[False,True],
    'kd_temperature':4.0,'kd_alpha':0.5,'kd_teacher':'task-specific frozen baseline',
    'kd_is_compression':False},
  'recovery':{'epochs':20,'optimizer':'SGD','learning_rate':0.01,'momentum':0.9,
    'weight_decay':0.0005,'batch_size':128,'lr_schedule':'CosineAnnealingLR(T_max=20, eta_min=0), step at epoch end',
    'seeds':[42,43,44],'checkpoint_rule':'highest validation accuracy, earliest epoch on ties',
    'paired_augmentation':True,'train_only_fitting':True},
  'search':{'strategy':'deterministic_predeclared','architecture_count_minimum_per_task':2,
    'joint_dimensions':['architecture','pruning','precision','KD'],
    'sequential_comparator':None,'joint_candidate_policy':None,
    'max_evaluated_candidates_per_task_per_strategy':None,
    'search_training_budget_per_candidate':None,
    'matched_compute_budget_policy':None,
    'screening_policy':None},
  'measurement':{'warmup_runs':None,'timed_runs':None,'threads':None,
    'batch_size':1,'latency_statistic':'median','memory_metric':'peak live inference activations/workspace excluding parameter storage',
    'memory_measurement_method':None,'compatibility_probe':True,
    'storage_metric':'actual serialized parameter storage',
    'energy_metric':None},
  'selection':{'validation_only':True,'test_access':'sealed_until_selection_frozen',
    'objectives':['minimize_validation_error','minimize_storage','minimize_peak_live_memory','minimize_latency'],
    'pareto':'non-dominated feasible candidates only','primary_preference':None,
    'tie_break_order':['latency','memory','validation_error'],
    'infeasible_policy':'report no feasible configuration; do not substitute infeasible model'},
  'comparison_requirements':['baseline','importance_P25_CE','random_P25_CE',
    'importance_P25_KD','importance_P50_KD','baseline_INT8',
    'sequential_prune_recover_INT8','genuine_joint_multi_architecture'],
  'outputs':['baseline_characterization','compression_ablation',
    'sequential_vs_joint_matched_budget','hardware_feasibility','pareto_front',
    'frozen_selection','heldout_final_evaluation'],
  'approval':{'human_approved':False,'approver':None,'manuscript_energy_alignment_confirmed':False,
    'full_training_authorized':False},
  'provenance':{'source':'frozen assets and methodology reviewed through Phase 7H',
    'baseline_retraining_permitted':False,'chameleon_required':False}
 }

def validate(d):
 errors=[]
 def require(v,label):
  if v is None or v=='' or v==[]: errors.append('Missing '+label)
 def positive(v,label):
  if not isinstance(v,(int,float)) or isinstance(v,bool) or v<=0: errors.append('Must be positive: '+label)
 if d.get('schema')!='cajo_final_experiment_config_v1': errors.append('Invalid schema')
 c=d['claims_scope']; m=d['measurement']; s=d['search']; a=d['approval']; t=d['techniques']
 for key in ('inference_backend','inference_device'): require(c.get(key),'claims_scope.'+key)
 if c.get('energy_policy')=='EXCLUDED_PENDING_MANUSCRIPT_REVISION':
  if not a.get('manuscript_energy_alignment_confirmed'):
   errors.append('Manuscript still requires measured energy: document approved revision or instrument energy')
 elif c.get('energy_policy')=='MEASURED_DEVICE_LEVEL':
  require(m.get('energy_metric'),'measurement.energy_metric')
 else: errors.append('Unsupported energy policy')
 for key in ('quantization_backend','quantization_calibration_train_only_count','quantization_calibration_policy'):
  require(t.get(key),'techniques.'+key)
 for key in ('sequential_comparator','joint_candidate_policy','max_evaluated_candidates_per_task_per_strategy','search_training_budget_per_candidate','matched_compute_budget_policy','screening_policy'):
  require(s.get(key),'search.'+key)
 positive(s.get('max_evaluated_candidates_per_task_per_strategy'),'search.max_evaluated_candidates_per_task_per_strategy')
 for key in ('warmup_runs','timed_runs','threads','memory_measurement_method'): require(m.get(key),'measurement.'+key)
 for key in ('warmup_runs','timed_runs','threads'): positive(m.get(key),'measurement.'+key)
 require(d['selection'].get('primary_preference'),'selection.primary_preference')
 for name,task in d['tasks'].items():
  if task.get('checkpoint_sha256')!=CHECKPOINTS[name][1] or task.get('split_manifest_sha256')!=SPLITS[name]:
   errors.append(name+': frozen hashes altered')
  arch=task.get('architectures',[])
  if len(arch)<2 or len({v.get('id') for v in arch})!=len(arch):
   errors.append(name+': require at least 2 distinct, explicitly defined architectures')
  for item in arch:
   if not item.get('id') or not item.get('definition'): errors.append(name+': incomplete architecture')
  for key in ('accuracy_floor_percent','max_total_inference_memory_mib','max_latency_ms','max_storage_mib'):
   positive(task.get(key),name+'.'+key)
  if isinstance(task.get('accuracy_floor_percent'),(int,float)) and task['accuracy_floor_percent']>100:
   errors.append(name+': accuracy floor above 100%')
 if a.get('full_training_authorized'): errors.append('Configuration step cannot authorize full training')
 if not a.get('human_approved'): errors.append('Human approval missing')
 return sorted(set(errors))

def main():
 p=argparse.ArgumentParser(); mode=p.add_mutually_exclusive_group(required=True)
 mode.add_argument('--init',action='store_true'); mode.add_argument('--validate',action='store_true'); mode.add_argument('--lock',action='store_true')
 p.add_argument('--approve',action='store_true',help='Explicit human approval required with --lock')
 p.add_argument('--config',type=Path,default=DEFAULT)
 args=p.parse_args(); path=args.config
 if args.init:
  if path.exists(): raise SystemExit('REFUSING TO OVERWRITE existing config: '+str(path))
  path.parent.mkdir(parents=True,exist_ok=True)
  path.write_text(json.dumps(initial(),indent=2)+'\n',encoding='utf-8')
  print('DRAFT CREATED:',path,'\nNo experiments authorized. Edit pending fields, then --validate.'); return
 if not path.is_file(): raise SystemExit('Missing config; first run --init: '+str(path))
 d=json.loads(path.read_text(encoding='utf-8'))
 if d.get('status')=='LOCKED':
  print('ALREADY LOCKED; no changes made. SHA256:',sha(path)); return
 errors=validate(d)
 if args.validate:
  print('CONFIG VALIDATION:', 'PASS' if not errors else 'PENDING ('+str(len(errors))+')')
  for e in errors: print(' -',e)
  print('No files changed. No training authorized.'); return
 if not args.approve: raise SystemExit('Lock requires --approve; no changes made')
 if errors:
  print('LOCK BLOCKED:'); [print(' -',e) for e in errors]
  raise SystemExit(2)
 d['status']='LOCKED'; d['approval']['locked_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
 path.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
 print('CONFIG LOCKED:',path,'\nSHA256:',sha(path),'\nFull training remains NOT AUTHORIZED.')
if __name__=='__main__': main()
