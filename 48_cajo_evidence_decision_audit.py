#!/usr/bin/env python3
"""CAJO Step 48: full-history manuscript evidence decision audit.
Read-only: never loads checkpoints, imports experimental scripts, trains, runs inference,
or opens test datasets. Standard library Python 3.10+.
"""
import argparse,csv,hashlib,json,re,statistics,collections,datetime,math,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

REQ={
'R01':('Accuracy and split integrity','accuracy'),
'R02':('Complete native inference memory','memory'),
'R03':('CPU inference latency','latency'),
'R04':('Energy measurement and uncertainty','energy'),
'R05':('Hardware/data/teacher admissibility','admissibility'),
'R06':('Layerwise pruning/quantization/architecture provenance','configuration'),
'R07':('KD-only ablation','kd'),
'R08':('Pruning-only ablation','pruning'),
'R09':('Quantization-only ablation','quant'),
'R10':('NAS-only matched ablation','nas'),
'R11':('Independent joint versus sequential','algorithm'),
'R12':('Matched search budgets','budget'),
'R13':('Five verified objective values','five'),
'R14':('Five-dimensional Pareto and feasibility','pareto'),
'R15':('Frozen selection and tie break','freeze'),
'R16':('Matched baseline comparisons','baseline'),
'R17':('Seed-level statistical uncertainty','statistics'),
'R18':('Sealed final test after freeze','test'),
'R19':('Validation versus test wording','wording_test'),
'R20':('Selection-rule wording','wording_selection'),
'R21':('Energy instrumentation wording','wording_energy'),
'R22':('Experimental declarations','wording_declarations')}
TASKS=('CIFAR10','CIFAR100','SpeechCommands_v0.02')
LOCKED='56361ba3ef2cb530d54a923cf4f5b3758a22bee82a9119822395cef0be9bbdc5'
LIMIT=30*1024*1024

def read(p):
 if not p.is_file() or p.stat().st_size>LIMIT:return None
 if p.suffix.lower()=='.json':return json.loads(p.read_text(encoding='utf-8-sig'))
 if p.suffix.lower()=='.csv':
  # HWiNFO CSV files may use Windows-1252 (degree symbol).
  data=p.read_bytes()
  for enc in ('utf-8-sig','cp1252','latin-1'):
   try:
    t=data.decode(enc);return list(csv.DictReader(t.splitlines()))
   except UnicodeError:continue
 return None

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def num(x):
 try:
  v=float(x);return v if math.isfinite(v) else None
 except (TypeError,ValueError):return None

def get(row,*keys):
 if not isinstance(row,dict):return None
 for key in keys:
  if key in row and row[key] not in ('',None,'NOT_MEASURED','NOT_AVAILABLE','null'):
   return row[key]
 return None

def docx_text(path):
 if not path or not path.is_file():return ''
 with zipfile.ZipFile(path) as z:
  xml=ET.fromstring(z.read('word/document.xml'))
 ns='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
 return '\n'.join(''.join(n.text or '' for n in p.iter(ns+'t')) for p in xml.iter(ns+'p'))

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',type=Path,default=Path('..'))
 ap.add_argument('--paper',type=Path,default=None)
 ap.add_argument('--out',type=Path,default=None)
 args=ap.parse_args();root=args.root.resolve();reports=root/'Reports';codes=root/'Codes'
 if not reports.is_dir():ap.error('Reports directory not found: '+str(reports))
 out=(args.out or reports/'CAJO_Step48_Evidence_Decisions').resolve();out.mkdir(parents=True,exist_ok=True)
 errors=[];inventory=[];datasets=[];index=collections.defaultdict(list)
 # Parse ALL numerical/report artifacts, not merely keyword-selected files.
 for base in (reports,codes):
  if not base.exists():continue
  for p in sorted(base.rglob('*')):
   if not p.is_file() or p.suffix.lower() not in ('.json','.csv'):continue
   if out==p.parent or out in p.parents:continue
   rel=p.relative_to(root).as_posix()
   try:
    obj=read(p)
    if obj is None:continue
    rows=obj if isinstance(obj,list) else [obj]
    cols=sorted({str(k) for r in rows[:50] if isinstance(r,dict) for k in r})
    record={'path':rel,'type':p.suffix.lower(),'rows':len(rows),'columns':';'.join(cols[:80]),'sha256':sha(p)}
    inventory.append(record);datasets.append((rel,obj))
    for key in cols:index[key.lower()].append(rel)
   except (OSError,ValueError,UnicodeError,json.JSONDecodeError,csv.Error) as e:
    errors.append({'path':rel,'error':str(e)})
 lookup={p:d for p,d in datasets}
 def find(suffix):
  # Prefer exact suffix; if several match, use the most specific shortest path.
  a=[(p,v) for p,v in datasets if p.replace('\\','/').endswith(suffix)]
  return min(a,key=lambda x:len(x[0])) if a else (None,None)
 def rows_of(suffix):
  p,d=find(suffix);return (p,d if isinstance(d,list) else [])
 def dict_of(suffix):
  p,d=find(suffix);return (p,d if isinstance(d,dict) else {})
 cfg_path=reports/'Optimization_Protocol'/'Final_Experiment_Config'/'cajo_experiment_config.json'
 config_ok=cfg_path.is_file() and sha(cfg_path)==LOCKED
 gatepath,gate=dict_of('CAJO_Step38_Integrated_Closure_v1/evidence_gate.json')
 checks=gate.get('checks',{}) if isinstance(gate.get('checks'),dict) else {}
 cpath,candidates=rows_of('CAJO_Step43_Experimental_Execution/measured_validation_latency_storage.csv')
 qpath,quant=rows_of('CAJO_Step43_Experimental_Execution/paired_precision_ablation.csv')
 bpath,baselines=rows_of('Baseline_Resource_Profiling/baseline_resource_comparison.csv')
 epath,energy=rows_of('CAJO_Step34_Audit/energy_models.csv')
 ppath,provisional=rows_of('CAJO_Step43_Experimental_Execution/provisional_three_objective_pareto.csv')
 vpath,verified=rows_of('CAJO_Step38_Integrated_Closure_v1/validated_objectives_input.csv')
 # Scan ALL JSON/CSV reports for evidence records with numerical fields and provenance.
 extracted=[]
 PATTERNS={
 'accuracy':('validation_accuracy_pct','best_validation_accuracy_pct','val_accuracy','accuracy_pct'),
 'latency':('latency_mean_ms','latency_ms','mean_latency_ms'),
 'storage':('storage_bytes','serialized_bytes','model_size_bytes'),
 'memory':('native_peak_bytes','native_inference_peak_bytes','model_native_peak_bytes'),
 'energy':('energy_mj','energy_per_inference_mj','net_energy_mj'),
 'seed':('seed','random_seed'),
 'candidate':('candidate_id','export_sha256','model_id')}
 for rel,obj in datasets:
  rs=obj if isinstance(obj,list) else [obj]
  if len(rs)>200000:continue
  for i,r in enumerate(rs):
   if not isinstance(r,dict):continue
   vals={name:get(r,*aliases) for name,aliases in PATTERNS.items()}
   numeric={k:num(vals[k]) for k in ('accuracy','latency','storage','memory','energy')}
   if not any(x is not None for x in numeric.values()):continue
   extracted.append({'source':rel,'row':i+1,'task':get(r,'task','dataset'),
     'candidate_id':get(r,'candidate_id','model_id'),
     'precision':get(r,'precision','dtype'),'seed':vals['seed'],
     'export_sha256':get(r,'export_sha256'),'accuracy':numeric['accuracy'],
     'latency_ms':numeric['latency'],'storage_bytes':numeric['storage'],
     'native_memory_bytes':numeric['memory'],'energy_mj':numeric['energy'],
     'verification_flag':get(r,'verified','certified','native_memory_verified','energy_verified')})
 # Step44 individual training arms, with source and paired comparisons.
 arms=[]
 for rel,obj in datasets:
  if 'CAJO_Step44_Controlled_Training/' not in rel or not rel.endswith('/result.json') or not isinstance(obj,dict):continue
  name=Path(rel).parent.name
  task=get(obj,'task') or next((t for t in TASKS if t in name),None)
  seed=get(obj,'seed');m=re.search(r'__s(\d+)(?:__|$)',name)
  if seed is None and m:seed=m.group(1)
  pm=re.search(r'__P(\d+)(?:__|$)',name)
  km=re.search(r'__(KD|CE)(?:__|$)',name)
  om=re.search(r'__(joint|sequential)(?:__|$)',name)
  arms.append({'source':rel,'task':task,'seed':str(seed) if seed is not None else None,
   'pruning':pm.group(1) if pm else None,'loss':km.group(1) if km else None,
   'order':om.group(1) if om else None,
   'accuracy':num(get(obj,'best_validation_accuracy_pct','validation_accuracy_pct'))})
 byarm=collections.defaultdict(list)
 for r in arms:byarm[(r['task'],r['seed'],r['pruning'],r['loss'],r['order'])].append(r)
 contrasts=[]
 contrasts_spec=[('KD only',('0','KD','joint'),('0','CE','joint')),
  ('Pruning CE',('25','CE','joint'),('0','CE','joint')),
  ('Pruning KD',('25','KD','joint'),('0','KD','joint')),
  ('Joint versus sequential CE',('25','CE','joint'),('25','CE','sequential')),
  ('Joint versus sequential KD',('25','KD','joint'),('25','KD','sequential'))]
 for title,a,b in contrasts_spec:
  for task in TASKS:
   for seed in ('42','43','44'):
    aa=byarm.get((task,seed,*a),[]);bb=byarm.get((task,seed,*b),[])
    if len(aa)!=1 or len(bb)!=1:continue
    if aa[0]['accuracy'] is None or bb[0]['accuracy'] is None:continue
    contrasts.append({'contrast':title,'task':task,'seed':seed,
     'delta_pp':round(aa[0]['accuracy']-bb[0]['accuracy'],6),
     'source_a':aa[0]['source'],'source_b':bb[0]['source']})
 paper=docx_text(args.paper.resolve()) if args.paper else ''
 text=paper.lower()
 conflicts={
  'R19':bool(re.search(r'number of correctly classified test samples|n_test',text)),
  'R20':bool('energy-critical deployment selects' in text or 'ties are resolved using latency, memory, and prediction error' in text),
  'R21':bool('measured supply voltage and current' in text),
  'R22':bool('no original software implementation' in text or 'no new datasets were generated or analyzed' in text or 'conceptual synthesis' in text)}
 # Exact evidence gates override optimistic inferred readiness. No file count certifies a result.
 def decide(rid):
  if rid in conflicts:
   if not paper:return 'VERIFY EXISTING EVIDENCE','Manuscript was not supplied; text-to-method consistency cannot be checked.',[]
   if conflicts[rid]:return 'MANUSCRIPT CORRECTION','The manuscript contains wording inconsistent with the experimental protocol or performed work.',[str(args.paper)]
   return 'VERIFY EXISTING EVIDENCE','Known conflict phrase not detected; confirm wording manually.',[str(args.paper)]
  if rid=='R01':
   n=sum(all(num(get(r,*k)) is not None for k in (('validation_accuracy_pct','accuracy_pct'),('latency_mean_ms',),('storage_bytes',))) for r in candidates)
   return ('USE EXISTING RESULTS' if n>=36 else 'VERIFY EXISTING EVIDENCE',f'{n} candidate rows include accuracy, latency and storage; verify split provenance separately.',[cpath] if cpath else [])
  if rid=='R02':return ('USE EXISTING RESULTS' if checks.get('native_memory') is True else 'VERIFY EXISTING EVIDENCE','Native inference memory gate is '+str(checks.get('native_memory'))+'; existing profiler/allocator outputs must be inspected before ordering a new meter.',[gatepath] if gatepath else [])
  if rid=='R03':return ('USE EXISTING RESULTS' if len(candidates)>=36 else 'VERIFY EXISTING EVIDENCE',f'{len(candidates)} Step43 deployment measurements; protocol matching and measurement variability need confirmation.',[cpath] if cpath else [])
  if rid=='R04':return ('USE EXISTING RESULTS' if checks.get('energy') is True else 'VERIFY EXISTING EVIDENCE',f'Energy gate={checks.get("energy")}; {len(energy)} historical energy rows. Resolve metrology before repeating measurements.',[x for x in (epath,gatepath) if x])
  if rid=='R05':return 'VERIFY EXISTING EVIDENCE','Search/configuration files exist; contextual admission/exclusion decisions must be extracted and checked.',[]
  if rid=='R06':return 'VERIFY EXISTING EVIDENCE','Candidate configuration and source implementation require checkpoint/export-level provenance, not just matching names.',[]
  if rid in ('R07','R08'):
   target='KD only' if rid=='R07' else 'Pruning CE';n=sum(x['contrast']==target for x in contrasts)
   return ('COMPLETE ANALYSIS ONLY' if n==9 else 'VERIFY EXISTING EVIDENCE',f'{n}/9 matched seed contrasts reconstructed; summarize uncertainty and training provenance.',[x['source_a'] for x in contrasts if x['contrast']==target][:3])
  if rid=='R09':return ('COMPLETE ANALYSIS ONLY' if len(quant)>=18 else 'VERIFY EXISTING EVIDENCE',f'{len(quant)} matched precision rows; quantify accuracy/storage/latency trade-offs and check identical export ancestry.',[qpath] if qpath else [])
  if rid=='R10':return 'VERIFY EXISTING EVIDENCE','Earlier architecture variants found in project history; a NAS-only matched ablation is not established from current summary files.',[]
  if rid=='R11':
   n=sum(x['contrast']=='Joint versus sequential KD' for x in contrasts)
   return 'VERIFY EXISTING EVIDENCE',f'{n}/9 matched Step44 order contrasts; independent four-component optimization and equal search budgets remain unverified (gate={checks.get("independent_algorithm_comparison")}).',[gatepath] if gatepath else []
  if rid=='R12':return 'VERIFY EXISTING EVIDENCE','Inspect actual search logs for candidate counts, compute, training epochs and selection budget for each policy.',[]
  if rid=='R13':return ('USE EXISTING RESULTS' if checks.get('five_objective_complete') is True and verified else 'COMPLETE ANALYSIS ONLY' if verified else 'VERIFY EXISTING EVIDENCE',f'{len(verified)} validated objective input rows; five-objective gate={checks.get("five_objective_complete")}.',[x for x in (vpath,gatepath) if x])
  if rid=='R14':return ('USE EXISTING RESULTS' if checks.get('five_objective_complete') is True and checks.get('three_tasks_feasible') is True else 'VERIFY EXISTING EVIDENCE',f'{len(provisional)} provisional 3D Pareto rows; cannot certify 5D frontier without validated memory and energy.',[x for x in (ppath,gatepath) if x])
  if rid=='R15':return 'VERIFY EXISTING EVIDENCE','No frozen choice is established by the supplied closure gate; selection requires a verified feasible Pareto set.',[gatepath] if gatepath else []
  if rid=='R16':return 'VERIFY EXISTING EVIDENCE',f'{len(baselines)} baseline resource profiles; determine whether complete matched CPU memory/latency/energy rows exist in earlier reports.',[bpath] if bpath else []
  if rid=='R17':return ('COMPLETE ANALYSIS ONLY' if contrasts else 'VERIFY EXISTING EVIDENCE',f'{len(contrasts)} seed-paired contrasts available; calculate uncertainty and method-level comparisons without mixing split protocols.',[x['source_a'] for x in contrasts[:3]])
  if rid=='R18':return 'NEW EXPERIMENT REQUIRED' if gate.get('sealed_test_access') in ('SEALED','SEALED_UNTIL_FROZEN') else 'VERIFY EXISTING EVIDENCE','Final held-out evaluation must occur exactly once AFTER documented selection freeze; do not access test data now.',[gatepath] if gatepath else []
  return 'VERIFY EXISTING EVIDENCE','No decisive proof found.',[]
 decisions=[]
 for rid,(name,topic) in REQ.items():
  status,reason,paths=decide(rid)
  # Show the strongest matching historical numerical sources, without certifying them.
  token_map={'memory':['memory','native','packed'],'energy':['energy','power','hwinfo'],
   'nas':['architecture','search'],'algorithm':['joint','sequential'],'baseline':['baseline'],
   'budget':['search','budget'],'configuration':['candidate','pruning'],
   'admissibility':['admissibility','eligibility'],'five':['five_objective','objective'],
   'pareto':['pareto'],'freeze':['freeze','selection']}
  tokens=token_map.get(topic,[])
  earlier=[r['source'] for r in extracted if any(t in r['source'].lower() for t in tokens)]
  earlier=list(dict.fromkeys(earlier))[:12]
  decisions.append({'requirement':rid,'paper_requirement':name,'decision':status,
   'evidence_observed':reason,'direct_sources':'; '.join(p for p in paths if p),
   'earlier_numeric_sources_to_review':'; '.join(earlier),
   'next_action':('Use cited evidence in results tables.' if status=='USE EXISTING RESULTS' else
    'Compute comparisons and uncertainty from existing files; no training.' if status=='COMPLETE ANALYSIS ONLY' else
    'Correct manuscript wording to match actual protocol.' if status=='MANUSCRIPT CORRECTION' else
    'Perform only after a frozen validation selection.' if rid=='R18' else
    'Inspect the cited earlier numerical artifacts and producing code before commissioning any experiment.')})
 def savecsv(name,rows,cols):
  with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
 savecsv('requirement_decisions.csv',decisions,list(decisions[0]))
 savecsv('numerical_evidence_index.csv',extracted,['source','row','task','candidate_id','precision','seed','export_sha256','accuracy','latency_ms','storage_bytes','native_memory_bytes','energy_mj','verification_flag'])
 savecsv('parsed_artifact_inventory.csv',inventory,['path','type','rows','columns','sha256'])
 savecsv('matched_training_contrasts.csv',contrasts,['contrast','task','seed','delta_pp','source_a','source_b'])
 groups=collections.defaultdict(list)
 for r in contrasts:groups[(r['contrast'],r['task'])].append(r['delta_pp'])
 contrast_summary=[]
 for (contrast,task),vals in sorted(groups.items()):
  contrast_summary.append({'contrast':contrast,'task':task,'n':len(vals),'mean_delta_pp':statistics.mean(vals),
   'sd_delta_pp':statistics.stdev(vals) if len(vals)>1 else None})
 savecsv('paired_contrast_summary.csv',contrast_summary,['contrast','task','n','mean_delta_pp','sd_delta_pp'])
 summary={'schema':'cajo_step48_evidence_decisions_v1','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'project_root':str(root),'manuscript':str(args.paper.resolve()) if args.paper else None,
  'config_sha256_match':config_ok,'gate_checks':checks,'sealed_test_access':gate.get('sealed_test_access'),
  'counts':{'parsed_artifacts':len(inventory),'numerical_rows':len(extracted),'step44_arms':len(arms),
   'paired_training_contrasts':len(contrasts),'step43_exports':len(candidates),'fp32_int8_pairs':len(quant),
   'baseline_rows':len(baselines),'historical_energy_rows':len(energy)},
  'decisions':decisions,'parse_errors':errors,'limitations':[
   'This audit inspects numerical file contents and source references, but does not execute experimental code or establish algorithmic independence by filename.',
   'Earlier numerical artifacts are candidate evidence until measurement scope, splits, and provenance are checked.',
   'An unverified gate does not prove a new experiment is required; except the eventual sealed test, experiment decisions remain conditional.',
   'No training, inference, checkpoints or test data were opened.']}
 (out/'evidence_decisions.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 lines=['# CAJO Step 48 — Evidence-to-results decisions','',
  f'Project: `{root}`','',f'Locked configuration SHA256 matches: **{config_ok}**','',
  'The decisions below identify whether existing results can be used, need analysis, need verification, or need a new experiment. They are not manuscript-readiness scores.','',
  '| ID | Manuscript requirement | Decision | Evidence-based explanation |','|---|---|---|---|']
 for d in decisions:
  lines.append('| '+' | '.join(str(d[k]).replace('|','/').replace('\n',' ') for k in ('requirement','paper_requirement','decision','evidence_observed'))+' |')
 lines+=['','## Important interpretation','',
  'A Step38 false gate is preserved, but does not itself establish that earlier results cannot be recovered.',
  'Do not repeat training to address an issue that can be resolved through paired analysis or source provenance.',
  'The final sealed test is required only after selection is frozen; it is not authorized by this audit.',
  'Consult `numerical_evidence_index.csv` and `parsed_artifact_inventory.csv` for earlier-stage results.',
  '','## Parsed file errors','']
 lines.extend(f"- `{e['path']}`: {e['error']}" for e in errors[:30])
 if not errors:lines.append('None.')
 (out/'evidence_decision_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({'output':str(out),'parsed_artifacts':len(inventory),'numeric_evidence_rows':len(extracted),
  'decisions':dict(collections.Counter(d['decision'] for d in decisions)),
  'parse_errors':len(errors),'config_matches':config_ok},indent=2))
if __name__=='__main__':main()
