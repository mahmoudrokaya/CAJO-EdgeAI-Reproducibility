#!/usr/bin/env python3
"""Read-only, whole-project CAJO evidence discovery and manuscript mapping.
Python 3.10+, stdlib only. Never runs scripts, loads model binaries or reads sealed tests.
Use alongside Step46: this audits *all* earlier stages instead of only known tables.
"""
import argparse, collections, csv, datetime as dt, hashlib, json, os, re, sys, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

REQUIREMENTS={
 'R01':('Accuracy and split integrity',('validation','accuracy','split','holdout')),
 'R02':('Complete native inference memory',('native','memory','packed','workspace','allocator')),
 'R03':('CPU latency',('latency','benchmark','warmup','oneDNN')),
 'R04':('Energy measurement and uncertainty',('energy','power','hwinfo','idle','uncertainty')),
 'R05':('Hardware/data/teacher admissibility',('eligibility','admissibility','constraint','teacher')),
 'R06':('Layerwise pruning, quantization, architecture provenance',('pruning','architecture','bitwidth','quantization','candidate')),
 'R07':('KD-only ablation',('distillation','kd','ablation')),
 'R08':('Pruning-only ablation',('pruning','ablation')),
 'R09':('Quantization-only ablation',('quantization','int8','fp32','ptq')),
 'R10':('NAS-only ablation',('nas','architecture','ablation')),
 'R11':('Independent joint vs sequential',('joint','sequential','independent','training')),
 'R12':('Matched search budgets',('search','budget','candidate','seed')),
 'R13':('Five verified objective values',('five_objective','objective','validation','energy','memory')),
 'R14':('Five-dimensional Pareto and feasibility',('pareto','feasible','dominan')),
 'R15':('Frozen selection and tie break',('selection','freeze','tie_break')),
 'R16':('Matched baseline comparisons',('baseline','matched','comparison')),
 'R17':('Seed-level statistical uncertainty',('paired','confidence','bootstrap','statistic','seed')),
 'R18':('Sealed final test after freeze',('sealed','final_test','heldout')),
 'R19':('Validation vs test wording',('validation','test')),
 'R20':('Selection-rule wording',('selection','energy','error')),
 'R21':('Energy instrumentation wording',('power','voltage','current','sensor')),
 'R22':('Experimental declarations',('data availability','code availability','declarations')),
}
SKIP_DIR={'__pycache__','.git','.venv','venv','node_modules','site-packages','CAJO_Full_History_Audit'}
TEST_RE=re.compile(r'(?:^|[/_\\-])(sealed[_-]?test|held[_-]?out[_-]?test|test[_-]?images|test[_-]?dataset)(?:[/_\\.-]|$)',re.I)
TASKS=('CIFAR10','CIFAR100','SpeechCommands_v0.02')
MAX_PARSE=8*1024*1024

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):
 if p.stat().st_size>MAX_PARSE:return None,'too_large'
 ext=p.suffix.lower()
 try:
  if ext=='.json':return json.loads(p.read_text(encoding='utf-8-sig')),'json'
  if ext in ('.csv','.tsv'):
   with p.open(encoding='utf-8-sig',newline='') as f:
    r=csv.DictReader(f,delimiter='\t' if ext=='.tsv' else ',')
    cols=r.fieldnames or []
    rows=[]
    for i,row in enumerate(r):
     if i<5:rows.append(row)
    return {'columns':cols,'sample_rows':rows,'row_count':i+1 if 'i' in locals() else 0},'table'
  if ext in ('.md','.txt','.log'):
   return p.read_text(encoding='utf-8-sig',errors='replace')[:50000],'text'
 except (ValueError,UnicodeError,OSError,csv.Error,json.JSONDecodeError) as e:return None,str(e)
 return None,'not_parsed'
def flatten_keys(x,prefix='',depth=0):
 if depth>5:return []
 if isinstance(x,dict):
  out=[]
  for k,v in list(x.items())[:100]:
   q=(prefix+'.' if prefix else '')+str(k)
   out.append(q)
   if isinstance(v,(dict,list)):out+=flatten_keys(v,q,depth+1)
  return out[:400]
 if isinstance(x,list) and x and isinstance(x[0],dict):return flatten_keys(x[0],prefix+'[]',depth+1)
 return []
def summary_data(d):
 if isinstance(d,dict):
  if 'columns' in d and 'row_count' in d:return {'rows':d['row_count'],'columns':d['columns'][:100]}
  return {'keys':flatten_keys(d)[:160]}
 if isinstance(d,list):return {'items':len(d),'keys':flatten_keys(d)[:120]}
 return {'text_chars':len(d)} if isinstance(d,str) else {}
def keywords(p,data):
 text=p.lower()+' '+json.dumps(summary_data(data),ensure_ascii=False).lower()
 # Text body is not trusted as verified results; only searchable hints.
 if isinstance(data,str):text+=' '+data[:15000].lower()
 return text
def stage(path):
 m=re.search(r'(?<!\d)(\d{2})(?:[_-]|\b)',path)
 return int(m.group(1)) if m and int(m.group(1))<=99 else None
def manuscript(path):
 if not path:return ''
 if path.suffix.lower()!='.docx':return ''
 with zipfile.ZipFile(path) as z:
  r=ET.fromstring(z.read('word/document.xml'))
 ns='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
 return '\n'.join(''.join(t.text or '' for t in p.iter(ns+'t')) for p in r.iter(ns+'p'))
def dump_csv(p,rows,fields):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser(description='Whole-history read-only CAJO report and code evidence audit')
 ap.add_argument('--root',type=Path,default=Path('..'))
 ap.add_argument('--paper',type=Path)
 ap.add_argument('--out',type=Path)
 args=ap.parse_args()
 root=args.root.resolve();reports=root/'Reports';codes=root/'Codes'
 if not reports.is_dir():ap.error('Missing Reports: '+str(reports))
 out=(args.out or reports/'CAJO_Full_History_Audit').resolve();out.mkdir(parents=True,exist_ok=True)
 records=[];issues=[];sources=[]
 for base,dirs,files in os.walk(root):
  here=Path(base)
  dirs[:]=[d for d in dirs if d not in SKIP_DIR and not (here==root and d not in ('Reports','Codes'))]
  for filename in files:
   p=here/filename;rel=p.relative_to(root).as_posix()
   if TEST_RE.search(rel):continue
   ext=p.suffix.lower()
   if ext not in ('.json','.csv','.tsv','.md','.txt','.log','.py','.png','.jpg','.svg','.html','.parquet','.pt','.pth','.onnx','.zip','.xlsx'):continue
   try:sz=p.stat().st_size
   except OSError:continue
   d,typ=load(p) if ext in ('.json','.csv','.tsv','.md','.txt','.log') else (None,'not_parsed')
   info=summary_data(d)
   if ext=='.py':
    try:
     code=p.read_text(encoding='utf-8',errors='replace') if sz<MAX_PARSE else ''
     # Static function names are not execution proof.
     info={'functions':re.findall(r'^def\s+([A-Za-z_]\w*)\s*\(',code,re.M)[:80],
           'entrypoint':bool(re.search(r'__name__\s*==\s*[\'\"]__main__',code))}
    except OSError:pass
   text=keywords(rel,d)+' '+json.dumps(info,ensure_ascii=False).lower()
   matches={rid:sum(bool(re.search(r'(?<![a-z])'+re.escape(w.lower())+r'(?![a-z])',text)) for w in words) for rid,(_,words) in REQUIREMENTS.items()}
   matched=[rid for rid,n in matches.items() if n>=2]
   row={'path':rel,'stage':stage(rel),'extension':ext,'bytes':sz,'parse_type':typ,
        'rows':info.get('rows'),'columns':'; '.join(map(str,info.get('columns',[]))),
        'keys':'; '.join(map(str,info.get('keys',[]))),
        'functions':'; '.join(info.get('functions',[])),
        'matches':';'.join(matched),'sha256':sha(p) if sz<=MAX_PARSE else None}
   records.append(row)
   if matched:
    for rid in matched:sources.append({'requirement':rid,'path':rel,'stage':row['stage'],'kind':ext,
       'strength':'SOURCE_CODE_ONLY' if ext=='.py' else 'POTENTIAL_EVIDENCE',
       'reason':'Keyword/schema match; verify data and execution provenance'})
   if typ not in ('json','table','text','not_parsed','too_large'):issues.append({'path':rel,'error':typ})
 # Explicit evidence anchors from previous audits; never infer from name alone.
 known={
 'R01':['CAJO_Step43_Experimental_Execution/measured_validation_latency_storage.csv'],
 'R03':['CAJO_Step43_Experimental_Execution/measured_validation_latency_storage.csv'],
 'R07':['CAJO_Step44_Controlled_Training'],
 'R08':['CAJO_Step44_Controlled_Training'],
 'R09':['CAJO_Step43_Experimental_Execution/paired_precision_ablation.csv'],
 'R14':['CAJO_Step43_Experimental_Execution/provisional_three_objective_pareto.csv'],
 'R16':['Baseline_Resource_Profiling/baseline_resource_comparison.csv'],
 'R04':['CAJO_Step34_Audit/energy_models.csv'],
 'R13':['CAJO_Step38_Integrated_Closure_v1/validated_objectives_input.csv'],
 'R18':['CAJO_Step38_Integrated_Closure_v1/evidence_gate.json'],
 }
 gatepath=reports/'CAJO_Step38_Integrated_Closure_v1/evidence_gate.json'
 gate={}
 if gatepath.is_file():
  try:gate=json.loads(gatepath.read_text(encoding='utf-8-sig'))
  except (ValueError,OSError):pass
 checks=gate.get('checks',{}) if isinstance(gate,dict) else {}
 # Manuscript checks: indicate relevant phrases, not assert semantic equivalence.
 paper_text=''
 if args.paper:
  try:paper_text=manuscript(args.paper.resolve())
  except (OSError,ValueError,KeyError,zipfile.BadZipFile) as e:issues.append({'path':str(args.paper),'error':str(e)})
 matrix=[]
 for rid,(claim,words) in REQUIREMENTS.items():
  relevant=[r for r in sources if r['requirement']==rid]
  unique=sorted({r['path'] for r in relevant})
  anchors=[p for p in known.get(rid,[]) if (reports/p).exists()]
  relevant_stages=sorted({r['stage'] for r in relevant if r['stage'] is not None})
  matrix.append({'requirement':rid,'manuscript_requirement':claim,
   'known_evidence_anchors':'; '.join(anchors),
   'earlier_stage_candidates':'; '.join(str(n) for n in relevant_stages if n<43),
   'candidate_source_count':len(unique),
   'candidate_sources_top_20':'; '.join(unique[:20]),
   'audit_decision':'REQUIRES_CONTENT_AND_PROVENANCE_REVIEW',
   'note':'Earlier sources are candidates, not automatically missing or certified.'})
 # Evidence gate values are authoritative where available.
 gate_map={'R02':'native_memory','R04':'energy','R11':'independent_algorithm_comparison',
           'R13':'five_objective_complete','R14':'five_objective_complete','R16':'baselines_ablations'}
 for r in matrix:
  g=gate_map.get(r['requirement'])
  if g and g in checks:
   r['audit_decision']='GATE_'+('PASS' if checks[g] else 'UNVERIFIED')
   r['note']+=' Step38 '+g+'='+str(checks[g])+'.'
 output={'schema':'cajo_full_history_evidence_audit_v1','generated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
  'project_root':str(root),'manuscript':str(args.paper) if args.paper else None,
  'files_scanned':len(records),'source_links':len(sources),
  'stages_found':dict(sorted(collections.Counter(str(r['stage']) if r['stage'] is not None else 'unassigned' for r in records).items())),
  'gate_checks':checks,'sealed_test_access':gate.get('sealed_test_access'),
  'requirements':matrix,'parse_issues':issues,
  'limitations':['Every Reports/ and Codes/ stage is scanned; matches are discovery leads, not proof.',
   'Scripts are never executed. Checkpoints and sealed-test files are not opened.',
   'Gate failures are preserved. An older artifact does not automatically satisfy a later protocol.',
   'Review candidate evidence rows before declaring any new experiment necessary.']}
 (out/'full_history_audit.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 dump_csv(out/'all_project_artifacts.csv',records,['path','stage','extension','bytes','parse_type','rows','columns','keys','functions','matches','sha256'])
 dump_csv(out/'manuscript_evidence_matrix.csv',matrix,['requirement','manuscript_requirement','known_evidence_anchors','earlier_stage_candidates','candidate_source_count','candidate_sources_top_20','audit_decision','note'])
 dump_csv(out/'evidence_source_links.csv',sources,['requirement','path','stage','kind','strength','reason'])
 lines=['# CAJO full-history manuscript evidence audit','',f'Project: `{root}`','',
        f'Artifacts indexed: **{len(records)}**; potential requirement-source links: **{len(sources)}**.','',
        'This includes **all stages**, not only Steps 43–44. Source matches are not evidence certification.','',
        '| Requirement | Earlier stages discovered | Potential sources | Existing gate |','|---|---|---:|---|']
 for r in matrix:
  lines.append('| '+ ' | '.join([r['requirement'],r['earlier_stage_candidates'] or '—',str(r['candidate_source_count']),r['audit_decision']])+' |')
 lines+=['','## Interpretation','',
 'Use `manuscript_evidence_matrix.csv` to locate potential prior results, then `evidence_source_links.csv` to trace the exact files.','',
 'A matching filename, code function, or report key does not establish a completed experiment or its validity.','',
 'Preserve the Step38 evidence gates and the sealed-test restriction. No new experiments should be ordered until existing relevant source files are inspected.']
 (out/'full_history_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({'files_scanned':len(records),'evidence_links':len(sources),
  'report':str(out/'full_history_audit.md'),'matrix':str(out/'manuscript_evidence_matrix.csv'),
  'gate_checks':checks,'sealed_test_access':gate.get('sealed_test_access')},indent=2))
if __name__=='__main__':sys.exit(main())
