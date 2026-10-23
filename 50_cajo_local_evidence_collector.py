#!/usr/bin/env python3
"""CAJO Step 50: local, read-only, bounded evidence collector for R02/R04/R10/R11/R12/R16.

Only JSON, CSV, text and Python source; no model loading, training, inference, or sealed test.
Produces compact JSON/CSV/Markdown and an optional ZIP of those summaries only.
"""
import argparse, ast, csv, hashlib, json, re, statistics, zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

TARGETS={
 'R02':{'dirs':['CAJO_Step35_Closure_v1/memory_raw','CAJO_Step37','CAJO_Step38','CAJO_Energy_Metrology'], 'terms':['memory','allocator','workspace','activation','packed','rss','uss','native','onednn','peak']},
 'R04':{'dirs':['CAJO_HWiNFO_Campaign_v1','CAJO_Energy_Metrology','CAJO_Step34_Audit','CAJO_Step35_Closure_v1/energy_raw','CAJO_Step37','CAJO_Step38'], 'terms':['energy','power','idle','hwinfo','joule','watt','timestamp','sensor','uncertainty']},
 'R10':{'dirs':['CAJO_Matched_Search','CAJO_Architecture_Search','CAJO_Final_Pipeline','Optimization_Protocol'], 'terms':['nas','architecture','arch_id','search','candidate','screening','fixed_arch','width','depth']},
 'R11':{'dirs':['CAJO_Matched_Search','CAJO_Independent_Closure_v1','CAJO_Final_Pipeline','CAJO_Step44_Controlled_Training','CAJO_Steps36_45_ResultsClosure_v1'], 'terms':['joint','sequential','policy','distill','kd','prun','quant','architecture','search']},
 'R12':{'dirs':['CAJO_Matched_Search','CAJO_Final_Pipeline','CAJO_Independent_Closure_v1','Optimization_Protocol','CAJO_Steps36_45_ResultsClosure_v1'], 'terms':['budget','epochs','candidate_count','compute','wall_time','runtime','gpu_hours','search','seed','executed']},
 'R16':{'dirs':['Baseline_Resource_Profiling','CAJO_Step43_Experimental_Execution/baselines','CAJO_INT8_CPU','CAJO_Final_Pipeline/CPU_Final'], 'terms':['baseline','accuracy','latency','memory','storage','energy','device','batch','thread','precision']},
}
SCRIPTS=['03_architecture_search_space','05_hardware_benchmark','16_cajo_gpu_hardware_benchmark','18_cajo_final_joint_training','22_cajo_int8_experiments','23_cajo_matched_joint_search','24_cajo_final_experiments_and_manuscript','26_cajo_final_closure','27_cajo_five_objective_meter','32_cajo_hwinfo_energy','35_cajo_one_pass_closure','37_cajo_targeted_memory_energy_acquisition','38_cajo_integrated_experimental_closure','43_','44_']
SKIP=re.compile(r'(?:^|[/\\_\-.])(sealed|held.?out|holdout|test.?images|test.?data|test.?loader|test.?predictions|test.?labels|test.?set)(?:$|[/\\_\-.])',re.I)
BAD_DIRS={'CAJO_Step46_Manuscript_Evidence_Audit','CAJO_Manuscript_Evidence_Audit','CAJO_Full_History_Audit','CAJO_Step48_Evidence_Decisions','CAJO_Step49_Targeted_Verification','CAJO_Step50_Local_Collection','CAJO_Results_Explorer','__pycache__'}
MAX_CELL=200; MAX_TEXT=600; MAX_SOURCE_LINES=160; MAX_FILES_PER_REQUIREMENT=120; MAX_ROWS_PER_FILE=120; MAX_BYTES=12*1024*1024

def textfile(p):
 for enc in ('utf-8-sig','cp1252','latin-1'):
  try:return p.read_text(encoding=enc),enc
  except UnicodeError:pass
 return None,None

def clip(v,n=MAX_CELL):
 if isinstance(v,(dict,list)):v=json.dumps(v,ensure_ascii=False,default=str)
 return str(v).replace('\x00','')[:n]

def safe(p,root):
 try:
  rel=p.relative_to(root).as_posix()
 except ValueError:return False
 return not SKIP.search(rel) and not any(x in BAD_DIRS for x in p.parts) and p.suffix.lower() in ('.json','.csv','.md','.txt','.log','.py') and p.stat().st_size<=MAX_BYTES

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def paths_for(root,rid):
 reports=root/'Reports';dirs=TARGETS[rid]['dirs'];res=set()
 for fragment in dirs:
  direct=reports/fragment
  if direct.exists():
   if direct.is_file():res.add(direct)
   else:
    for p in direct.rglob('*'):
     if p.is_file():res.add(p)
  else:
   head=fragment.split('/')[0]
   for d in reports.glob('*'+head+'*'):
    if d.is_dir():
     for p in d.rglob('*'):
      if p.is_file():res.add(p)
 # Strong direct anchors, regardless of folder naming
 for pat in ('*baseline*comparison*.csv','*screening*evidence*.csv','*energy*analysis*.json','*evidence*gate*.json','*combined*objectives*.json'):
  res.update(reports.rglob(pat))
 return sorted((p for p in res if safe(p,root)),key=lambda p:(0 if any(t in p.name.lower() for t in TARGETS[rid]['terms']) else 1,0 if p.suffix.lower() in ('.json','.csv') else 1,p.as_posix()))

def flatten(o,prefix='',depth=0):
 if depth>6:return
 if isinstance(o,dict):
  for k,v in o.items():yield from flatten(v,(prefix+'.' if prefix else '')+str(k),depth+1)
 elif isinstance(o,list):
  if o and all(not isinstance(x,(dict,list)) for x in o):yield prefix,o
  else:
   for i,v in enumerate(o[:12]):yield from flatten(v,prefix+'['+str(i)+']',depth+1)
 else:yield prefix,o

def extract(p,rid):
 terms=TARGETS[rid]['terms'];t,encoding=textfile(p)
 if t is None:return None
 result={'path':None,'encoding':encoding,'size_bytes':p.stat().st_size,'sha256':sha(p),'type':p.suffix.lower(),'total_records':None,'selected_rows':[],'selected_fields':[],'flags':[]}
 if p.suffix.lower()=='.json':
  try:o=json.loads(t)
  except Exception as e:return {**result,'error':'invalid JSON: '+str(e)}
  result['total_records']=len(o) if isinstance(o,list) else 1
  rows=o if isinstance(o,list) else [o]
  for i,record in enumerate(rows[:MAX_ROWS_PER_FILE]):
   f=list(flatten(record)); hits=[(k,v) for k,v in f if any(term in k.lower() for term in terms)]
   if hits:
    result['selected_rows'].append({'index':i,'fields':{k:clip(v) for k,v in hits[:35]}})
   if i==0:result['selected_fields']=[k for k,v in f[:100]]
 elif p.suffix.lower()=='.csv':
  try:
   reader=csv.DictReader(t.splitlines());result['selected_fields']=list(reader.fieldnames or [])
   total=0
   for i,row in enumerate(reader):
    total+=1
    if i<MAX_ROWS_PER_FILE:
     fields={k:clip(v) for k,v in row.items() if k and (any(term in k.lower() for term in terms) or k.lower() in ('task','dataset','seed','model','candidate','id','name','mode','policy','status','precision','path','source'))}
     if fields:result['selected_rows'].append({'index':i,'fields':dict(list(fields.items())[:35])})
   result['total_records']=total
  except Exception as e:return {**result,'error':'CSV parse: '+str(e)}
 else:
  lines=t.splitlines();result['total_records']=len(lines)
  result['selected_rows']=[{'line':i+1,'excerpt':clip(line,MAX_TEXT)} for i,line in enumerate(lines) if any(term in line.lower() for term in terms)][:MAX_ROWS_PER_FILE]
 if result['total_records'] and len(result['selected_rows'])<result['total_records']:result['flags'].append('SAMPLED_NOT_EXHAUSTIVE')
 return result

def code_snippets(p,rid):
 t,enc=textfile(p)
 if t is None:return {'error':'cannot decode'}
 terms=TARGETS[rid]['terms']; lines=t.splitlines()
 try:
  tree=ast.parse(t); functions=[]
  for node in ast.walk(tree):
   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
    body='\n'.join(lines[node.lineno-1:min(len(lines),node.end_lineno or node.lineno)])
    if any(term in (node.name+' '+body).lower() for term in terms):functions.append({'name':node.name,'start':node.lineno,'end':node.end_lineno})
 except SyntaxError as e:functions=[{'parse_error':str(e)}]
 hits=[i for i,line in enumerate(lines) if any(re.search(r'\b'+re.escape(term),line,re.I) for term in terms)]
 included=set()
 for i in hits[:40]:included.update(range(max(0,i-2),min(len(lines),i+3)))
 selected=[{'line':i+1,'text':clip(lines[i],MAX_TEXT)} for i in sorted(included)[:MAX_SOURCE_LINES]]
 return {'sha256':sha(p),'lines':len(lines),'functions':functions[:35],'source_excerpts':selected,'truncated':len(included)>MAX_SOURCE_LINES}

def write_csv(p,rows,cols):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
 global MAX_ROWS_PER_FILE
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument('--root',type=Path,default=Path('..'));a.add_argument('--out',type=Path)
 a.add_argument('--max-files',type=int,default=MAX_FILES_PER_REQUIREMENT)
 a.add_argument('--max-rows',type=int,default=MAX_ROWS_PER_FILE)
 args=a.parse_args();root=args.root.resolve();out=(args.out or root/'Reports'/'CAJO_Step50_Local_Collection').resolve()
 if not (root/'Codes').is_dir() or not (root/'Reports').is_dir():a.error('--root must contain Codes and Reports')
 if args.max_files<1 or args.max_rows<1:a.error('max-files and max-rows must be positive')
 MAX_ROWS_PER_FILE=args.max_rows
 out.mkdir(parents=True,exist_ok=True)
 data={'schema':'cajo_step50_local_evidence_collector_v1','created_utc':datetime.now(timezone.utc).isoformat(),'root':str(root),'safety':'read-only, no sealed test or binary checkpoints','requirements':{},'code':{},'issues':[]}
 inventory=[];source_rows=[];used=set()
 for rid in TARGETS:
  paths=paths_for(root,rid); chosen=paths[:args.max_files];records=[]
  for p in chosen:
   rel=p.relative_to(root).as_posix()
   if rel in used:pass
   try:
    r=extract(p,rid)
    if r is None:continue
    r['path']=rel;records.append(r)
    inventory.append({'requirement':rid,'source':rel,'bytes':r['size_bytes'],'sha256':r['sha256'],'records':r['total_records'],'sampled_rows':len(r['selected_rows']),'flags':';'.join(r['flags']),'error':r.get('error','')})
    for row in r['selected_rows']:
     source_rows.append({'requirement':rid,'source':rel,'row_or_line':row.get('index',row.get('line')),'evidence':clip(row.get('fields',row.get('excerpt','')),500)})
   except Exception as e:data['issues'].append({'requirement':rid,'path':rel,'error':str(e)})
  data['requirements'][rid]={'candidate_file_count':len(paths),'collected_file_count':len(records),'omitted_file_count':max(0,len(paths)-len(chosen)),'records':records}
  if len(paths)>len(chosen):data['issues'].append({'requirement':rid,'error':'File cap reached; rerun --max-files higher or inspect omitted paths'})
 for p in sorted((root/'Codes').glob('*.py')):
  if not any(p.name.startswith(x) for x in SCRIPTS):continue
  if not safe(p,root):continue
  rel=p.relative_to(root).as_posix();data['code'][rel]={rid:code_snippets(p,rid) for rid in TARGETS}
 (out/'collected_evidence.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
 write_csv(out/'source_inventory.csv',inventory,['requirement','source','bytes','sha256','records','sampled_rows','flags','error'])
 write_csv(out/'evidence_excerpts.csv',source_rows,['requirement','source','row_or_line','evidence'])
 (out/'source_code_excerpts.json').write_text(json.dumps(data['code'],indent=2,ensure_ascii=False),encoding='utf-8')
 lines=['# CAJO Step 50 — Local evidence collection','', 'Read-only collection. **Source matches and field excerpts do not certify the experimental claims.**','', '| Requirement | Candidate files | Files inspected | Sampled evidence rows |', '|---|---:|---:|---:|']
 for rid,x in data['requirements'].items():lines.append(f"| {rid} | {x['candidate_file_count']} | {x['collected_file_count']} | {sum(len(r['selected_rows']) for r in x['records'])} |")
 lines+=['','## Interpretation','', 'For R02, verify that a simultaneous native CPU inference peak includes packed weights, activations and workspaces.','For R04, verify timestamp alignment, idle correction, sensor resolution, and uncertainty; do not substitute GPU power.','For R10/R11/R12, compare executed code paths, selected architectures, seed/split equality and actual budget logs.','For R16, verify identical CPU measurement conditions and distinguish reference-only baseline accuracy.','',f"Issues: {len(data['issues'])}. Full details are in collected_evidence.json."]
 (out/'collection_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 zip_path=out/'CAJO_Step50_Compact_Evidence.zip'
 with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
  for name in ('collected_evidence.json','source_inventory.csv','evidence_excerpts.csv','source_code_excerpts.json','collection_summary.md'):z.write(out/name,name)
 print('Output:',out);print('Compact ZIP:',zip_path,'bytes:',zip_path.stat().st_size)
 for rid,x in data['requirements'].items():print(rid,'collected',x['collected_file_count'],'of',x['candidate_file_count'])
 print('Issues:',len(data['issues']))

if __name__=='__main__':main()
