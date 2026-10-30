"""Read real RLDS TFRecords and reconstruct the historical cache key on CPU.

Uses the actual pinned ActionTokenizer and PurePromptBuilder, local tokenizer,
recorded RLDS normalization, complete 8-step chunks, and int64 token bytes.
No model weights or teacher inference. Source data and cache files remain unchanged.
The output contains aggregate counts and witness locations, without raw images.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_NUM_INTRAOP_THREADS']='2';os.environ['TF_NUM_INTEROP_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2';os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,hashlib,importlib.util,json,gc,time,ast
import dlimp as dl
from types import SimpleNamespace
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import tensorflow as tf
from transformers import AutoTokenizer

def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--oft-repo',required=True, type=Path)
parser.add_argument('--data-root',required=True, type=Path)
parser.add_argument('--checkpoint-root',required=True, type=Path)
parser.add_argument('--cache-root',required=True, type=Path)
parser.add_argument('--output-dir',required=True, type=Path)
parser.add_argument('--per-shard',type=int,default=0,help='0 scans every record; positive values sample each shard')
parser.add_argument('--suite',nargs='+',choices=['libero_object','libero_spatial','libero_goal','libero_10'],default=['libero_object','libero_spatial','libero_goal','libero_10'])
args=parser.parse_args()
if args.per_shard < 0:parser.error('--per-shard must be nonnegative')
ROOT=args.oft_repo.expanduser().resolve()
from autoprune.prepare_oft import verify_checkout
verify_checkout(ROOT)
ActionTokenizer=module(ROOT/'prismatic/vla/action_tokenizer.py','audit_action').ActionTokenizer
PurePromptBuilder=module(ROOT/'prismatic/models/backbones/llm/prompting/base_prompter.py','audit_prompt').PurePromptBuilder
# Execute the actual normalization function to retain NumPy/TF dtype behavior.
source_path=ROOT/'prismatic/vla/datasets/rlds/utils/data_utils.py'
fn=next(n for n in ast.parse(source_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='normalize_action_and_proprio')
ns={'tf':tf,'dl':dl,'NormalizationType':SimpleNamespace(NORMAL='normal',BOUNDS='bounds',BOUNDS_Q99='bounds_q99')}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])),str(source_path),'exec'),ns)
normalize=ns['normalize_action_and_proprio']
OUT=args.output_dir.expanduser().resolve();OUT.mkdir(parents=True,exist_ok=False)
DATA=args.data_root.expanduser().resolve();BASE=args.checkpoint_root.expanduser().resolve();CACHE=args.cache_root.expanduser().resolve()

def digest(*pieces):
 h=hashlib.sha256()
 for p in pieces:h.update(p)
 return h.hexdigest()

def scan(suite,per_shard):
 started=time.time();name=suite+'_no_noops';folder=DATA/name/'1.0.0'
 stats_files=list(folder.glob('dataset_statistics_*.json'));assert len(stats_files)==1
 stats=json.loads(stats_files[0].read_text())
 stats_np={key:{k:np.array(v) for k,v in stats[key].items()} for key in ['action','proprio']}
 stats_np['action']['mask']=np.array([True]*6+[False])
 short=suite.replace('libero_','');ckpt=BASE/('openvla-7b-oft-finetuned-libero-'+short)
 tok=AutoTokenizer.from_pretrained(str(ckpt),local_files_only=True)
 action_tok=ActionTokenizer(tok)
 keys=set();shards=sorted((CACHE/name).glob('cache_shard_*.pt'))
 for shard in shards:
  cache=torch.load(str(shard),map_location='cpu',mmap=True,weights_only=True);keys.update(cache);del cache;gc.collect()
 metadata=torch.load(CACHE/name/'metadata.pt',map_location='cpu',weights_only=False)
 if metadata.get("format")=="autoprune-teacher-v2":
  raise ValueError("This diagnostic audits legacy token-only caches; v2 requires stable source IDs and must not be interpreted with legacy hashes")
 print(suite,'cache keys',len(keys),'loaded',flush=True)
 seen={};ambiguous=set();episodes=0;examples=[];step_total=0;languages=Counter();shard_counts={};cache_hits=0;missing_by_language=Counter()
 lows=np.array(stats['action']['q01'],dtype=np.float32);highs=np.array(stats['action']['q99'],dtype=np.float32)
 zeros=np.array(stats['action']['min'])==np.array(stats['action']['max'])
 for shard_idx,path in enumerate(sorted(folder.glob('*.tfrecord-*'))):
  taken=0
  ds=tf.data.TFRecordDataset(str(path));opts=tf.data.Options();opts.threading.private_threadpool_size=1;ds=ds.with_options(opts)
  for record_idx,record in enumerate(ds):
   if per_shard and taken>=per_shard:break
   fields=tf.train.Example.FromString(bytes(record.numpy())).features.feature
   act=np.array(fields['steps/action'].float_list.value,dtype=np.float32).reshape(-1,7)
   state=np.array(fields['steps/observation/state'].float_list.value,dtype=np.float32).reshape(-1,8)
   main=fields['steps/observation/image'].bytes_list.value;wrist=fields['steps/observation/wrist_image'].bytes_list.value
   langs=fields['steps/language_instruction'].bytes_list.value;n=len(act)
   assert n==len(state)==len(main)==len(wrist)==len(langs)
   act[:,-1]=1-np.clip(act[:,-1],0,1)
   normalized=normalize({'action':tf.constant(act)},stats_np,'bounds_q99')['action'].numpy()
   valid=[i for i in range(max(0,n-7)) if langs[i].strip()]
   prompts=[]
   for i in valid:
    chunk=normalized[i:i+8];action_string=action_tok(chunk[0])+''.join(action_tok(chunk[1:]))
    prompt=PurePromptBuilder('openvla');lang=langs[i].decode().lower()
    prompt.add_turn('human',f'What action should the robot take to {lang}?');prompt.add_turn('gpt',action_string);prompts.append(prompt.get_prompt())
   ids_list=tok(prompts,add_special_tokens=True,truncation=True,max_length=tok.model_max_length)['input_ids'] if prompts else []
   for i,ids in zip(valid,ids_list):
    key=hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest();cache_hit=key in keys;cache_hits+=cache_hit
    if not cache_hit:missing_by_language[langs[i].decode()]+=1
    obs=digest(state[i].tobytes(),main[i],wrist[i]);state_hash=digest(state[i].tobytes());images_hash=digest(main[i],wrist[i])
    loc={'shard':path.name,'record':record_idx,'step':i};step_total+=1;languages[langs[i].decode()]+=1
    if key not in seen:
     seen[key]={'observation':obs,'state':state_hash,'images':images_hash,'count':1,'location':loc,'state_values':state[i].tolist()}
    else:
     entry=seen[key];entry['count']+=1
     if obs!=entry['observation']:
      fresh=key not in ambiguous;ambiguous.add(key)
      if fresh and len(examples)<5:
       example={'key':key,'cache_hit':cache_hit,'first':entry['location'],'second':loc,'state_different':state_hash!=entry['state'],'encoded_images_different':images_hash!=entry['images'],'max_abs_state_delta':float(np.max(np.abs(state[i]-np.array(entry['state_values']))))}
       examples.append(example)
       # Save real witness inputs for follow-up exact pipeline checks (private local artifacts).
   taken+=1;episodes+=1
  shard_counts[path.name]=taken
  if (shard_idx+1)%8==0:print(suite,'shards',shard_idx+1,'episodes',episodes,'samples',step_total,'ambiguous keys',len(ambiguous),flush=True)
 ambiguous_samples=sum(seen[k]['count'] for k in ambiguous)
 result={'suite':suite,'sampling':'all records' if not per_shard else f'first {per_shard} records of every shard','episodes':episodes,'action_chunk_samples':step_total,'unique_token_keys':len(seen),'duplicate_key_samples':step_total-len(seen),'ambiguous_observation_keys':len(ambiguous),'samples_in_ambiguous_groups':ambiguous_samples,'ambiguous_groups_present_in_cache':len(ambiguous&keys),'samples_matching_existing_cache':cache_hits,'cache_hit_fraction':cache_hits/max(1,step_total),'existing_cache_key_count':len(keys),'metadata':{k:metadata.get(k) for k in ['image_aug','num_patches','use_proprio','num_samples','num_actions_chunk','action_dim']},'missing_by_language':dict(missing_by_language),'distinct_task_instructions':len(languages),'witnesses':examples,'normalization_file_sha256':digest(stats_files[0].read_bytes()),'tokenizer_json_sha256':digest((ckpt/'tokenizer.json').read_bytes()),'seconds':round(time.time()-started,2),'normalization_implementation':'Actual pinned normalize_action_and_proprio function with original NumPy metadata dtypes','scope':'Raw observation distinction; no teacher output differences or robot behavior inferred.'}
 (OUT/f'{suite}_results.json').write_text(json.dumps(result,indent=2)+'\n');print('RESULT',json.dumps(result),flush=True)
 return result

if __name__=='__main__':
 result=[scan(s,args.per_shard) for s in args.suite]
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
