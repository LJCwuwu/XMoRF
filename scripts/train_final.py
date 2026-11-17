"""Run one original final-model training job with the released fold assignment."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from morf.config import Config


def load_configs(role, fold, seed):
 if role not in {'nested','core','region'} or fold not in range(4) or seed not in range(6):
  raise ValueError('Expected role nested/core/region, fold 0-3 and seed 0-5')
 names=['nested_r.json','nested_q.json'] if role=='nested' else [f'{role}.json']
 configs=[]
 for name in names:
  c=Config.load(str(ROOT/'configs/training'/name))
  c.fold=fold;c.seed=seed
  configs.append(c)
 return configs


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--role',choices=['nested','core','region'],required=True)
 ap.add_argument('--fold',type=int,choices=range(4),required=True)
 ap.add_argument('--seed',type=int,choices=range(6),required=True)
 ap.add_argument('--data',required=True)
 ap.add_argument('--features')
 ap.add_argument('--output',default='training_outputs')
 ap.add_argument('--device',default='cuda')
 ap.add_argument('--check-only',action='store_true')
 a=ap.parse_args();data=Path(a.data).resolve();out=Path(a.output).resolve()/a.role
 from morf.data.parse import read_any
 files={'TRAIN':str(data/'training/TRAIN421.txt'),
        'MC2EXT':[str(data/f'training/MC2-ext/CV{i}.af') for i in range(1,5)]}
 cl=read_any(files['TRAIN'],verbose=False);mc2=read_any(files['MC2EXT'],verbose=False)
 assert len(cl)==421 and len(mc2)==444
 split=json.loads((ROOT/'configs/train421_folds.json').read_text())
 assert set(p.key for p in cl)==set(split['fold_of_key'])
 cfgs=load_configs(a.role,a.fold,a.seed)
 for c in cfgs:
  assert c.fold==a.fold and c.seed==a.seed and c.spec_mode=='global' and c.spec_freq_step==360.0
  assert not c.finetune and not c.init_from and not c.cdan_target and not c.adv
  c.raw_dir='';c.data_files=files;c.out_dir=str(out);c.eval_sets=[];c.n_boot=0;c.device=a.device
  if a.features:c.feat_dir=str(Path(a.features).resolve())
 print(json.dumps({'role':a.role,'fold':a.fold,'seed':a.seed,'TRAIN421':len(cl),'MC2-ext':len(mc2),
       'classic_validation_records':sum(split['fold_of_key'][p.key]==a.fold for p in cl),
       'region_validation_records':sum(p.group==f'CV{a.fold+1}' for p in mc2)},indent=2))
 if a.check_only:return
 if not a.features:ap.error('--features is required for training')
 from morf.train.trainer import run_training
 from morf.train.nested import train_nested
 # Each fold/seed job has a new output tree. The original project is never written.
 run=out/'runs'/f"{a.role}_f{a.fold}_s{a.seed}"
 if run.exists():raise FileExistsError(f'Choose a new output directory: {run}')
 out.mkdir(parents=True,exist_ok=True)
 foldfile=out/'cv4_seed0.json';raw=(ROOT/'configs/train421_folds.json').read_bytes()
 if foldfile.exists():assert foldfile.read_bytes()==raw,'Existing fold assignment differs'
 else:foldfile.write_bytes(raw)
 if a.role=='nested':
  train_nested(cfgs[0],cfgs[1],0,{'cl':cl,'mc2':mc2},str(run),
               {'cand':'S0','k_stop':0,'r_mode':'global','source':'paper final model'})
 else:
  c=cfgs[0];c.tag=a.role
  # The original trainer appends _f{fold}_s{seed} to this tag.
  run_training(c)


if __name__=='__main__':main()
