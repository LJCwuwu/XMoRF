"""Compare all four components and three final outputs to original cached scores."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from xmorf.predict import Ensemble,read_fasta,load_features

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--weights',required=True)
 ap.add_argument('--examples',default='examples')
 ap.add_argument('--device',default='cpu')
 ap.add_argument('--report',default='verification.json')
 args=ap.parse_args();torch.set_num_threads(4)
 root=Path(args.examples);records=read_fasta(root/'proteins.fasta')
 z=np.load(root/'expected.npz',allow_pickle=False);model=Ensemble(args.weights,args.device)
 assert [p for p,s in records]==z['pid'].tolist()
 assert [len(s) for p,s in records]==z['length'].tolist()
 rows=[];offset=0
 for pid,seq in records:
  pred=model.predict(seq,load_features(seq,root/'features',model.device))
  errors={k:float(np.max(np.abs(v-z[k][offset:offset+len(seq)]))) for k,v in pred.items()}
  if max(errors.values())>1e-5:raise AssertionError(f'Prediction mismatch for {pid}: {errors}')
  rows.append({'protein_id':pid,'length':len(seq),'max_absolute_error':errors})
  offset+=len(seq)
 result={'passed':True,'torch':torch.__version__,'device':args.device,'models':72,'proteins':len(rows),
         'residues':offset,'tolerance':1e-5,'rows':rows,
         'numerics':'A separate test establishes bitwise equivalence to the original forward pass on the same CPU. This tolerance compares CPU inference with the original saved GPU predictions.',
         'scope':'Inference from supplied cached float16 embeddings. New PLM extraction and full retraining are not covered.'}
 Path(args.report).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
