"""Full-protein inference from cached ProtT5 and ProstT5 residue embeddings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from .model import Predictor,NestedPredictor


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read_fasta(path):
    records=[]
    for line in Path(path).read_text().splitlines():
        line=line.strip()
        if not line or line.startswith(';'):
            continue
        if line.startswith('>'):
            fields=line[1:].split()
            if not fields:raise ValueError('Empty FASTA identifier')
            records.append([fields[0],''])
        else:
            if not records:raise ValueError('Sequence before FASTA header')
            records[-1][1]+=re.sub(r'\s+','',line).upper()
    if not records or any(not seq or re.fullmatch('[A-Z]+',seq) is None for _,seq in records):
        raise ValueError('Expected a nonempty, sequence-only protein FASTA file')
    if len({pid for pid,_ in records})!=len(records):raise ValueError('FASTA identifiers must be unique')
    return records


def feature_key(sequence):
    return hashlib.md5(sequence.encode()).hexdigest()


def load_features(sequence,directory,device):
    key=feature_key(sequence)
    out={}
    for name in ['prott5','prostt5']:
        path=Path(directory)/name/(key+'.npy')
        a=np.load(path,allow_pickle=False)
        if a.shape!=(len(sequence),1024) or a.dtype!=np.float16 or not np.isfinite(a).all():
            raise ValueError(f'Invalid feature shape, dtype or values: {path}')
        out[name]=torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).to(device)
    return out


def smooth(prob,window):
    if window<=1:return prob
    if window%2!=1:raise ValueError('Smoothing windows must be odd')
    return np.convolve(np.pad(prob,(window//2,window//2),mode='edge'),np.ones(window)/window,mode='valid').astype(prob.dtype)


def fuse(core,region,weight=0.35):
    def logit(p):
        p=np.clip(np.asarray(p,np.float64),1e-6,1-1e-6)
        return np.log(p)-np.log1p(-p)
    return 1/(1+np.exp(-(weight*logit(core)+(1-weight)*logit(region))))


class Ensemble:
    def __init__(self,weights,device='cpu'):
        self.directory=Path(weights)
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        self.device=torch.device(device)
        entries=self.manifest['models']
        expected={(r,f,s) for r in ['nested','core','region'] for f in range(4) for s in range(6)}
        if len(entries)!=72 or {(x['role'],x['fold'],x['seed']) for x in entries}!=expected:
            raise ValueError('Expected all 72 final checkpoints, exactly once')
        self.models=[]
        for e in entries:
            path=(self.directory/e['path']).resolve()
            if not path.is_relative_to(self.directory.resolve()):raise ValueError('Unsafe checkpoint path')
            if sha256(path)!=e['sha256']:raise ValueError(f'Checkpoint checksum mismatch: {path.name}')
            model=NestedPredictor() if e['role']=='nested' else Predictor()
            state=torch.load(path,map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True)
            if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite checkpoint value')
            self.models.append((e,model.to(self.device).float().eval()))

    @torch.inference_mode()
    def predict(self,sequence,features):
        mask=torch.ones((1,len(sequence)),device=self.device)
        components={k:[] for k in ['rq','r','q_prime','r_prime']}
        for e,model in self.models:
            if e['role']=='nested':
                a,b=model(features,mask)
                r=torch.sigmoid(a[0].float()).cpu().numpy().astype(np.float64)
                q=torch.sigmoid(b[0].float()).cpu().numpy().astype(np.float64)
                components['r'].append(smooth(r,e['window']['region']))
                components['rq'].append(smooth(r*q,e['window']['core']))
            else:
                prob=torch.sigmoid(model(features,mask)[0].float()).cpu().numpy()
                # Original independent-head smoothing occurs in float32 before averaging in float64.
                prob=smooth(prob,e['window']).astype(np.float64)
                components['q_prime' if e['role']=='core' else 'r_prime'].append(prob)
        mean={k:np.mean(v,axis=0) for k,v in components.items()}
        core=(mean['rq']+mean['q_prime'])/2
        region=(mean['r']+mean['r_prime'])/2
        return {'core':core,'region':region,'any_length':fuse(core,region),**mean}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fasta',required=True)
    ap.add_argument('--features',required=True)
    ap.add_argument('--weights',required=True)
    ap.add_argument('--output',required=True,help='New output TSV; existing files are not overwritten')
    ap.add_argument('--device',default='cpu',help='cpu or cuda[:index]')
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    records=read_fasta(args.fasta)
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    ensemble=Ensemble(args.weights,args.device)
    output.parent.mkdir(parents=True,exist_ok=True)
    thresholds=ensemble.manifest['thresholds']
    with output.open('x',newline='') as f:
        w=csv.writer(f,delimiter='\t')
        w.writerow(['protein_id','position','residue','core','region','any_length',
                    'core_label','region_label','any_length_label'])
        for pid,seq in records:
            scores=ensemble.predict(seq,load_features(seq,args.features,ensemble.device))
            for i,aa in enumerate(seq):
                keys=['core','region','any_length']
                values=[scores[k][i] for k in keys]
                w.writerow([pid,i+1,aa,*[format(v,'.10g') for v in values],
                            *[int(v>=thresholds[k]) for k,v in zip(keys,values)]])
    print(f'Wrote {sum(len(s) for _,s in records)} residue predictions for {len(records)} proteins to {output}')


if __name__=='__main__':main()
