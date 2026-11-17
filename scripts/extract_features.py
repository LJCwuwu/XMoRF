"""Extract the original ProtT5/ProstT5 features using 1000/750 overlap-add chunks."""
import argparse,gc,json,os,sys
from pathlib import Path
import numpy as np
import torch
from xmorf.predict import read_fasta,feature_key
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from morf.features.plm import make_extractor

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--fasta',required=True)
 ap.add_argument('--output',default='features')
 ap.add_argument('--device',default='cuda')
 ap.add_argument('--prott5',default='Rostlab/prot_t5_xl_half_uniref50-enc')
 ap.add_argument('--prostt5',default='Rostlab/ProstT5_fp16')
 a=ap.parse_args();records=read_fasta(a.fasta);out=Path(a.output)
 # Different identifiers may contain the same sequence/cache key.
 records=list({feature_key(seq):(pid,seq) for pid,seq in records}.values())
 for name,source in [('prott5',a.prott5),('prostt5',a.prostt5)]:
  dest=out/name;dest.mkdir(parents=True,exist_ok=True)
  metadata={'name':name,'source':source,'window':1000,'stride':750,'storage_dtype':'float16',
            'extraction_dtype':'float16' if a.device.startswith('cuda') else 'float32'}
  meta=dest/'extraction.json';existing=list(dest.glob('*.npy'))
  if existing and (not meta.exists() or json.loads(meta.read_text())!=metadata):
   raise ValueError(f'Use a new feature directory; existing cache provenance differs: {dest}')
  meta.write_text(json.dumps(metadata,indent=2)+'\n')
  missing=[(pid,seq) for pid,seq in records if not (dest/(feature_key(seq)+'.npy')).exists()]
  if not missing:continue
  os.environ['MORF_PROTT5_HF' if name=='prott5' else 'MORF_PROSTT5_HF']=source
  extractor=make_extractor(name,device=a.device)
  for pid,seq in missing:
   array=extractor(seq).astype(np.float16)
   if array.shape!=(len(seq),1024) or not np.isfinite(array).all():raise ValueError(f'Invalid embedding: {pid}')
   path=dest/(feature_key(seq)+'.npy')
   with path.open('xb') as f:np.save(f,array,allow_pickle=False)
   print(name,pid,len(seq),flush=True)
  del extractor;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()

if __name__=='__main__':main()
