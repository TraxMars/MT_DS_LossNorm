"""Manifest current C1 sources/results, including Git-ignored raw arrays."""
from datetime import datetime,timezone
import json
import platform
import sys
import torch,numpy,scipy,pandas,matplotlib,h5py
from bridge import ROOT,PROJECT,b,write_json

def main():
    assert json.loads((ROOT/'FINAL_STATUS.json').read_text())['status']=='COMPLETE'
    write_json(ROOT/'ENVIRONMENT.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),executable=sys.executable,
        torch=torch.__version__,CUDA=torch.version.cuda,numpy=numpy.__version__,scipy=scipy.__version__,pandas=pandas.__version__,
        matplotlib=matplotlib.__version__,h5py=h5py.__version__,gpu=json.loads((ROOT/'runs/probe/source_at_run.json').read_text())['gpu'],
        gpu_device='cuda:0',network_precision='FP32',diagnostic_loss_precision='FP64',TF32=False,cudnn_deterministic=True,cudnn_benchmark=False,
        torch_deterministic_algorithms=False,seed=142,model_mode='eval',new_packages_installed=False))
    paths=sorted([p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json']+[PROJECT/'tests/test_stage1c_c1.py'])
    entries=[dict(path=str(p.relative_to(PROJECT)),bytes=p.stat().st_size,sha256=b.sha256(p)) for p in paths]
    write_json(ROOT/'ARTIFACT_MANIFEST.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files=entries,file_count=len(entries),
        total_bytes=sum(r['bytes'] for r in entries),excludes=['manifest itself','__pycache__'],raw_arrays_remain_local_and_Git_ignored=True))
    print('C1 manifest:',len(entries),'files;',sum(r['bytes'] for r in entries),'bytes')

if __name__=='__main__':main()
