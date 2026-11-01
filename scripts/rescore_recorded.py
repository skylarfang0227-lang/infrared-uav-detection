"""Recompute real published detection scores from retained predictions on CPU."""
from pathlib import Path
import argparse, contextlib, gzip, io, json, sys, os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'logs/ultralytics'))
from scripts.evaluate_sirst_protocol import score
E=ROOT
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');a=p.parse_args()
    records=read(E/'data/staged_manifest.json');jobs=[]
    for metrics in sorted((E/'results/test').glob('*/metrics.json')):jobs.append((metrics,records['test']))
    for metrics in sorted((E/'benchmarks/detectors/results').glob('*/metrics.json')):jobs.append((metrics,records['test']))
    if a.all:
        for group in ['dev','long_dev']:
            for metrics in sorted((E/'results'/group).glob('*/metrics.json')):jobs.append((metrics,records['dev']))
        sidd=read(E/'data/sidd_clean_records.json')
        for metrics in sorted((E/'results/sidd').glob('*/metrics.json')):jobs.append((metrics,sidd))
    checks=[]
    for path,rows in jobs:
        original=read(path);ids={int(r['id']) for r in rows}
        with gzip.open(path.parent/'predictions.json.gz','rt',encoding='utf-8') as f:pred=[x for x in json.load(f) if x['image_id'] in ids]
        with contextlib.redirect_stdout(io.StringIO()):actual=score(rows,pred)
        assert actual['metrics']==original['metrics'] and actual['primary']==original['primary'],str(path)
        checks.append({'metrics':path.relative_to(ROOT).as_posix(),'primary':actual['primary'],'exact_match':True})
        print(path.parent.name,'exact match',actual['primary'],flush=True)
    out=ROOT/'verification';out.mkdir(exist_ok=True)
    (out/'rescore.json').write_text(json.dumps({'runs':checks,'all_exact':True},indent=2))
    print(f'Exact agreement for {len(checks)} retained prediction sets.')
if __name__=='__main__':main()
