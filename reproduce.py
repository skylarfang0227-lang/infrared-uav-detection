"""Submission entry point. Create a separate workspace before running experiments."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys

ROOT=Path(__file__).resolve().parent
STUDY=ROOT
PAPER=ROOT

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def call(script,*args):
    subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/script),*map(str,args)],cwd=ROOT,check=True)

def prepare(a):
    if (ROOT/'.workspace.json').exists():raise RuntimeError('Prepare from the original checkout.')
    target=a.output.resolve()
    if target==ROOT or target.is_relative_to(ROOT) or target.exists():raise ValueError('Choose a new directory outside the checkout.')
    def ignore(folder,names):
        skipped={'.git','__pycache__','.venv','.workspace.json'}
        if a.mode=='training' and Path(folder)==ROOT/'results':return names
        return [x for x in names if x in skipped]
    shutil.copytree(ROOT,target,ignore=ignore)
    import gzip
    def expand(bundle):
        with gzip.open(ROOT/bundle,'rt',encoding='utf-8') as f:contents=json.load(f)
        for relative,value in contents.items():
            dest=(target/relative).resolve()
            if not dest.is_relative_to(target):raise ValueError(relative)
            dest.parent.mkdir(parents=True,exist_ok=True)
            text=json.dumps(value,ensure_ascii=False,indent=2) if relative.endswith('.json') else value
            dest.write_text(text.replace('@PROJECT_ROOT@',target.as_posix()),encoding='utf-8')
    expand('data/records.json.gz')
    if a.mode=='recorded':
        expand('results/measurements.json.gz')
        for item in read(ROOT/'results/index.json'):
            source=target/item['file'];dest=target/item['destination']
            assert sha(source)==item['sha256']
            dest.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(source,dest)
            except OSError:shutil.copyfile(source,dest)
    for rel in ['experiments','results','analysis','logs','audit','figures','benchmarks/detectors/runs','benchmarks/detectors/results','benchmarks/detectors/logs','benchmarks/detectors/audit']:
        (target/rel).mkdir(parents=True,exist_ok=True)
    # Configuration paths are localized; scientific settings are unchanged.
    for p in list((target/'configs').rglob('*.json'))+[target/'benchmarks/detectors/protocol.json']:
        p.write_text(p.read_text().replace('@PROJECT_ROOT@',target.as_posix()),encoding='utf-8')
    write(target/'.workspace.json',{'mode':a.mode})
    subprocess.run([sys.executable,'-B',str(target/'reproduce.py'),'register'],cwd=target,check=True)
    if a.sirst_root:subprocess.run([sys.executable,'-B',str(target/'reproduce.py'),'attach-data','--sirst-root',str(a.sirst_root.resolve())],cwd=target,check=True)
    print('Prepared '+str(target))

def register(a):
    if not (ROOT/'.workspace.json').exists():raise RuntimeError('Run prepare first.')
    # Only operational identity fields are rebound. Published scores never change.
    manifest=STUDY/'data/staged_manifest.json';members=STUDY/'data/train.txt'
    for name in ['factorial_weights_frozen','long_schedule_weights_frozen']:
        p=STUDY/f'results/{name}.json'
        if p.exists():
            d=read(p);d['protocol_sha256']=sha(STUDY/'configs'/('grouped_protocol.json' if name.startswith('factorial') else 'long_schedule.json'))
            if 'manifest_sha256' in d:d['manifest_sha256']=sha(manifest)
            write(p,d)
    b=STUDY/'benchmarks/detectors';p=b/'protocol.json';d=read(p)
    d['manifest_sha256']=sha(manifest);d['train_members_sha256']=sha(members)
    for item in d['reuse']:
        q=ROOT/item['metrics']
        if q.exists():item['metrics_sha256']=sha(q);item['predictions_sha256']=sha(q.parent/'predictions.json.gz')
    write(p,d)
    p=b/'results/comparison.json'
    if p.exists():
        d=read(p);d['protocol_sha256']=sha(b/'protocol.json');write(p,d)
    p=ROOT/'configs/dataset_evidence.json';d=read(p)
    for row in d['splits'].values():
        q=ROOT/row['path']
        if q.exists():row['sha256']=sha(q)
    write(p,d)
    write(ROOT/'logs/workspace_registration.json',{'mode':read(ROOT/'.workspace.json')['mode'],'protocols':{p.relative_to(ROOT).as_posix():sha(p) for p in [STUDY/'configs/grouped_protocol.json',STUDY/'configs/long_schedule.json',b/'protocol.json']},'manifest_sha256':sha(manifest),'train_members_sha256':sha(members),'note':'These are relocated execution fingerprints, not the original experiment fingerprints.'})

def attach_data(a):
    if not (ROOT/'.workspace.json').exists():raise RuntimeError('Run prepare first.')
    source=a.sirst_root.resolve();rows=read(ROOT/'data/splits/sirst_uavb/manifest.json')['records'];count=0
    for r in rows:
        rel=Path(r['image']).relative_to('data/processed/sirst_uavb')
        for kind,sub,key in [('image',rel,'image_sha256'),('label',Path(str(rel).replace('images/','labels/').replace('images\\','labels\\')).with_suffix('.txt'),'label_sha256')]:
            src=source/sub;dst=ROOT/'data/processed/sirst_uavb'/sub
            if not src.is_file() or sha(src)!=r[key]:raise ValueError('Dataset fingerprint mismatch: '+str(src))
            if a.verify_only:continue
            if dst.exists():
                if sha(dst)!=r[key]:raise ValueError('Existing destination differs: '+str(dst))
            else:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        count+=1
    for split,rs in read(STUDY/'data/staged_manifest.json').items():
        for r in rs:
            image=(source/Path(r['source_image']).relative_to('data/processed/sirst_uavb')) if a.verify_only else ROOT/r['source_image']
            label=Path(str(image).replace('images/','labels/').replace('images\\','labels\\')).with_suffix('.txt')
            for src,dst,key in [(image,Path(r['image']),'image_sha256'),(label,Path(r['label']),'label_sha256')]:
                if sha(src)!=r[key]:raise ValueError(src)
                if not dst.is_relative_to(ROOT):raise ValueError('Destination escapes workspace')
                if a.verify_only:continue
                dst.parent.mkdir(parents=True,exist_ok=True)
                if not dst.exists():shutil.copyfile(src,dst)
                elif sha(dst)!=r[key]:raise ValueError(dst)
    print(f'{"Verified" if a.verify_only else "Attached and verified"} {count} original images and the frozen grouped split.')

def import_weights(a):
    # Explicit mapping: logical destination from configs/checkpoints.json -> supplied file.
    if not (ROOT/'.workspace.json').exists():raise RuntimeError('Run prepare first.')
    expected={x['path']:x for x in read(ROOT/'configs/checkpoints.json')['files']}
    mapping=read(a.mapping)
    for name,path in mapping.items():
        item=expected[name];src=Path(path).resolve();dst=(ROOT/name).resolve()
        if not dst.is_relative_to(ROOT) or sha(src)!=item['sha256']:raise ValueError('Unrecognized checkpoint: '+name)
        if dst.exists() and sha(dst)!=item['sha256']:raise FileExistsError(dst)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copyfile(src,dst)
    print(f'Imported {len(mapping)} verified checkpoints.')

def train(a):
    if not (ROOT/'.workspace.json').exists() or read(ROOT/'.workspace.json')['mode']!='training':raise RuntimeError('Use a fresh training workspace.')
    if not (STUDY/'data/dataset/images/train').exists():raise RuntimeError('Attach the fingerprint-matched SIRST data first.')
    s='scripts/'
    if a.experiment=='factorial':
        cfg=STUDY/'configs/grouped_protocol.json';members=STUDY/'data/smoke_train.txt'
        for name,variant,stop,resume in [('smoke_v2_full','combined',2,None),('smoke_v2_partial','combined',1,None),('smoke_v2_resume','combined',2,'smoke_v2_partial'),('smoke_reference','reference',2,None),('smoke_collapsed','phase_collapsed',2,None)]:
            args=['--variant',variant,'--seed','17','--protocol',cfg,'--run',name,'--members',members,'--stop-after',str(stop)]
            if resume:args+=['--resume',STUDY/f'experiments/{resume}/train/weights/last.pt']
            call(s+'train.py',*args)
        # Verify into a new path, then use this run's check in the campaign.
        output=STUDY/'analysis/resume_submission_check.json'
        call(s+'verify_resume.py','--output',output)
        write(STUDY/'analysis/resume_check.json',read(output));call(s+'run_factorial.py')
    elif a.experiment=='detectors':
        register(a)
        call('benchmarks/detectors/campaign.py')
    elif a.experiment=='long':
        jobs=[]
        for variant in ['reference','combined']:
            name=f'long_{variant}_s17'
            call(s+'train.py','--variant',variant,'--seed','17','--protocol',STUDY/'configs/long_schedule.json','--run',name)
            path=STUDY/f'experiments/{name}/train/weights/epoch100.pt'
            jobs.append({'variant':variant,'seed':17,'checkpoint':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(path)})
        write(STUDY/'results/long_schedule_weights_frozen.json',{'protocol_sha256':sha(STUDY/'configs/long_schedule.json'),'runs':jobs})
    elif a.experiment=='original':
        jobs=[]
        for seed in [17,29,43]:
            for method in ['reference','candidate']:
                call('scripts/train_original_split.py','--method',method,'--seed',str(seed))
                p=ROOT/f'experiments/original_split/{method}_s{seed}/train/weights/epoch50.pt'
                jobs.append({'method':method,'seed':seed,'checkpoint':p.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(p)})
        write(ROOT/'experiments/original_split/frozen/weights_frozen.json',{'runs':jobs})

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare',help='Create an isolated recorded-result or training workspace');q.add_argument('--output',type=Path,required=True);q.add_argument('--mode',choices=['recorded','training'],default='recorded');q.add_argument('--sirst-root',type=Path);q.set_defaults(func=prepare)
    q=sub.add_parser('register');q.set_defaults(func=register)
    q=sub.add_parser('attach-data');q.add_argument('--sirst-root',type=Path,required=True);q.add_argument('--verify-only',action='store_true');q.set_defaults(func=attach_data)
    q=sub.add_parser('import-weights');q.add_argument('--mapping',type=Path,required=True);q.set_defaults(func=import_weights)
    q=sub.add_parser('train');q.add_argument('--experiment',choices=['factorial','detectors','long','original'],required=True);q.set_defaults(func=train)
    q=sub.add_parser('verify');q.set_defaults(func=lambda a:call('tests/verify_submission.py'))
    q=sub.add_parser('rescore');q.add_argument('--all',action='store_true');q.set_defaults(func=lambda a:call('scripts/rescore_recorded.py',*(['--all'] if a.all else [])))
    q=sub.add_parser('figures');q.add_argument('--with-examples',action='store_true')
    def figures(a):
        if not (ROOT/'.workspace.json').exists():raise RuntimeError('Run prepare first.')
        call('scripts/plot_figures.py',*([] if a.with_examples else ['--skip-examples']))
        call('scripts/plot_match_turnover.py')
        call('benchmarks/detectors/plot_comparison.py','--replace')
    q.set_defaults(func=figures)
    a=p.parse_args();os.environ['PYTHONDONTWRITEBYTECODE']='1';a.func(a)
if __name__=='__main__':main()
