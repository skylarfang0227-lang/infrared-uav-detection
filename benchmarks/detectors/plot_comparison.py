"""Render measured primary AP only; refuses incomplete or mismatched evidence."""
from pathlib import Path
import json,hashlib,math,argparse,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;E=B.parents[1];R=E
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replace',action='store_true');args=parser.parse_args()
    comparison=read(B/'results/comparison.json');protocol=read(B/'protocol.json')
    assert comparison['seed']==17 and comparison['epochs']==50 and comparison['protocol_sha256']==sha(B/'protocol.json')
    assert [r['model'] for r in comparison['models']]==[r['id'] for r in protocol['models']]+['reference','head']
    evidence=[]
    for r in comparison['models']:
        p=R/r['metrics'];m=read(p)
        assert m['seed']==17 and m['epoch']==50 and m['primary']==r['primary']==m['metrics']['UAV/all']['ap50_95']
        assert m['metrics']['UAV/all']['targets']==384 and m['metrics']['UAV/tiny']['targets']==49
        assert 0<=r['primary']<=1
        evidence.append({'label':r['label'],'primary_raw':r['primary'],'percent':100*r['primary'],'metrics':r['metrics'],'sha256':sha(p)})
    labels=[r['label'] for r in comparison['models']];labels[-2:]=['A: MSDA-IRSTD','H: proposed head'];labels[2]='YOLO11n*'
    values=[r['percent'] for r in evidence]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    fig,ax=plt.subplots(figsize=(7.3,3.8));bars=ax.barh(range(6),values,color=['#8B98A5']*4+['#4A6785','#1676AA'],height=.58,zorder=3)
    ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlabel('UAV AP50:95 (%)')
    ax.set_xlim(0,math.ceil((max(values)+3)/5)*5);ax.grid(axis='x',alpha=.18,zorder=0);ax.tick_params(axis='y',length=0,pad=8)
    ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=5,fontsize=10)
    fig.subplots_adjust(left=.27,right=.94,bottom=.25,top=.98)
    fig.text(.03,.025,'Seed 17 · 50 epochs · terminal EMA\n* Collapsed training run; no detections at the fixed confidence threshold.',fontsize=8.5,color='#444444')
    out=E/'figures';name='single_seed_detector_comparison'
    for ext in ['png','svg']:
        path=out/f'{name}.{ext}'
        if path.exists():
            assert args.replace,'Use --replace explicitly to archive and revise this new figure.'
            archive=B/'audit/figure_versions';archive.mkdir(exist_ok=True)
            backup=archive/f'{name}_{sha(path)}.{ext}'
            if not backup.exists():shutil.copy2(path,backup)
        fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    (B/'audit/figure_provenance.json').write_text(json.dumps({'generator':'Python Matplotlib','script_sha256':sha(Path(__file__)),'comparison_sha256':sha(B/'results/comparison.json'),'six_verified_inputs':evidence,'png_sha256':sha(out/f'{name}.png'),'no_generated_imagery':True},indent=2),encoding='utf-8')
    print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
