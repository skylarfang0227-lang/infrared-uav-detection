"""Plot retained post-NMS match counts; no model inference or new evaluation."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
N=Path(__file__).resolve().parents[1]
source=N/'analysis/prediction_diagnostics.json'
data=json.loads(source.read_text(encoding='utf-8'))
rows=data['aggregate']
# Confirm the pooled counts against each of the three paired runs.
for row in rows:
    pairs=[p for p in data['paired'] if p['group']==row['group'] and p['iou']==row['iou']]
    assert {p['seed'] for p in pairs}=={17,29,43}
    for field in ['A_only','H_only','both','neither']:
        assert row[field]==sum(p[field] for p in pairs)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(7.05,2.55))
for ax,group,title in zip(axes,['all','tiny'],['(a) All UAVs','(b) Tiny UAVs']):
    selected=sorted([r for r in rows if r['group']==group],key=lambda r:r['iou'])
    x=np.arange(2)
    for offset,key,label,color in [(-.18,'H_only','Gained (H only)','#28659A'),(.18,'A_only','Lost (A only)','#CC7838')]:
        bars=ax.bar(x+offset,[r[key] for r in selected],width=.34,label=label,color=color,zorder=3)
        ax.bar_label(bars,padding=3,fontsize=9)
    ax.set_xticks(x,['0.50','0.75'])
    ax.set_xlabel('Matching IoU threshold')
    ax.set_ylabel('Target–seed occurrences')
    ax.set_title(title,loc='left',pad=8)
    ax.set_ylim(0,185 if group=='all' else 25)
    ax.set_yticks([0,50,100,150] if group=='all' else [0,5,10,15,20,25])
    ax.grid(axis='y',color='#dddddd',linewidth=.6,zorder=0)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,1.025),ncol=2,frameon=False)
fig.subplots_adjust(left=.085,right=.985,bottom=.22,top=.81,wspace=.34)
for suffix in ['png','pdf']:
    fig.savefig(N/'figures'/f'match_turnover.{suffix}',dpi=240,bbox_inches='tight',facecolor='white')
manifest={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'counts':rows,'aggregation_checked':True,'unit':'target–seed occurrences, not independent targets or rates','panels_have_different_scales':True}
(N/'audit/match_turnover_source.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
