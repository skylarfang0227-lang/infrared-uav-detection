"""Redraw existing measurements and selected examples; no new model evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
N=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
def save(fig,name):
    for ext in ['png','svg']:fig.savefig(N/'figures'/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)

from draw_overview_figures import draw
draw(save)
d=json.loads((N/'analysis/threshold_analysis.json').read_text(encoding='utf-8'))
colors=['#454e5c','#0072b2','#b47718','#aa4675']
variants=['reference','head','stem','combined']
fig,axs=plt.subplots(1,2,figsize=(8.2,2.9),sharey=True)
for ax,group,title in zip(axs,['all','tiny'],['All UAVs (384 targets)','Tiny UAVs (49 targets)']):
    for i,v in enumerate(variants):
        y=np.mean([r['curves'][group]['AP'] for r in d['runs'] if r['variant']==v],axis=0)*100
        ax.plot(d['iou_thresholds'],y,color=colors[i],marker=['o','s','^','D'][i],ms=3.3,lw=1.5,ls='-' if i<2 else '--',label=['A: reference','H: head','S: stem','HS: both'][i])
    ax.axvline(.75,color='#9aa2a9',ls=':',lw=1,zorder=0)
    ax.set(title=title,xlabel='IoU threshold',xlim=(.49,.96),ylim=(0,85),xticks=[.5,.6,.7,.75,.8,.9,.95],yticks=[0,20,40,60,80])
    ax.tick_params(axis='x',labelsize=8.5);ax.grid(axis='y',alpha=.16)
axs[0].set_ylabel('AP (%)')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,1.08),ncol=4,frameon=False,fontsize=9)
fig.tight_layout(w_pad=2);save(fig,'threshold_profiles')
fig,axs=plt.subplots(1,2,figsize=(8.2,2.8),sharey=True)
for ax,group,title in zip(axs,['all','tiny'],['All UAVs: H − A','Tiny UAVs: H − A']):
    for j,metric in enumerate(['AP50','AP75']):
        vals=np.array(d['contrasts'][f'{group}/{metric}/AP/H-A']['values'])*100
        xx=np.arange(3)+(-.13 if j==0 else .13)
        ax.scatter(xx,vals,s=43,marker=['o','s'][j],color=[colors[2],colors[1]][j],label=metric,zorder=3)
        for x,y in zip(xx,vals):ax.annotate(f'{y:+.2f}',(x,y),xytext=(-9,-13) if j==0 else (9,7),textcoords='offset points',ha='center',fontsize=8.4,bbox={'facecolor':'white','edgecolor':'none','pad':.2},zorder=4)
    ax.axhline(0,color='#7d858d',lw=.9)
    ax.set(title=title,xlabel='Paired training seed',xticks=[0,1,2],xticklabels=['17','29','43'],xlim=(-.55,2.55),ylim=(-20,15),yticks=[-20,-10,0,10])
    ax.grid(axis='y',alpha=.16)
axs[0].set_ylabel('AP difference (percentage points)')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,1.08),ncol=2,frameon=False,fontsize=9)
fig.tight_layout(w_pad=2);save(fig,'paired_thresholds')
import sys
if '--skip-examples' in sys.argv:sys.exit(0)
# Reuse exactly the same selected examples and boxes; only arrangement changes.
sel=json.loads((N/'analysis/qualitative_selection.json').read_text(encoding='utf-8'))
fig,axs=plt.subplots(2,3,figsize=(8.2,5.0))
cases=[('cross75','H reaches IoU 0.75'),('loss50','H below IoU 0.50'),('shared_miss','Shared miss')]
for col,(key,title) in enumerate(cases):
    q=sel['selected'][key];im=np.array(Image.open(q['image']).convert('L'));b=q['box'];size=80
    x0=max(0,min(im.shape[1]-size,int(b[0]+b[2]/2-size/2)));y0=max(0,min(im.shape[0]-size,int(b[1]+b[3]/2-size/2)))
    for row,v in enumerate(['A','H']):
        ax=axs[row,col];ax.imshow(im[y0:y0+size,x0:x0+size],cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=(x0-.5,x0+size-.5,y0+size-.5,y0-.5))
        ax.add_patch(Rectangle(b[:2],b[2],b[3],fill=False,ec='#4bdf70',lw=1.2))
        if q[v]['prediction']:
            bb=q[v]['prediction']['bbox'];ax.add_patch(Rectangle(bb[:2],bb[2],bb[3],fill=False,ec='#e6a035' if v=='A' else '#36b8ed',lw=1.2))
        ax.set(xticks=[],yticks=[],xlim=(x0-.5,x0+size-.5),ylim=(y0+size-.5,y0-.5))
        ax.set_title((title+'\n' if row==0 else '')+f"{v}: IoU {q[v]['iou']:.2f} · ID {q['image_id']}",fontsize=11.5)
fig.tight_layout(w_pad=.9,h_pad=1.0);save(fig,'tiny_examples')
sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [N/'analysis/threshold_analysis.json',N/'analysis/qualitative_selection.json']}
(N/'audit/figure_sources.json').write_text(json.dumps({'sources_sha256':sources,'changes':'Layout and annotations only; same complete curves, all paired seeds, selected examples and source boxes.'},indent=2),encoding='utf-8')
