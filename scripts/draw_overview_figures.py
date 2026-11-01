"""Draw Figures 1-2 from the frozen specification and existing evidence only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

N = Path(__file__).resolve().parents[1]
E=N


def draw(save):
    # An explicit multiline layout leaves padding around every module label.
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.subplots_adjust(left=.015, right=.985, bottom=.02, top=.98)
    ax.set(xlim=(0, 12), ylim=(0, 7.3)); ax.axis('off')
    boxes = []
    def box(x, y, w, h, label, accent=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.07',
                          facecolor='#e9f3f9' if accent else '#f2f3f5',
                          edgecolor='#277ba0' if accent else '#737f8d', linewidth=1)
        ax.add_patch(p)
        t = ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=9,
                    linespacing=1.3)
        boxes.append((p, t))
    def arrow(a, b):
        ax.annotate('', xy=b, xytext=a,
                    arrowprops={'arrowstyle':'->', 'color':'#535f6e', 'lw':1.1})
    ax.text(.1, 7.05, '(a) Shared detector: A / H / S / HS', fontsize=10, weight='bold')
    box(.1, 5.5, 1.5, 1.2, 'Input\n3 × 640²')
    box(1.95, 5.5, 2.15, 1.2, 'Input stem\nOrdinary or\nfour-padding (S)')
    box(4.45, 5.5, 3.05, 1.2, 'Complete MSDA\nASFE · C2DYT\nMFA · DSA')
    box(7.85, 5.5, 4.05, 1.2, 'P2 / P3 / P4 / P5 features\nStrides 4 / 8 / 16 / 32')
    for a, b in [((1.6,6.1),(1.95,6.1)),((4.1,6.1),(4.45,6.1)),((7.5,6.1),(7.85,6.1))]:
        arrow(a,b)
    ax.text(.1, 4.85, '(b) H: final P2 prediction adaptation', fontsize=10, weight='bold')
    box(.1, 2.2, 1.65, 1.2, 'P2 feature\n32 × 160²')
    for y, role, width, proj, out in [(3.15,'Regression',64,256,64), (1.2,'Classification',32,8,2)]:
        box(2.2, y, 2.3, 1.2, f'{role} tower\n{width} × 160²')
        box(5.05, y, 2.65, 1.2, f'Final 1 × 1 projection\nExpanded channels\n{proj} × 160²', True)
        box(8.25, y, 3.6, 1.2, f'PixelShuffle(2)\n{out} × 320²\nOutput stride 2', True)
        arrow((1.75,2.8),(2.2,y+.6))
        arrow((4.5,y+.6),(5.05,y+.6)); arrow((7.7,y+.6),(8.25,y+.6))
    ax.text(.1,.65,'Blue: H intervention. Both feature towers retain stride 4.',fontsize=9)
    ax.text(.1,.2,'S changes only the input stem; P3–P5 heads are unchanged. Batch dimension omitted.',fontsize=9)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); checks=[]
    for patch, text in boxes:
        inner=patch.get_window_extent(renderer).padded(-4)
        tb=text.get_window_extent(renderer)
        inside=(tb.x0>=inner.x0 and tb.y0>=inner.y0 and tb.x1<=inner.x1 and tb.y1<=inner.y1)
        checks.append({'label':text.get_text(), 'inside_box_with_padding':bool(inside)})
    assert all(x['inside_box_with_padding'] for x in checks), checks
    save(fig,'architecture')

    integrity=E/'analysis/data_integrity.json'; manifest=E/'data/staged_manifest.json'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    audit=load(integrity); members=load(manifest)
    fig, ax=plt.subplots(figsize=(8.2,3.25),layout='constrained')
    values=[audit[k] for k in ['images','unique_decoded_images','eligible_pixel_unique_images']]
    bars=ax.bar(np.arange(3),values,width=.65,color=['#84919b','#417d9f','#417d9f'])
    ax.bar_label(bars,padding=3,fontsize=10)
    bottom=0
    for split,label,color in [('train','Train','#276799'),('dev','Dev','#e9a65a'),('test','Final','#5b9c78')]:
        count=len(members[split]); groups=len({r['background_group'] for r in members[split]})
        ax.bar(3,count,bottom=bottom,width=.65,color=color,label=f'{label} ({groups} proxy groups)')
        ax.text(3,bottom+count/2,str(count),ha='center',va='center',fontsize=10,weight='bold',color='white')
        bottom+=count
    assert bottom==values[-1]
    ax.set(xticks=range(4),xticklabels=['Raw images','Pixel-unique','Eligible unique','Grouped split'],
           ylabel='Number of images',ylim=(0,3500))
    ax.legend(loc='upper right',frameon=False,fontsize=8)
    save(fig,'data_audit')

    sources={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [integrity,manifest]}
    (N/'audit/overview_figure_validation.json').write_text(json.dumps({
        'generator':'Python / Matplotlib', 'architecture_box_checks':checks,
        'sources_sha256':sources, 'policy':'Redraw existing data; no training, inference or experimental changes.'
    },indent=2),encoding='utf-8')
