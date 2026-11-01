from pathlib import Path
import collections,json,zipfile,hashlib
E=Path(__file__).resolve().parents[1];R=E
assert not (E/'results/sidd').exists(),'Register before SIDD inference'
rows=json.loads((E/'data/sidd_all_records.json').read_text());groups=collections.defaultdict(list)
for r in rows:groups[r['decoded_sha256']].append(r)
first=[];last=[]
with zipfile.ZipFile(R/'SIDD-main.zip') as z:
    for g in groups.values():
        a=dict(g[0]);p=Path(a['image'])
        if not p.exists():p.write_bytes(z.read(a['source_member']))
        assert hashlib.sha256(p.read_bytes()).hexdigest()==a['image_sha256']
        first.append(a);last.append(dict(a,boxes=g[-1]['boxes'],native_boxes=g[-1]['native_boxes'],alternate_annotation_source_id=g[-1]['id']))
for name,rs in [('sidd_extended_first_records.json',first),('sidd_extended_last_records.json',last)]:
    p=E/'data'/name
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(rs,indent=2),encoding='utf-8')
print({'unique_images':len(first),'original_primary_unchanged':len(json.loads((E/'data/sidd_clean_records.json').read_text()))})
