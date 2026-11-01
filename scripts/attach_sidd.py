"""Attach exactly the source images in the frozen SIDD annotation populations."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(x):return hashlib.sha256(x).hexdigest()
def main():
    archive=ROOT/'SIDD-main.zip';config=json.loads((ROOT/'configs/sidd_external.json').read_text())
    assert sha(archive.read_bytes())==config['archive_sha256']
    rows=json.loads((ROOT/'data/sidd_extended_first_records.json').read_text())
    with zipfile.ZipFile(archive) as z:
        for r in rows:
            dest=Path(r['image']).resolve()
            assert dest.is_relative_to(ROOT/'data/sidd')
            content=z.read(r['source_member']);assert sha(content)==r['image_sha256']
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(dest.read_bytes())==r['image_sha256']
            else:dest.write_bytes(content)
    print(f'Attached {len(rows)} fingerprint-verified SIDD images.')
if __name__=='__main__':main()
