"""Download the fixed prediction archive and verify every member before use."""
from pathlib import Path
import hashlib,json,tempfile,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cfg=json.loads((ROOT/'results/download.json').read_text())
    entries=json.loads((ROOT/'results/index.json').read_text())
    if all((ROOT/x['file']).is_file() and sha(ROOT/x['file'])==x['sha256'] for x in entries):
        print('All retained predictions are already present and verified.');return
    with tempfile.TemporaryDirectory(prefix='uav-predictions-') as folder:
        archive=Path(folder)/'predictions.zip'
        urllib.request.urlretrieve(cfg['url'],archive)
        if sha(archive)!=cfg['sha256']:raise ValueError('Archive checksum mismatch')
        with zipfile.ZipFile(archive) as z:
            assert set(z.namelist())=={x['file'] for x in entries}
            for x in entries:
                content=z.read(x['file'])
                if hashlib.sha256(content).hexdigest()!=x['sha256']:raise ValueError(x['file'])
                dest=(ROOT/x['file']).resolve()
                if not dest.is_relative_to(ROOT/'results/predictions'):raise ValueError(x['file'])
                if dest.exists() and sha(dest)!=x['sha256']:raise FileExistsError(dest)
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
    print(f'Verified {len(entries)} prediction files.')
if __name__=='__main__':main()
