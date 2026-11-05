"""Download the versioned GitHub Release model with mandatory SHA256 verification."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upgrade', action='store_true', help='Upgrade a recognized older release and keep its backup.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    spec = json.loads((root / 'model_manifest.json').read_text(encoding='utf-8'))
    target = root / 'weights' / spec['filename']
    target.parent.mkdir(exist_ok=True)
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):h.update(chunk)
        return h.hexdigest()
    previous = None
    if target.exists():
        current_hash = digest(target)
        if current_hash == spec['sha256']:
            print('Model already verified.')
            return
        previous = next((item for item in spec.get('previous_versions', [])
                         if item['sha256'] == current_hash), None)
        if previous is None:
            raise RuntimeError('Existing model has an unexpected hash; move it aside before retrying.')
        if not args.upgrade:
            raise RuntimeError('Older model found. Run: python download_weights.py --upgrade')
    temporary = target.with_suffix('.download')
    if temporary.exists():
        raise RuntimeError('An incomplete .download file exists; move it aside before retrying.')
    req = urllib.request.Request(spec['url'], headers={'User-Agent': 'facecheck-local'})
    print('Downloading model (about 100 MiB)...')
    with urllib.request.urlopen(req, timeout=120) as response, temporary.open('xb') as f:
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > spec['bytes']:
                raise RuntimeError('Download exceeded expected size.')
            f.write(chunk)
    if temporary.stat().st_size != spec['bytes'] or digest(temporary) != spec['sha256']:
        raise RuntimeError('Model verification failed. The .download file was preserved for inspection.')
    if previous is not None:
        backup = target.with_name(target.stem + '.' + previous['version'] + target.suffix)
        if backup.exists():
            if digest(backup) != previous['sha256']:
                raise RuntimeError('Older-model backup has an unexpected hash. Download preserved.')
            temporary.replace(target)
        else:
            target.rename(backup)
            try:
                temporary.replace(target)
            except OSError:
                backup.rename(target)
                raise
    else:
        temporary.replace(target)
    print('Model downloaded and SHA256 verified.')


if __name__ == '__main__':main()
