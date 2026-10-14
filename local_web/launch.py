"""Double-click launcher: start a hidden local service or reuse this app."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:8766'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def status():
    try:
        with OPENER.open(URL + '/api/status', timeout=1) as response:
            data = json.load(response)
        return data.get('service') == 'facecheck-public-local'
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    if args.stop:
        if not status():
            print('Local service is not running.');return
        with OPENER.open(URL, timeout=2) as response:
            html = response.read().decode('utf-8')
        token = re.search(r'name="local-token" content="([^"]+)"', html).group(1)
        request = urllib.request.Request(URL + '/api/shutdown', data=b'{}',
                                        headers={'Content-Type': 'application/json', 'X-Local-Token': token})
        with OPENER.open(request, timeout=3) as response:
            assert json.load(response)['stopped']
        print('Local service stopped.');return
    if not (ROOT / 'weights/facecheck-ensemble.pt').is_file():
        raise RuntimeError('Model missing. Run .venv\\Scripts\\python.exe download_weights.py first.')
    if not status():
        folder = ROOT / 'outputs/local_web';folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'server.log').open('ab') as log:
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            proc = subprocess.Popen([sys.executable, str(ROOT / 'local_web/server.py')], cwd=ROOT,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=flags)
        for _ in range(100):
            if status():break
            if proc.poll() is not None:
                raise RuntimeError('Unable to start local service. See outputs/local_web/server.log (port 8766 may be occupied).')
            time.sleep(.2)
        else:raise RuntimeError('Local service did not become ready. See outputs/local_web/server.log.')
    if not args.no_browser:
        webbrowser.open(URL)
    print('Ready: ' + URL)


if __name__ == '__main__':main()
