"""Loopback-only image inference UI; uses the existing trained checkpoint."""
import argparse
import base64
import binascii
import io
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
TOKEN = secrets.token_urlsafe(32)
MAX_BODY = 36 * 1024 * 1024
MAX_FILE = 8 * 1024 * 1024
LOCK = threading.Lock()


class Engine:
    def __init__(self):
        self.model = None
        self.device = None

    def predict(self, files):
        from PIL import Image, ImageOps, UnidentifiedImageError
        import torch
        from core import load_checkpoint, transform
        tensors = []
        names = set()
        if not isinstance(files, list) or not 1 <= len(files) <= 20:
            raise ValueError('每次请选择1至20张图片。')
        # Decode the entire request before running inference; never save uploads.
        tf = transform(224)
        for item in files:
            if not isinstance(item, dict):
                raise ValueError('图片请求格式错误。')
            name, encoded = item.get('name'), item.get('data')
            if not isinstance(name, str) or not name or len(name) > 255 or any(c in name for c in '/\\\r\n\x00'):
                raise ValueError('图片文件名无效。')
            if Path(name).suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                raise ValueError('仅支持 JPG、JPEG 和 PNG 图片。')
            if name in names:
                raise ValueError('图片不能重名，请重命名后重试。')
            names.add(name)
            if not isinstance(encoded, str) or len(encoded) > MAX_FILE * 4 // 3 + 4:
                raise ValueError('单张图片不能超过8MB。')
            try:
                data = base64.b64decode(encoded, validate=True)
                if len(data) > MAX_FILE:
                    raise ValueError('单张图片不能超过8MB。')
                with Image.open(io.BytesIO(data)) as im:
                    if im.format not in {'JPEG', 'PNG'}:
                        raise ValueError('图片实际格式必须为JPEG或PNG。')
                    if im.width * im.height > 20_000_000:
                        raise ValueError('单张图片不能超过2000万像素。')
                    tensors.append(tf(ImageOps.exif_transpose(im).convert('RGB')))
            except (binascii.Error, UnidentifiedImageError, OSError) as e:
                raise ValueError(f'无法读取图片：{name}') from e
        if self.model is None:
            torch.set_num_threads(8)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model, _ = load_checkpoint(ROOT / 'weights/facecheck-ensemble.pt', self.device)
        start = time.perf_counter()
        probabilities = []
        with torch.inference_mode():
            for offset in range(0, len(tensors), 4):
                p = self.model(torch.stack(tensors[offset:offset + 4]).to(self.device)).flatten().sigmoid().cpu()
                if not torch.isfinite(p).all():
                    raise RuntimeError('模型返回异常概率。')
                probabilities.extend(p.tolist())
        return dict(results=[dict(filename=f['name'], fake_probability=p,
                                 prediction='AI生成' if p >= .5 else '真实倾向')
                             for f, p in zip(files, probabilities)],
                    device=torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU',
                    seconds=round(time.perf_counter() - start, 3))


ENGINE = Engine()


class Handler(BaseHTTPRequestHandler):
    def allowed(self):
        hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        return self.headers.get('Host') in hosts

    def reply(self, status, value, content_type='application/json; charset=utf-8'):
        body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return self.reply(403, {'error': '仅允许本机访问。'})
        if self.path == '/':
            html = (ROOT / 'local_web/index.html').read_text(encoding='utf-8').replace('__LOCAL_TOKEN__', TOKEN)
            return self.reply(200, html.encode('utf-8'), 'text/html; charset=utf-8')
        if self.path == '/api/status':
            return self.reply(200, dict(service='facecheck-public-local', loaded=ENGINE.model is not None,
                                        device=ENGINE.device, busy=LOCK.locked()))
        if self.path == '/favicon.ico':
            return self.reply(204, b'', 'image/x-icon')
        return self.reply(404, {'error': '页面不存在。'})

    def do_POST(self):
        origin = self.headers.get('Origin')
        origins = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
        if not self.allowed() or (origin and origin not in origins) or not secrets.compare_digest(self.headers.get('X-Local-Token', ''), TOKEN):
            return self.reply(403, {'error': '请从本机网页发起检测。'})
        if self.path == '/api/shutdown':
            self.reply(200, {'stopped': True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if self.path != '/api/predict':
            return self.reply(404, {'error': '接口不存在。'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return self.reply(400, {'error': '请求长度无效。'})
        if not 0 < length <= MAX_BODY:
            return self.reply(413, {'error': '本次图片总体积过大，请减少图片数量。'})
        if self.headers.get_content_type() != 'application/json':
            return self.reply(415, {'error': '请求格式错误。'})
        if not LOCK.acquire(blocking=False):
            return self.reply(409, {'error': '正在检测，请稍后再试。'})
        try:
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError('请求格式错误。')
            result = ENGINE.predict(request.get('files'))
            self.reply(200, result)
        except (ValueError, UnicodeError) as e:
            self.reply(400, {'error': str(e)})
        except Exception:
            import traceback
            traceback.print_exc()
            self.reply(500, {'error': '检测未完成，请重试或查看本地服务日志。'})
        finally:
            LOCK.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == '__main__':
    main()
