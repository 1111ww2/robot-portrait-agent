"""通过本机自签 CA 的 HTTPS 服务提供已配对的 iPad 页面。"""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import signal
import socket
import sys
import time

from leica_tracker.guest_https import create_certificates,GuestHTTPS,load_pairing,HTTPSRequestTimings

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'tmp/guidance-live'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address',help='Laptop private LAN IPv4')
    parser.add_argument('--port',type=int,default=8768)
    parser.add_argument('--setup-port',type=int,default=8769)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--stop-file',type=Path)
    parser.add_argument('--timing-output',type=Path,
                        help='Opt-in private numeric timing capture, written once to a new workspace file')
    parser.add_argument('--timing-duration',type=float,default=30.,
                        help='Timing observation window, 1 to 120 seconds; HTTPS continues afterward')
    args=parser.parse_args(argv)
    if not math.isfinite(args.timing_duration) or not 1<=args.timing_duration<=120:
        parser.error('Timing duration must be from 1 to 120 seconds')
    timing_path=None
    if args.timing_output is not None:
        timing_path=(args.timing_output if args.timing_output.is_absolute() else ROOT/args.timing_output).resolve()
        if timing_path==ROOT.resolve() or not timing_path.is_relative_to(ROOT.resolve()):
            parser.error('Timing output must remain inside the project workspace')
        if timing_path.exists():parser.error('Timing output already exists; preserve it and choose a new file')
        if args.prepare_only:parser.error('Timing capture requires the HTTPS service, without --prepare-only')
    if args.stop_file and args.stop_file.exists():parser.error('Stop file already exists')
    pair=load_pairing(OUTPUT/'guest-link.json',OUTPUT/'https/pair.json')
    address=args.address
    if not address:
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as route:
            route.connect(('192.0.2.1',9));address=route.getsockname()[0]
    files=create_certificates(OUTPUT/'https',address)
    setup_url=f'http://{address}:{args.setup_port}/leica-root.mobileconfig'
    app_url=f"https://{address}:{args.port}/?key={pair['key']}"
    import qrcode
    qrcode.make(setup_url).save(OUTPUT/'https-certificate-qr.png')
    qrcode.make(app_url).save(OUTPUT/'https-ipad-qr.png')
    fingerprint=hashlib.sha256(files['der'].read_bytes()).hexdigest().upper()
    page='''<!doctype html><meta charset="utf-8"><meta name="referrer" content="no-referrer">
<title>iPad HTTPS pairing</title><style>body{font:18px system-ui;max-width:900px;margin:32px auto;padding:24px;line-height:1.6}img{width:280px}section{display:inline-block;vertical-align:top;width:45%;margin-right:4%}code{overflow-wrap:anywhere;font-size:12px}</style>
<h1>iPad HTTPS · First connection</h1><section><h2>1. Install certificate</h2><img src="https-certificate-qr.png"><p><a href="'''+html.escape(setup_url,quote=True)+'''">Download the local certificate profile</a></p></section>
<section><h2>2. Open the interaction page</h2><img src="https-ipad-qr.png"><p><a href="'''+html.escape(app_url,quote=True)+'''">Trust the certificate, then open the HTTPS page</a></p></section>
<ol><li>Download the certificate on the left using iPad Safari. Open Settings > Profile Downloaded > Install.</li>
<li>Open Settings > General > About > Certificate Trust Settings. Enable full trust for Leica Local iPad Root.</li>
<li>Scan the code on the right, enable the front camera and grant permission. Once the preview appears, tap Start. This legacy setup page also describes sound and speech controls; their availability depends on the selected workflow. If a certificate warning appears, check installation, trust and the computer network address.</li></ol>
<p>This file installs only the project root certificate; it does not include device management. iPad camera mode requires HTTPS. Start the computer application with --camera-source ipad.</p>
<p>Root certificate SHA-256：<code>'''+fingerprint+'''</code></p>'''
    (OUTPUT/'https-ipad-pair.html').write_text(page,encoding='utf-8')
    (OUTPUT/'https-public-status.json').write_text(json.dumps({'address':address,'port':args.port,'setup_port':args.setup_port,'root_sha256':fingerprint}),encoding='utf-8')
    print('Local HTTPS pairing files prepared under tmp/guidance-live; pairing token not logged.',flush=True)
    if args.prepare_only:return
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    capture=HTTPSRequestTimings(timing_path,duration=args.timing_duration) if timing_path is not None else None
    timing_reported=False
    def finish_timing(*,due_only=False):
        nonlocal timing_reported
        if capture is None:return
        done=capture.finish_if_due() if due_only else capture.finish()
        if done and not timing_reported:
            timing_reported=True
            if capture.write_error:print(f'HTTPS_TIMING_WRITE_ERROR {capture.write_error}',file=sys.stderr,flush=True)
            else:print('HTTPS_TIMING_CAPTURE_COMPLETE: private numeric capture closed; HTTPS continues.',flush=True)
    options={'timing_callback':capture.record} if capture is not None else {}
    try:
        with GuestHTTPS(files,upstream_port=pair['port'],port=args.port,setup_port=args.setup_port,addresses=[address],**options):
            print(f'HTTPS port {args.port}; certificate-only setup port {args.setup_port}; no system trust changed.',flush=True)
            while not stopping and not (args.stop_file and args.stop_file.exists()):
                finish_timing(due_only=True)
                time.sleep(.2)
    finally:finish_timing()


if __name__=='__main__':main()
