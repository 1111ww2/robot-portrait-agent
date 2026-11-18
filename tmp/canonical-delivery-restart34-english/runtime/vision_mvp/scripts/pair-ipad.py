"""在本机生成并打开配对二维码，避免把配对网址传给在线生成服务。"""
import argparse
import html
import io
import ipaddress
import json
import os
from pathlib import Path
import socket
import urllib.request
import webbrowser


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',help='Laptop LAN IPv4, if automatic routing chooses the wrong adapter')
    p.add_argument('--no-open',action='store_true')
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    link=root/'tmp/guidance-live/guest-link.json'
    if not link.exists():raise SystemExit('Start run-guidance-live.py with --guest-display first.')
    data=json.loads(link.read_text());port=int(data['port']);key=data['key']
    req=urllib.request.Request(f'http://127.0.0.1:{port}/state',headers={'X-Guest-Token':key})
    try:
        with urllib.request.urlopen(req,timeout=2) as r:json.load(r)
    except OSError:raise SystemExit('Guest server is not available for this pairing file. Check camera/launcher log.')
    host=args.host
    if not host:
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:
            # 仅查询路由；此处 UDP connect 不发送数据包、图像或令牌。
            s.connect(('192.0.2.1',9));host=s.getsockname()[0]
    address=ipaddress.ip_address(host)
    if address.version!=4 or not address.is_private or address.is_loopback or address.is_link_local:
        raise SystemExit('Use --host with the laptop LAN IPv4, not loopback/public/link-local.')
    import qrcode
    import qrcode.image.svg
    url=f'http://{address}:{port}/?key={key}'
    svg=io.BytesIO();qrcode.make(url,image_factory=qrcode.image.svg.SvgPathImage).save(svg)
    page=('<!doctype html><meta charset="utf-8"><meta name="referrer" content="no-referrer">'
          '<title>Local iPad pairing</title><style>body{font:20px sans-serif;max-width:800px;margin:40px auto;padding:24px}svg{width:360px;height:360px}a{overflow-wrap:anywhere}</style>'
          '<h1>Who Frames Whom · iPad pairing</h1><p>Connect the iPad and this computer to the same reachable local network. Scan the code with the camera and open the link in Safari.</p>'
          +svg.getvalue().decode()+'<p><a href="'+html.escape(url,quote=True)+'">'+html.escape(url)+'</a></p>'
          '<p>Valid only for this service session. Do not share or upload this file. If connection fails, check the LAN IP, device isolation and firewall.</p>')
    target=link.with_name('ipad-pair.html')
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as f:f.write(page)
    target.chmod(0o600)
    if not args.no_open:webbrowser.open(target.as_uri())
    print('Private local QR page: '+str(target)+' (do not upload or paste its contents).')


if __name__=='__main__':main()
