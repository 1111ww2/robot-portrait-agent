"""在当前工作区内准备固定版本的教学仿真器。"""
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'tmp' / 'python-lab-review'
ARCHIVE = 'UR5e-Python-Lab-0.3.8-source.tar.gz'
URL = 'https://etech.hoggenmueller.com/python-lab-updates/releases/0.3.8/' + ARCHIVE
SHA256 = 'c2694d65c8ebc696d453f7e641779630db427da4a3bbcc984db2c11aa740353a'


def download_archive(target):
    # 发布服务器会对 urllib 的默认客户端返回 HTTP 403。
    request=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0',
                                               'Accept':'application/octet-stream'})
    partial=target.with_suffix(target.suffix+'.part')
    try:
        with urllib.request.urlopen(request,timeout=45) as response,partial.open('wb') as f:
            shutil.copyfileobj(response,f)
        if hashlib.sha256(partial.read_bytes()).hexdigest()!=SHA256:
            raise SystemExit('Source checksum mismatch; refusing to cache/install.')
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def main():
    if sys.version_info < (3, 12):
        raise SystemExit('Run this setup with the project Python 3.12 environment.')
    uv = shutil.which('uv')
    npm = shutil.which('npm.cmd' if sys.platform == 'win32' else 'npm')
    if not uv or not npm:
        raise SystemExit('Install uv and Node.js (20 or newer) first.')
    RUNTIME.mkdir(parents=True, exist_ok=True)
    archive = RUNTIME / ARCHIVE
    if not archive.exists():
        download_archive(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise SystemExit('Source checksum mismatch; refusing to install.')
    source = RUNTIME / 'source'
    if not source.exists():
        source.mkdir()
        with tarfile.open(archive) as bundle:
            bundle.extractall(source, filter='data')
    # 保留正在运行或已经在本地修改的源码目录。
    python = RUNTIME / '.venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.exists():
        subprocess.run([uv, 'venv', str(RUNTIME / '.venv'), '--python', sys.executable], check=True)
    subprocess.run([uv, 'pip', 'install', '--python', str(python), '-r',
                    str(ROOT / 'vision_mvp/sim/teacher-requirements.txt')], check=True)
    wheel = source / 'vendor/urx/urx-0.11.0+deco2021.6-py3-none-any.whl'
    subprocess.run([uv, 'pip', 'install', '--python', str(python), '--no-deps',
                    str(source), str(wheel), '-e', str(ROOT / 'vision_mvp')], check=True)
    subprocess.run([npm, 'ci', '--ignore-scripts'], cwd=source / 'web', check=True)
    # 发布包缺少 npm 构建后校验所用的测试夹具。先运行实际的 Vite 构建，再单独验证运行中的 API。
    subprocess.run([npm, 'exec', '--', 'vite', 'build', '--mode', 'workbench-v3'],
                   cwd=source / 'web', check=True)
    print('Teacher simulator installed. See LAPTOP_START_HERE.md for runtime checks on port 8766.')


if __name__ == '__main__':
    main()
