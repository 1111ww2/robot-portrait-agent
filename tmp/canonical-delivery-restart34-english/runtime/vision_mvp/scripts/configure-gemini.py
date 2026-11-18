"""在本机输入 Gemini 密钥，不回显，也不从命令行参数读取密钥。"""
import argparse
import getpass
import os
from pathlib import Path
import warnings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replace',action='store_true')
    args=p.parse_args()
    target=Path(__file__).resolve().parents[2]/'.env.local'
    if target.exists() and not args.replace:
        raise SystemExit('.env.local already exists; test it first. --replace intentionally replaces its Gemini configuration.')
    with warnings.catch_warnings():
        warnings.simplefilter('error',getpass.GetPassWarning)
        try:key=getpass.getpass('Gemini API key (hidden, local terminal only): ').strip()
        except getpass.GetPassWarning:raise SystemExit('No hidden-input terminal available; open a local interactive terminal. Nothing written.')
    if not key or any(c in key for c in '\r\n\"\''):
        raise SystemExit('Invalid empty/multiline/quoted input; nothing written.')
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as f:f.write('GEMINI_API_KEY='+key+'\n')
    target.chmod(0o600)
    print('Saved ignored .env.local; key not printed. Windows users should keep this in their private user checkout.')


if __name__=='__main__':main()
