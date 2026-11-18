"""通过不回显的输入保存 Anthropic 密钥，保留其他本地设置。"""
import getpass
import os
from pathlib import Path
import warnings


def main():
    target = Path(__file__).resolve().parents[2] / '.env.local'
    if target.is_symlink():
        raise SystemExit('Refusing a symlink configuration.')
    with warnings.catch_warnings():
        warnings.simplefilter('error', getpass.GetPassWarning)
        try:
            key = getpass.getpass('Anthropic API key (hidden): ').strip()
        except getpass.GetPassWarning:
            raise SystemExit('A hidden-input terminal is required; nothing written.')
    if not key.startswith('sk-ant-') or any(c in key for c in '\r\n\"\''):
        raise SystemExit('Invalid key format; nothing written.')
    lines = target.read_text(encoding='utf-8').splitlines() if target.exists() else []
    lines = [line for line in lines if line.partition('=')[0].strip() != 'ANTHROPIC_API_KEY']
    lines.append('ANTHROPIC_API_KEY=' + key)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    target.chmod(0o600)
    print('Saved local Anthropic configuration; existing settings preserved.')


if __name__ == '__main__':
    main()
