"""离线验证独立的 UR5e 配置，--execute 仅供现场使用。"""
from pathlib import Path
import sys

from leica_tracker.physical_system import main


if __name__ == '__main__':
    try:
        result = main(default_model='ur5e',
                      default_profile=Path('vision_mvp/config/selected-ur5e-0.7.json'))
    except (ValueError, OSError) as exc:
        print(f'UR5e configuration is not ready: {exc}', file=sys.stderr)
        raise SystemExit(2) from None
    raise SystemExit(result)
