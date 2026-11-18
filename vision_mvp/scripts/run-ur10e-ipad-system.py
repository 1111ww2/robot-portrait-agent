"""默认离线验证，使用 --execute 时才接管完整的现场 iPad 服务。"""
from pathlib import Path
from leica_tracker.physical_system import main


if __name__ == '__main__':
    raise SystemExit(main(default_model='ur10e',
                         default_profile=Path('vision_mvp/config/selected-ur10e-0.7.json')))
