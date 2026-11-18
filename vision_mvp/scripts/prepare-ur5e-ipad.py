"""离线准备独立审核的 UR5e 运行包。默认草稿不能执行。"""
from pathlib import Path

from leica_tracker.physical_delivery import ROOT, main


if __name__ == '__main__':
    raise SystemExit(main(default_model='ur5e',
                          default_profile=ROOT / Path('vision_mvp/config/physical-ur5e.template.json')))
