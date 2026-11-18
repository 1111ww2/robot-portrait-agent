"""离线编译完整的已审核控制包和目标速度审计记录。"""
from leica_tracker.physical_delivery import main


if __name__=='__main__':
    raise SystemExit(main(default_model='ur10e'))
