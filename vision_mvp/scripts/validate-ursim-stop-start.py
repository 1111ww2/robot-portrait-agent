"""仅运行限定在官方 URSim 环境中的“停止后重新开始”验证。"""
from leica_tracker.ursim_restart_validation import main


if __name__ == '__main__':
    raise SystemExit(main())
