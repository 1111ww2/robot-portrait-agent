"""离线检查运行包源码标识，不打开设备或网络连接。"""
import argparse
from pathlib import Path
import sys

# 使用冻结包随附的校验程序，避免调用已安装的旧版本。
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from leica_tracker.canonical_delivery import verify_runtime_source

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle',type=Path,required=True)
parser.add_argument('--runtime-root',type=Path,required=True)
args=parser.parse_args()
verify_runtime_source(args.bundle,args.runtime_root)
print('CANONICAL_RUNTIME_SOURCE_MATCHED — no device connection opened')
