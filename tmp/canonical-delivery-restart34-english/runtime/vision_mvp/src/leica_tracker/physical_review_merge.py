"""将明确接受的预览合并到仍待审核的草案中，不连接设备或产生动作。
匹配记录是人为声明，不构成硬件证明、路径审核或对动力学及停止行为的批准。"""
import argparse
import copy
import json
from pathlib import Path

from .physical_intro_program import _vector
from .physical_robot_model import model_for_profile
from .physical_runtime import validate_preview


def merge_preview(draft,report):
    if not isinstance(draft,dict) or draft.get('status')!='DRAFT' or not isinstance(draft.get('intro'),dict):
        raise ValueError('An unapproved physical draft is required')
    intro=draft['intro']
    robot_model=model_for_profile(draft)
    if model_for_profile(intro)!=robot_model:
        raise ValueError('Candidate draft and intro robot models differ')
    try:
        q=_vector(intro['reference_joints'],'reference joints')
        tcp=_vector(intro['tcp'],'TCP')
        points=intro['waypoints']
        if not isinstance(points,list) or len(points)!=14:raise ValueError('Exactly 14 candidates required')
        points=[_vector(point,'waypoint') for point in points]
        validate_preview(report,draft['robot_ip'],q,tcp,points,
                         reference_pose=intro.get('reference_pose'),robot_model=robot_model)
    except (KeyError,TypeError) as exc:
        raise ValueError('Draft or preview is incomplete') from exc
    result=copy.deepcopy(draft)
    result['intro_preview']=copy.deepcopy(report)
    return result


def main(argv=None):
    root=Path(__file__).resolve().parents[3]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft',type=Path,required=True)
    parser.add_argument('--acceptance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    draft,acceptance,output=(p.resolve() for p in (args.draft,args.acceptance,args.output))
    if any(not p.is_relative_to(root) for p in (draft,acceptance,output)):
        raise ValueError('All files must be inside the project workspace')
    result=merge_preview(json.loads(draft.read_text(encoding='utf-8-sig')),
                         json.loads(acceptance.read_text(encoding='utf-8-sig')))
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print('Preview merged into an unapproved draft; no robot connection or motion.')
    return 0


if __name__=='__main__':raise SystemExit(main())
