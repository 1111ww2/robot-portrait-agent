"""手动预览实体机器人十四点动作，仅允许已审核的分段。默认离线列出步骤；
--templates 写入待审核草稿，--execute 只启动配对的访客页面。
每输入一次 next 才打开一个已审核步骤；stop 抑制当前 RTDE 动作并保留独立读取通道供检查。
此功能不代替急停或安全等级控制器。"""
import argparse,importlib.util,json,os,queue,sys,threading,time
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
VISION=ROOT/'vision_mvp'


def _launcher():
    path=VISION/'scripts'/'run-physical-live.py'
    spec=importlib.util.spec_from_file_location('physical_live_launcher',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _inside(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Path must be inside this workspace')
    return path


def write_templates(draft,directory):
    from leica_tracker.physical_step_preview import draft_step_template
    directory=_inside(directory)
    directory.mkdir(parents=True,exist_ok=True)
    paths=[directory/f'{i:02}.json' for i in range(1,15)]
    if any(path.exists() for path in paths):raise ValueError('Review template already exists; choose a new directory')
    for i,path in enumerate(paths):
        with path.open('x',encoding='utf-8') as f:json.dump(draft_step_template(draft,i),f,indent=2)
    return paths


def _show(preview,output):
    state=preview.snapshot()
    print(json.dumps(state,allow_nan=False),file=output,flush=True)
    if preview.observations:
        print('Last observed point: '+json.dumps(preview.observations[-1],allow_nan=False),file=output,flush=True)
    if state['state']=='STOPPED':
        sample=preview.stopped_feedback()
        if sample is not None:
            data,stamp=sample
            print('Latest independent feedback: '+json.dumps({'sampled_at':stamp,'joints':data.get('joints'),
                'tcp_pose':data.get('tcp_pose'),'tcp_speed':data.get('tcp_speed'),
                'joint_speed':data.get('joint_speed')},allow_nan=False),file=output,flush=True)


def write_current_template(preview,review_dir):
    """根据最近一次测得的 TCP 姿态生成待审核模板。"""
    from leica_tracker.physical_step_preview import draft_step_template
    with preview._lock:
        if preview.state!='WAITING' or preview._opening or preview.index>=len(preview.points):
            raise ValueError('Wait for a completed point before drafting the next segment')
        index=preview.index;draft=preview.draft;observations=list(preview.observations)
    item=draft_step_template(draft,index,observations=observations)
    path=_inside(review_dir/f'{index+1:02}-live.json')
    with path.open('x',encoding='utf-8') as f:json.dump(item,f,indent=2)
    return path


def step_review_path(review_dir,index):
    live=_inside(review_dir/f'{index+1:02}-live.json')
    return live if live.exists() else _inside(review_dir/f'{index+1:02}.json')


def shared_next(preview,plan,draft_path,plan_path, *, prepare=None):
    """读取最新状态，按未变更的审核记录准备单步动作。"""
    from leica_tracker.physical_step_preview import prepare_shared_step
    prepare=prepare_shared_step if prepare is None else prepare
    def unchanged():
        draft=json.loads(_inside(draft_path).read_text(encoding='utf-8-sig'))
        reviewed=json.loads(_inside(plan_path).read_text(encoding='utf-8-sig'))
        return plan.matches(draft,reviewed)
    try:
        if not unchanged():raise ValueError('Reviewed plan or draft changed')
        with preview._lock:
            if preview.state!='WAITING' or preview._opening:raise RuntimeError('Preview is not waiting for Next')
            index=preview.index;observations=list(preview.observations)
        profile=prepare(preview.host,plan,index,observations)
        if not unchanged():raise ValueError('Reviewed plan or draft changed during measurement')
        preview.next(profile)
    except Exception:
        preview.stop('Shared plan or measurement unavailable; no automatic retry')
        raise


def interactive(preview,review_dir,report_path, *, stdin=sys.stdin,output=sys.stdout,guest_process=None,
                shared_plan=None,plan_path=None,draft_path=None):
    commands=queue.Queue();worker_error=queue.Queue();last_writer=last_reader=None
    def read_input():
        for line in stdin:commands.put(line.strip().lower())
        commands.put('eof')
    threading.Thread(target=read_input,name='preview-operator-input',daemon=True).start()
    print('Type next, template, status, stop, close, or accept after all 14 points.',file=output,flush=True)
    _show(preview,output)
    def start_next(profile=None):
        try:
            if shared_plan is None:preview.next(profile)
            else:shared_next(preview,shared_plan,draft_path,plan_path)
        except Exception as error:worker_error.put(str(error))
    while True:
        if guest_process is not None and guest_process.poll() is not None:
            preview.stop('iPad guest exited');raise RuntimeError('iPad guest exited')
        try:error=worker_error.get_nowait()
        except queue.Empty:error=None
        if error:print('Step could not start: '+error,file=output,flush=True);_show(preview,output)
        try:command=commands.get_nowait()
        except queue.Empty:command=None
        if command=='next':
            state=preview.snapshot();index=state['index']
            if state['state']!='WAITING' or state['opening']:
                print('Not waiting for Next.',file=output,flush=True)
            else:
                try:
                    profile=None
                    if shared_plan is None:
                        path=step_review_path(review_dir,index)
                        profile=json.loads(path.read_text(encoding='utf-8-sig'))
                    threading.Thread(target=start_next,args=(profile,),daemon=True).start()
                except (OSError,ValueError) as error:print('Review file unavailable: '+str(error),file=output,flush=True)
        elif command=='template':
            try:
                if shared_plan is not None:raise ValueError('Shared review already supplies the step parameters')
                path=write_current_template(preview,review_dir)
                print('Unreviewed current-step template: '+str(path),file=output,flush=True)
            except (OSError,ValueError) as error:print('Template not written: '+str(error),file=output,flush=True)
        elif command=='status':_show(preview,output)
        elif command=='stop':
            preview.stop();print('Stop requested. Inspect independent feedback with status; close after inspection.',file=output,flush=True)
        elif command=='accept':
            try:
                record=preview.acceptance('accept')
                with report_path.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,allow_nan=False)
                print('Operator acceptance recorded at '+str(report_path),file=output,flush=True)
            except (OSError,ValueError) as error:print('Acceptance not recorded: '+str(error),file=output,flush=True)
        elif command in ('close','eof'):
            preview.close();return
        elif command:print('Unknown command.',file=output,flush=True)
        state=preview.snapshot()
        if state['state']=='MOVING':
            with preview._lock:active=preview.active
            if active is not None:
                try:
                    writer,ws=active.writer.latest;reader,rs=active.reader.latest;now=time.monotonic()
                    if (active.writer.failure or active.reader.failure or not active.gate.can_send or
                            not 0<=now-ws<=.15 or not 0<=now-rs<=.15):
                        raise RuntimeError('Step watchdog or telemetry lost')
                    if (last_writer is None or ws>last_writer) and (last_reader is None or rs>last_reader):
                        result=preview.tick();last_writer=active.progress.last_writer;last_reader=active.progress.last_reader
                        if result in ('WAITING','DONE'):
                            last_writer=last_reader=None;_show(preview,output)
                except Exception as error:
                    preview.stop('Step fault');print('Step stopped: '+str(error),file=output,flush=True)
        time.sleep(.01)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft',type=Path,required=True)
    parser.add_argument('--templates',type=Path,help='write 14 unapproved review templates and exit')
    parser.add_argument('--plan-template',type=Path,help='write one unapproved shared plan and exit')
    parser.add_argument('--plan',type=Path,help='explicitly reviewed 14-segment plan with common parameters')
    parser.add_argument('--reviews',type=Path,help='directory containing 01.json through 14.json')
    parser.add_argument('--guest-session',type=Path,default=ROOT/'tmp'/'guidance-live'/'guest-link.json')
    parser.add_argument('--report',type=Path,default=ROOT/'tmp'/'robot-network'/'intro-preview-accepted.json')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    draft_path=_inside(args.draft)
    draft=json.loads(draft_path.read_text(encoding='utf-8-sig'))
    from leica_tracker.physical_step_preview import draft_step_template,shared_plan_template
    # 先验证草稿中的每个点，再与正在运行的仿真器交互。
    for i in range(14):draft_step_template(draft,i)
    if args.plan_template:
        path=_inside(args.plan_template)
        with path.open('x',encoding='utf-8') as f:json.dump(shared_plan_template(draft),f,indent=2)
        print('Wrote one unapproved plan; no app or robot started.')
        return 0
    if args.templates:
        paths=write_templates(draft,args.templates)
        print(f'Wrote {len(paths)} unreviewed templates; no app or robot started.')
        return 0
    if not args.execute:
        for i in range(14):
            item=draft_step_template(draft,i)
            print(json.dumps({'point':i+1,'target_joints':item['target_joints'],'status':'DRAFT'}))
        return 0
    if (args.reviews is None)==(args.plan is None):parser.error('--execute requires exactly one of --reviews or --plan')
    reviews=_inside(args.reviews) if args.reviews is not None else None
    report=_inside(args.report);pair_path=_inside(args.guest_session)
    if reviews is not None and not reviews.is_dir():raise ValueError('Review directory does not exist')
    from leica_tracker.physical_step_preview import StepSequence,SharedPreviewPlan
    preflight=StepSequence(draft,draft.get('robot_ip'),lambda:None)
    shared_plan=None;plan_path=None
    if args.plan is not None:
        plan_path=_inside(args.plan)
        try:
            shared_plan=SharedPreviewPlan(draft,json.loads(plan_path.read_text(encoding='utf-8-sig')))
            first_review=shared_plan.preflight_profile()
        except (OSError,ValueError,KeyError,TypeError) as exc:
            raise ValueError('Shared 14-step review is missing or incomplete; existing guest was not changed') from exc
    else:
        first_path=step_review_path(reviews,0)
        try:first_review=json.loads(first_path.read_text(encoding='utf-8-sig'))
        except (OSError,ValueError) as exc:
            raise ValueError('First step review 01.json is missing or invalid; existing guest was not changed') from exc
    try:preflight.validate_next(first_review)
    except (ValueError,RuntimeError) as exc:
        raise ValueError('First step review does not match the candidate or is incomplete; existing guest was not changed') from exc
    report.parent.mkdir(parents=True,exist_ok=True)
    launcher=_launcher();port,key=launcher.validate_pairing(json.loads(pair_path.read_text(encoding='utf-8-sig')))
    # 切换前读取配对信息，以便复用已有密钥。
    launcher.ensure_guest_port_available(port)
    environment=launcher.camera_environment(os.environ,key)
    environment['LEICA_PHYSICAL_PREVIEW']='1'
    guest=subprocess.Popen(launcher.camera_command(sys.executable,port,ROOT/'.env.local'),cwd=VISION,
        stdout=subprocess.DEVNULL,env=environment)
    preview=None
    try:
        launcher._wait_for_guest(guest,port,key)
        from leica_tracker.resident_guest_watchdog import LocalGuestControlReader
        reader=LocalGuestControlReader(port,key,timeout=.1)
        print('On iPad, prepare the camera and press Start; waiting for its live guest session.',flush=True)
        deadline=time.monotonic()+120.;initial=None
        while time.monotonic()<deadline:
            if guest.poll() is not None:raise RuntimeError('iPad guest exited')
            state=reader()
            if state.get('active') is True and state.get('camera_connected') is True and state.get('camera_fresh') is True:
                initial=(state['instance'],state['epoch']);break
            time.sleep(.1)
        if initial is None:raise RuntimeError('iPad guest did not become active')
        def same_guest():
            state=reader()
            if (state.get('instance'),state.get('epoch'))!=initial:
                return {**state,'active':False}
            return state
        preview=StepSequence(draft,draft['robot_ip'],same_guest)
        interactive(preview,reviews,report,guest_process=guest,shared_plan=shared_plan,
                    plan_path=plan_path,draft_path=draft_path)
    finally:
        if preview is not None:preview.close()
        if guest.poll() is None:
            guest.terminate()
            try:guest.wait(timeout=3)
            except subprocess.TimeoutExpired:guest.kill();guest.wait(timeout=3)
    return 0


if __name__=='__main__':raise SystemExit(main())
