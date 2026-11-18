"""实际的PowerShell路由，使用仅文件模型模板和真正的验证器。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
ENTRY=ROOT/'vision_mvp/scripts/start-canonical-ipad.ps1'
pytestmark=pytest.mark.skipif(sys.platform!='win32',reason='Windows PowerShell entry')


@pytest.fixture
def canonical_root(tmp_path):
    root=tmp_path/"canonical root '\u56fe'";package=root/'vision_mvp/src/leica_tracker'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    for name in ('canonical_delivery.py','motion_configuration.py','native_gesture.py','physical_robot_model.py','cue_audio.py'):
        shutil.copyfile(ROOT/'vision_mvp/src/leica_tracker'/name,package/name)
    (package/'version.py').write_text('VALUE=1\n')
    scripts=root/'vision_mvp/scripts';scripts.mkdir()
    shutil.copyfile(ROOT/'vision_mvp/scripts/verify-canonical-runtime.py',scripts/'verify-canonical-runtime.py')
    python=root/'vision_mvp/.venv/Scripts/python.exe';python.parent.mkdir(parents=True)
    # 无需安装环境即可重复使用现有的工作区解释器。
    os.link(sys.executable,python)
    (python.parent.parent/'pyvenv.cfg').write_text('home = '+str(Path(sys._base_executable).parent)+'\ninclude-system-site-packages = false\n')
    bundle=root/'tmp/bundle';bundle.mkdir(parents=True)
    files=list(package.glob('*.py'))+list(scripts.glob('*.py'))
    hashes={}
    for source in files:
        name='runtime/'+source.relative_to(root).as_posix();destination=bundle/name
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.read_bytes())
        hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    source_hash=hashlib.sha256((json.dumps(dict(sorted(hashes.items())),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    manifest=dict(shared_runtime_source_sha256=source_hash,runtime_files=hashes,
                  joint_speed_rad_s=.7,workflow='ONE_SHARED_NATIVE_RUNTIME',models={})
    profiles=bundle/'profiles';profiles.mkdir()
    for model in ('ur5e','ur10e'):
        profile=dict(robot_model=model,intro=dict(robot_model=model,joint_speed=.7),
            installation=dict(robot_model=model,base_rate=.7,gesture=dict(joint_speed_limit=.7)))
        (profiles/(model+'.json')).write_text(json.dumps(profile))
        manifest['models'][model]=dict(profile='profiles/'+model+'.json',entry='start-'+model+'.ps1')
        (bundle/('start-'+model+'.ps1')).write_text('''param([string]$RuntimeRoot,[string]$Profile,[string]$GuestSession)
@{model='''+"'"+model+"'"+''';profile=$Profile;guest=$GuestSession;root=$RuntimeRoot} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $RuntimeRoot 'device-entry.json')
exit 0
''')
    manifest['generated_files']={name:hashlib.sha256((bundle/name).read_bytes()).hexdigest()
        for model in ('ur5e','ur10e') for name in (f'start-{model}.ps1',f'profiles/{model}.json')}
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    (root/'tmp/canonical-delivery-current.json').write_text(json.dumps(dict(bundle_path='tmp/bundle',
        shared_runtime_source_sha256=source_hash,
        manifest_sha256=hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest())))
    return root


def invoke(root,model='ur5e',*extra):
    return subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
        '-File',str(ENTRY),'-RobotModel',model,'-RuntimeRoot',str(root),*extra],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)


def read_json(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write_json(path,value):path.write_text(json.dumps(value),encoding='utf-8')


def test_offline_entry_runs_real_source_verifier_and_never_model_stub(canonical_root):
    result=invoke(canonical_root)
    assert result.returncode==0,result.stderr
    assert 'CANONICAL_RUNTIME_SOURCE_MATCHED' in result.stdout
    assert '0.7' in result.stdout and 'ur5e' in result.stdout.lower()
    assert not (canonical_root/'device-entry.json').exists()


@pytest.mark.parametrize('model',['ur5e','ur10e'])
def test_execute_routes_only_requested_current_bundle_model(canonical_root,model):
    result=invoke(canonical_root,model,'-Execute')
    assert result.returncode==0,result.stderr
    record=read_json(canonical_root/'device-entry.json')
    assert record['model']==model
    assert Path(record['profile'])==canonical_root/'tmp/bundle/profiles'/(model+'.json')
    assert Path(record['root'])==canonical_root


@pytest.mark.parametrize('failure',['outside','hash','stale_source','wrong_profile_model','wrong_manifest_model','old_speed'])
def test_invalid_pointer_or_model_never_reaches_device_entry(canonical_root,failure):
    pointer=canonical_root/'tmp/canonical-delivery-current.json';value=read_json(pointer)
    manifest_path=canonical_root/'tmp/bundle/manifest.json';manifest=read_json(manifest_path)
    profile_path=canonical_root/'tmp/bundle/profiles/ur5e.json';profile=read_json(profile_path)
    if failure=='outside':value['bundle_path']='../outside'
    elif failure=='hash':value['shared_runtime_source_sha256']='0'*64
    elif failure=='stale_source':(canonical_root/'vision_mvp/src/leica_tracker/version.py').write_text('VALUE=2\n')
    elif failure=='wrong_profile_model':profile['robot_model']='ur10e'
    elif failure=='wrong_manifest_model':manifest['models'].pop('ur5e')
    elif failure=='old_speed':profile['intro']['joint_speed']=.1
    write_json(pointer,value);write_json(manifest_path,manifest);write_json(profile_path,profile)
    result=invoke(canonical_root,'ur5e','-Execute')
    assert result.returncode!=0
    assert not (canonical_root/'device-entry.json').exists()


def test_execute_passes_explicit_profile_and_guest_and_propagates_failure(canonical_root):
    profile=canonical_root/'onsite profile.json'
    profile.write_bytes((canonical_root/'tmp/bundle/profiles/ur10e.json').read_bytes())
    guest=canonical_root/'session.json';guest.write_text('{}')
    entry=canonical_root/'tmp/bundle/start-ur10e.ps1'
    entry.write_text(entry.read_text().replace('exit 0','exit 19'))
    # 此测试在路由前故意准备一个新的已接受的仅文件的模板；单独的验证器测试拒绝未记录的生成的文件漂移。
    manifest_path=canonical_root/'tmp/bundle/manifest.json';manifest=read_json(manifest_path)
    manifest['generated_files']['start-ur10e.ps1']=hashlib.sha256(entry.read_bytes()).hexdigest()
    write_json(manifest_path,manifest)
    pointer_path=canonical_root/'tmp/canonical-delivery-current.json';pointer=read_json(pointer_path)
    pointer['manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    write_json(pointer_path,pointer)
    result=invoke(canonical_root,'ur10e','-Execute','-Profile',str(profile),'-GuestSession',str(guest))
    assert result.returncode==19,result.stderr
    record=read_json(canonical_root/'device-entry.json')
    assert Path(record['profile'])==profile and Path(record['guest'])==guest


@pytest.mark.parametrize('failure',['missing','malformed','different','mutated_manifest'])
def test_manifest_pointer_binding_is_required_before_any_model_entry(canonical_root,failure):
    pointer_path=canonical_root/'tmp/canonical-delivery-current.json';pointer=read_json(pointer_path)
    manifest_path=canonical_root/'tmp/bundle/manifest.json'
    original_source_hash=pointer['shared_runtime_source_sha256']
    if failure=='missing':pointer.pop('manifest_sha256')
    elif failure=='malformed':pointer['manifest_sha256']='not-a-sha256'
    elif failure=='different':pointer['manifest_sha256']='0'*64
    else:
        manifest=read_json(manifest_path)
        # 仅更改元数据。运行时身份和生成的文件哈希仍然通过源验证器，因此只有发布的配置文件绑定失败。
        manifest['unpublished_change']='must not run'
        write_json(manifest_path,manifest)
        assert manifest['shared_runtime_source_sha256']==original_source_hash
    write_json(pointer_path,pointer)
    result=invoke(canonical_root,'ur5e','-Execute')
    assert result.returncode!=0
    assert not (canonical_root/'device-entry.json').exists()


def test_all_public_buttons_route_through_current_canonical_pointer():
    for model in ('UR5e','UR10e'):
        for suffix in ('','-0.7'):
            text=(ROOT/('Start-Leica-'+model+suffix+'.cmd')).read_text()
            assert 'start-canonical-ipad.ps1' in text
            assert '-RobotModel '+model.lower() in text and '-Execute' in text
            assert 'physical-ipad-' not in text and 'final02' not in text
