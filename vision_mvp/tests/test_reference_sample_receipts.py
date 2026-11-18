"""将真实参考位门控接入监管器，只注入模拟的静止遥测，不使用 HTTP 或机器人通道。
软件夹具进入 READY 后再替换引擎 tick，此处没有原生通道。"""
import io
import json
from email.message import Message
from types import SimpleNamespace
import threading

import pytest

from leica_tracker.guest_display import GuestDisplay
from leica_tracker import guest_display as display_module
from leica_tracker.physical_interaction_supervisor import LocalReferenceGate
from leica_tracker.reference_gate import ReferenceGate
from test_physical_interaction_supervisor import setup,ready,step,Immediate


class CoupledReference:
    def __init__(self,server,session):
        self.server,self.session=server,session
        self.now=10.
        self.samples=[]
        self.releases=[]

    def sample(self,session,pose,speed,sampled_at):
        self.server.sample(session,pose,speed,sampled_at,self.now)
        receipt=dict(ok=True,sampled_at=sampled_at,
            return_verified=self.server.return_verified(session,self.now))
        self.samples.append(dict(receipt,steady_since=self.server.steady_since))
        return receipt

    def release(self,session):
        self.releases.append(dict(now=self.now,verified=self.server.return_verified(session,self.now),
            steady_since=self.server.steady_since,last_sample=self.server.last_sample))
        self.server.release(session,self.now)
        return self.server.release_ns

    def fault(self):return True


def coupled(monkeypatch):
    cfg,motion,engine,_ref,supervisor=setup()
    ready(supervisor,cfg,motion)
    monkeypatch.setattr(engine,'tick',lambda *_args,**_kwargs:None)
    server=ReferenceGate();session=server.begin(cfg.neutral,9.9);server.acknowledge()
    ref=CoupledReference(server,session)
    supervisor.reference=ref;supervisor._session=session
    supervisor.state='MOVING';supervisor._path=[]
    supervisor.gate_timeout=.25
    return cfg,motion,supervisor,ref


def test_channel_skew_cannot_release_before_server_has_accepted_600ms(monkeypatch):
    cfg,motion,supervisor,ref=coupled(monkeypatch)
    for i in range(24):
        t=round(10.08+i*.04,8);ref.now=t
        step(supervisor,cfg,motion,t,joint_time=10.0 if i==0 else t)
        if t<=10.64:assert not ref.releases
        if supervisor.state=='OBSERVING':break
    assert supervisor.state=='OBSERVING'
    assert len(ref.releases)==1 and ref.releases[0]['verified'] is True
    release=ref.releases[0]
    assert release['last_sample']-release['steady_since']>=.6
    assert any(receipt['return_verified'] is False for receipt in ref.samples)
    assert not motion.stopped


def test_server_sample_gap_restarts_server_dwell_without_false_release(monkeypatch):
    cfg,motion,supervisor,ref=coupled(monkeypatch)
    delayed=[False]

    class OneDelayedSample(Immediate):
        def __init__(self,fn,*args):
            self.finished_at=ref.now
            if getattr(fn,'__name__',None)=='sample' and not delayed[0]:
                delayed[0]=True;self.finished_at+=.18
            try:self.value,self.error=fn(*args),None
            except Exception as exc:self.value,self.error=None,exc
            self.elapsed=self.finished_at-ref.now

        @property
        def done(self):return ref.now>=self.finished_at

    supervisor.worker_factory=OneDelayedSample
    for i in range(65):
        t=round(10.+i*.02,8);ref.now=t
        step(supervisor,cfg,motion,t)
        if t<=10.76:assert not ref.releases
        if supervisor.state=='OBSERVING':break
    assert supervisor.state=='OBSERVING' and len(ref.releases)==1
    assert ref.samples[0]['steady_since']==10.
    assert ref.samples[1]['sampled_at']-ref.samples[0]['sampled_at']>.15
    assert ref.samples[1]['steady_since']==ref.samples[1]['sampled_at']
    release=ref.releases[0]
    assert release['verified'] is True and release['last_sample']-release['steady_since']>=.6
    assert not motion.stopped


def test_delayed_ready_receipt_is_not_released_after_its_sample_has_gone_stale(monkeypatch):
    cfg,motion,supervisor,ref=coupled(monkeypatch)
    delayed=[False]

    class OneDelayedReady(Immediate):
        def __init__(self,fn,*args):
            self.finished_at=ref.now
            try:self.value,self.error=fn(*args),None
            except Exception as exc:self.value,self.error=None,exc
            if (getattr(fn,'__name__',None)=='sample' and isinstance(self.value,dict)
                    and self.value.get('return_verified') is True and not delayed[0]):
                delayed[0]=True;self.finished_at+=.18
            self.elapsed=self.finished_at-ref.now

        @property
        def done(self):return ref.now>=self.finished_at

    supervisor.worker_factory=OneDelayedReady
    for i in range(100):
        t=round(10.+i*.02,8);ref.now=t
        step(supervisor,cfg,motion,t)
        if supervisor.state=='OBSERVING':break
    assert delayed[0] and supervisor.state=='OBSERVING'
    assert len(ref.releases)==1 and ref.releases[0]['verified'] is True
    release=ref.releases[0]
    assert 0<=release['now']-release['last_sample']<=.15
    assert not motion.stopped


@pytest.mark.parametrize('receipt',[
    True,
    {'ok':True,'sampled_at':10.08},
    {'ok':True,'sampled_at':10.08,'return_verified':1},
    {'ok':True,'sampled_at':float('nan'),'return_verified':False},
    # 此值等于此前各关节独立计算的时间，与实际发送的 TCP 时间不同。
    {'ok':True,'sampled_at':10.0,'return_verified':True},
])
def test_supervisor_rejects_legacy_or_malformed_or_wrong_payload_receipt(monkeypatch,receipt):
    cfg,motion,supervisor,ref=coupled(monkeypatch)
    monkeypatch.setattr(ref,'sample',lambda *_args:receipt)
    ref.now=10.08;step(supervisor,cfg,motion,10.08,joint_time=10.)
    ref.now=10.12
    with pytest.raises(RuntimeError,match='Reference sample rejected'):
        step(supervisor,cfg,motion,10.12)
    assert supervisor.state=='FAULT' and motion.stopped and not ref.releases


@pytest.mark.parametrize('receipt',[
    {'ok':True,'sampled_at':10.},
    {'ok':True,'sampled_at':10.,'return_verified':'false'},
    {'ok':True,'sampled_at':10.,'return_verified':1},
    {'ok':True,'sampled_at':float('nan'),'return_verified':False},
    {'ok':True,'sampled_at':float('inf'),'return_verified':False},
    {'ok':True,'sampled_at':10.01,'return_verified':False},
    {'ok':1,'sampled_at':10.,'return_verified':False},
    {'ok':True,'sampled_at':True,'return_verified':False},
])
def test_local_client_requires_strict_safe_sample_receipt(monkeypatch,receipt):
    client=LocalReferenceGate(1,'fixture-token',timeout=.25,
        control={'instance':'fixture','epoch':1,'started_ns':1},lease='fixture-lease')
    monkeypatch.setattr(client,'_request',lambda *_args:receipt)
    with pytest.raises(RuntimeError,match='receipt'):
        client.sample('fixture-session',[0.]*6,[0.]*6,10.)


def test_local_client_exports_only_safe_receipt_fields(monkeypatch):
    client=LocalReferenceGate(1,'fixture-token',timeout=.25,
        control={'instance':'fixture','epoch':1,'started_ns':1},lease='fixture-lease')
    monkeypatch.setattr(client,'_request',lambda *_args:dict(ok=True,sampled_at=10.,
        return_verified=False,private_message='must-not-leave-transport'))
    assert client.sample('fixture-session',[0.]*6,[0.]*6,10.)==dict(
        ok=True,sampled_at=10.,return_verified=False)


def test_return_verified_uses_unchanged_release_boundaries_and_is_read_only():
    server=ReferenceGate();session=server.begin([0.]*6,1.);server.acknowledge()
    for sampled in (1.,1.1,1.2,1.3,1.4,1.5,1.599):
        server.sample(session,[0.]*6,[0.]*6,sampled,sampled)
    assert not server.return_verified(session,1.599)
    with pytest.raises(ValueError,match='not verified'):server.release(session,1.599)
    server.sample(session,[0.]*6,[0.]*6,1.6,1.6)
    before=(server.session,server.steady_since,server.last_sample,server.release_ns)
    assert server.return_verified(session,1.6)
    assert not server.return_verified('wrong-session',1.6)
    assert not server.return_verified(session,1.751)
    assert not server.return_verified(session,float('nan'))
    assert (server.session,server.steady_since,server.last_sample,server.release_ns)==before
    server.release(session,1.6)
    assert not server.blocked and server.release_ns==1_600_000_000


def test_http_handler_emits_safe_exact_receipt_without_opening_a_socket(monkeypatch):
    # 跳过会绑定监听端口的 GuestDisplay.__init__。
    # 用内存中的标头、正文和响应调用真实 HTTP 路由方法。
    display=object.__new__(GuestDisplay)
    display.remote_capture=None;display._lock=threading.Lock()
    display._physical_feedback_enabled=False;display._physical_revocation=None
    server=display._reference_gate=ReferenceGate()
    session=server.begin([0.]*6,1.);server.acknowledge()
    clock=[1.]
    monkeypatch.setattr(display_module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    handler_class=display._handler()
    for sampled in (1.,1.1,1.2,1.3,1.4,1.5,1.7,1.8,1.9,2.,2.1,2.2,2.31):
        # 从 1.5 到 1.7 的采样间隔会重置停留计时，直到最后一个样本才满足 600 毫秒要求。
        clock[0]=sampled
        raw=json.dumps(dict(op='sample',session=session,pose=[0.]*6,speed=[0.]*6,sampled_at=sampled)).encode()
        handler=object.__new__(handler_class);handler.path='/reference-gate'
        handler.client_address=('127.0.0.1',1);handler.rfile=io.BytesIO(raw)
        handler.connection=SimpleNamespace(settimeout=lambda _timeout:None)
        headers=handler.headers=Message()
        for key,value in {'Origin':'http://fixture','Host':'fixture','Content-Type':'application/json',
                'Content-Length':str(len(raw)),'X-Guest-Client':'fixture-client'}.items():headers[key]=value
        handler.authorized=lambda **_kwargs:True
        replies=[];handler.reply=lambda code,data,*_args,**_kwargs:replies.append((code,data))
        handler.do_POST()
        assert replies==[(200,dict(ok=True,sampled_at=sampled,return_verified=sampled==2.31))]
    assert server.steady_since==1.7 and server.last_sample==2.31
