"""原始空间RTDE合同与内存套接字；没有打开端点。"""
import importlib
import io
import json
import math
import queue
import socket
import struct
from types import SimpleNamespace

import pytest

from leica_tracker.ur_rtde_transport import AckReceiptClock, validate_spatial_target


def module():
    name='leica_tracker.atomic_rtde_transport'
    assert importlib.util.find_spec(name) is not None, 'Batched framed spatial transport is missing'
    return importlib.import_module(name)


def frame(kind,body=b''):return struct.pack('!HB',len(body)+3,kind)+body


OUTPUT_TYPES=b'DOUBLE,INT32,INT32,INT32,VECTOR6D,VECTOR6D,VECTOR6D'
INPUT_TYPES=b'INT32,INT32,INT32,DOUBLE,DOUBLE,DOUBLE'


class Clock:
    def __init__(self,value=10.):self.value=value
    def __call__(self):return self.value


class Socket:
    def __init__(self,data=b'',chunks=None):
        self.chunks=list(chunks) if chunks is not None else [data]
        self.writes=[];self.timeouts=[];self.options=[];self.closed=False;self.after_send=None
    def setsockopt(self,*value):self.options.append(value)
    def settimeout(self,value):assert value>0;self.timeouts.append(value)
    def recv(self,size):
        if not self.chunks:raise socket.timeout()
        value=self.chunks.pop(0)
        if len(value)>size:self.chunks.insert(0,value[size:]);value=value[:size]
        return value
    def sendall(self,value):
        self.writes.append(value)
        if self.after_send:self.after_send()
    def shutdown(self,how):pass
    def close(self):self.closed=True


def replies(readonly=False,*,version=b'\x01',output=None,inputs=None,start=b'\x01'):
    output=output if output is not None else b'\x01'+OUTPUT_TYPES+(b',VECTOR6D' if readonly else b'')
    inputs=inputs if inputs is not None else b'\x02'+INPUT_TYPES
    return frame(86,version)+frame(79,output)+(b'' if readonly else frame(73,inputs))+frame(83,start)


def output_packet(stamp=1.,state=0,ack=1,session=123,*,readonly=False,recipe=1,values=None):
    vectors=[0.]*6+[.1,.2,.3,0.,0.,0.]+[.4]*6+([0.]*6 if readonly else []) if values is None else values
    return frame(85,struct.pack('!Bdiii'+str(len(vectors))+'d',recipe,stamp,state,ack,session,*vectors))


def channel(readonly=False,**kwargs):
    adapter=module();sock=kwargs.pop('sock',Socket(replies(readonly)));clock=kwargs.pop('clock',Clock())
    def factory(address,timeout):
        assert address==('unused.invalid',30004) and timeout==.5
        return sock
    instance=adapter.SpatialRTDEChannel('unused.invalid',receive_only=readonly,
        validate_update=validate_spatial_target,socket_factory=factory,clock=clock,**kwargs)
    return instance,sock,clock


def test_v2_handshake_is_exact_and_one_socket_owns_all_spatial_fields():
    instance,sock,clock=channel()
    assert [data[2] for data in sock.writes]==[86,79,73,83]
    assert sock.writes[0]==frame(86,b'\x00\x02')
    assert sock.writes[1]==frame(79,struct.pack('!d',50.)+
        b'timestamp,output_int_register_42,output_int_register_43,output_int_register_40,actual_TCP_speed,actual_TCP_pose,actual_q')
    assert sock.writes[2]==frame(73,
        b'input_int_register_42,input_int_register_43,input_int_register_44,input_double_register_42,input_double_register_43,input_double_register_44')
    assert (socket.IPPROTO_TCP,socket.TCP_NODELAY,1) in sock.options
    instance.close();assert sock.closed


def test_one_sendall_contains_negative_stage_then_identical_positive_commit():
    instance,sock,clock=channel();before=len(sock.writes)
    instance.send((7,2,123,[.1,.2,.3]),10.15)
    assert len(sock.writes)==before+1
    assert sock.writes[-1]==frame(85,struct.pack('!Biiiddd',2,-7,2,123,.1,.2,.3))+frame(85,struct.pack('!Biiiddd',2,7,2,123,.1,.2,.3))
    instance.close()


@pytest.mark.parametrize('prefix',range(81))
def test_partial_tcp_frames_never_commit_unstaged_or_mixed_xyz(prefix):
    data=module().encode_spatial_batch(2,(7,2,123,[.1,.2,.3]),validate_spatial_target)
    incoming=data[:prefix];registers=dict(sequence=99,flag=0,session=456,x=-.1,y=-.2,z=-.3)
    accepted=[]
    while len(incoming)>=3:
        size,kind=struct.unpack('!HB',incoming[:3])
        if len(incoming)<size:break
        recipe,seq,flag,session,x,y,z=struct.unpack('!Biiiddd',incoming[3:size])
        assert kind==85 and recipe==2
        for key,value in zip(('sequence','flag','session','x','y','z'),(seq,flag,session,x,y,z)):
            registers[key]=value
            # 在每个应用的电线字段之间建模任意原生读取。
            if registers['sequence']>0 and registers['session']==123:
                accepted.append(dict(registers))
        incoming=incoming[size:]
    assert all(value==dict(sequence=7,flag=2,session=123,x=.1,y=.2,z=.3) for value in accepted)
    assert bool(accepted)==(prefix==80)


def test_read_only_never_claims_an_input_recipe_or_sends_any_data_update():
    instance,sock,clock=channel(True)
    assert [data[2] for data in sock.writes]==[86,79,83]
    before=list(sock.writes)
    with pytest.raises(RuntimeError):instance.send((1,0,123,[0,0,0]),10.15)
    assert sock.writes==before
    instance.close()


@pytest.mark.parametrize('kwargs',[dict(version=b'\x00'),dict(output=b'\x00'+OUTPUT_TYPES),
    dict(output=b'\x01DOUBLE,NOT_FOUND'),dict(inputs=b'\x00'+INPUT_TYPES),
    dict(inputs=b'\x02INT32,IN_USE'),dict(start=b'\x00')])
def test_rejected_or_mistyped_handshake_fails_closed_without_data_write(kwargs):
    sock=Socket(replies(**kwargs))
    with pytest.raises((ValueError,RuntimeError)):channel(sock=sock)
    assert sock.closed and not any(value[2]==85 for value in sock.writes)


def test_fragmented_handshake_cannot_refresh_the_total_three_second_deadline():
    sock=Socket(chunks=[bytes([value]) for value in replies()]);clock=Clock()
    original=sock.recv
    def receive(size):clock.value+=.2;return original(size)
    sock.recv=receive
    with pytest.raises(TimeoutError):channel(sock=sock,clock=clock)
    assert sock.closed and len(sock.writes)==2 and 13.0<=clock.value<=13.21


def delayed_socket(clock,delays,*,readonly=False):
    sock=Socket(replies(readonly));current=[None];waiting=[False]
    original_send=sock.sendall;original_recv=sock.recv
    def send(data):
        original_send(data);current[0]=data[2];waiting[0]=True
    def receive(size):
        if waiting[0]:clock.value+=delays.get(current[0],0.);waiting[0]=False
        return original_recv(size)
    sock.sendall=send;sock.recv=receive
    return sock


def test_realistic_550ms_version_reply_is_inside_total_startup_budget():
    clock=Clock();sock=delayed_socket(clock,{86:.55,79:.01,83:.01},readonly=True)
    instance,sock,clock=channel(True,sock=sock,clock=clock)
    assert [data[2] for data in sock.writes]==[86,79,83]
    assert math.isclose(clock.value,10.57,abs_tol=1e-12)
    assert not sock.closed
    instance.close()


def test_separate_handshake_phases_share_one_deadline_instead_of_resetting_it():
    clock=Clock();sock=delayed_socket(clock,{86:.8,79:.8,73:.8,83:.8})
    with pytest.raises(TimeoutError):channel(sock=sock,clock=clock)
    assert [data[2] for data in sock.writes]==[86,79,73,83]
    assert sock.closed and math.isclose(clock.value,13.2,abs_tol=1e-12)


def test_all_complete_feedback_frames_are_preserved_in_wire_order():
    instance,sock,clock=channel()
    sock.chunks=[output_packet(stamp=1.,ack=1)+output_packet(stamp=1.02,ack=2)]
    result=instance.poll()
    assert [(value['timestamp'],value['sequence']) for value in result]==[(1.,1),(1.02,2)]
    assert result[0]['tcp_pose']==[.1,.2,.3,0.,0.,0.] and result[0]['joints']==[.4]*6
    instance.close()


def test_first_output_may_precede_start_reply_without_being_lost():
    data=frame(86,b'\x01')+frame(79,b'\x01'+OUTPUT_TYPES)+frame(73,b'\x02'+INPUT_TYPES)
    data+=output_packet(stamp=1.,ack=0)+frame(83,b'\x01')
    instance,sock,clock=channel(sock=Socket(data))
    assert instance.poll()==[dict(timestamp=1.,state=0,sequence=0,session=123,
        tcp_speed=[0.]*6,tcp_pose=[.1,.2,.3,0.,0.,0.],joints=[.4]*6)]
    instance.close()


def test_partial_runtime_packet_cannot_publish_mixed_feedback():
    instance,sock,clock=channel(True);data=output_packet(readonly=True)
    sock.chunks=[data[:2],data[2:17],data[17:]]
    assert instance.poll()==[] and instance.poll()==[]
    value=instance.poll();assert len(value)==1 and value[0]['joint_speed']==[0.]*6
    instance.close()


@pytest.mark.parametrize('data',[frame(85,b'\x01'),output_packet(recipe=9),output_packet(stamp=math.nan),
    output_packet(ack=-1),output_packet(values=[math.inf]+[0.]*17),b'\x00\x02U',b'\xff\xffU'])
def test_invalid_runtime_framing_recipe_or_values_fail_without_fresh_feedback(data):
    instance,sock,clock=channel();sock.chunks=[data]
    with pytest.raises((ValueError,RuntimeError)):instance.poll()
    instance.close()


def test_fault_state_remains_visible_to_native_supervisor():
    instance,sock,clock=channel();sock.chunks=[output_packet(state=-12)]
    assert instance.poll()[0]['state']==-12
    instance.close()


@pytest.mark.parametrize('update',[(0,0,123,[0,0,0]),(-1,0,123,[0,0,0]),
    (1,True,123,[0,0,0]),(1,1,123,[0,math.nan,0])])
def test_external_invalid_update_cannot_send_even_invalidation(update):
    instance,sock,clock=channel();before=list(sock.writes)
    with pytest.raises(ValueError):instance.send(update,10.15)
    assert sock.writes==before;instance.close()


def test_commit_expired_before_send_never_writes_and_uncertain_send_is_never_replayed():
    instance,sock,clock=channel();before=list(sock.writes)
    with pytest.raises(TimeoutError):instance.send((1,0,123,[0,0,0]),9.99)
    assert sock.writes==before
    instance.close()
    instance,sock,clock=channel()
    def uncertain():raise OSError('uncertain partial send')
    sock.after_send=uncertain;before=len(sock.writes)
    with pytest.raises(OSError):instance.send((1,0,123,[0,0,0]),10.15)
    assert len(sock.writes)==before+1 and sock.closed
    with pytest.raises(RuntimeError):instance.send((1,0,123,[0,0,0]),10.15)
    assert len(sock.writes)==before+1


def test_sendall_finishing_after_absolute_commit_deadline_still_faults_and_closes():
    instance,sock,clock=channel();sock.after_send=lambda:setattr(clock,'value',10.151)
    with pytest.raises(TimeoutError):instance.send((1,0,123,[0,0,0]),10.15)
    assert sock.closed


def test_internal_negative_staging_never_bypasses_external_positive_validation():
    seen=[]
    def validate(update):seen.append(update);validate_spatial_target(update)
    update=(7,2,123,[.1,.2,.3])
    module().encode_spatial_batch(2,update,validate)
    assert seen[0]==update and all(value[0]==7 for value in seen)


def test_mutating_target_iterator_cannot_change_between_stage_and_commit():
    class Changing(list):
        def __init__(self):super().__init__([.1,.2,.3]);self.iterations=0
        def __iter__(self):
            self.iterations+=1
            return iter([self.iterations*.1,.2,.3])
    data=module().encode_spatial_batch(2,(7,2,123,Changing()),validate_spatial_target)
    negative=struct.unpack('!Biiiddd',data[3:40])
    positive=struct.unpack('!Biiiddd',data[43:80])
    assert negative[2:]==positive[2:]==(2,123,.1,.2,.3)


def test_actual_immutable_snapshot_is_revalidated_before_encoding_any_frame():
    target=[.1,.2,.3];calls=[]
    def validate(update):
        validate_spatial_target(update);calls.append(update)
        if len(calls)==1:target[1]=math.nan
    with pytest.raises(ValueError):module().encode_spatial_batch(2,(7,2,123,target),validate)


def test_waiting_for_write_lock_cannot_renew_the_original_commit_budget():
    instance,sock,clock=channel()
    class Lock:
        def acquire(self,*,timeout):clock.value=10.14;return True
        def release(self):pass
    instance.write_lock=Lock();sock.after_send=lambda:setattr(clock,'value',10.151)
    with pytest.raises(TimeoutError):instance.send((1,0,123,[0,0,0]),10.15)
    assert sock.closed


def test_controller_protocol_error_text_is_not_exposed_as_arbitrary_log_content():
    instance,sock,clock=channel();private=b'not-a-loggable-controller-string'
    sock.chunks=[frame(77,bytes([len(private)])+private+b'\x00\x01')]
    with pytest.raises(RuntimeError) as error:instance.poll()
    assert private.decode() not in str(error.value)
    instance.close()


def worker_frames(monkeypatch,frames,*,pending=None,receive_only=False):
    adapter=module();clock=Clock();position=[0];calls=[];output=io.StringIO()
    class Event:
        def is_set(self):return position[0]>=len(frames)
        def set(self):position[0]=len(frames)
    class Thread:
        def __init__(self,*,target,daemon):assert callable(target) and daemon is True
        def start(self):pass
    class Channel:
        def __init__(self,host,**kwargs):
            assert host=='unused.invalid' and kwargs['receive_only'] is receive_only
            assert kwargs['validate_update'] is validate_spatial_target
            assert kwargs['clock'] is clock
        def poll(self,*,timeout):
            assert timeout==.005
            stamp,ack,local=frames[position[0]];position[0]+=1;clock.value=local
            return [dict(timestamp=stamp,sequence=ack,session=123,state=0,
                tcp_speed=[0.]*6,tcp_pose=[0.,0.,.5,0.,0.,0.],joints=[0.]*6,
                **({'joint_speed':[0.]*6} if receive_only else {}))]
        def send(self,update,deadline):calls.append((update,deadline))
        def close(self):calls.append('closed')
    incoming=queue.Queue(maxsize=1)
    if pending is not None:incoming.put_nowait(pending)
    monkeypatch.setattr(adapter,'threading',SimpleNamespace(Event=Event,Thread=Thread))
    monkeypatch.setattr(adapter,'queue',SimpleNamespace(Queue=lambda **kwargs:incoming,Empty=queue.Empty))
    result=adapter.worker('unused.invalid',receive_only=receive_only,ack_clock_factory=AckReceiptClock,
        validate_update=validate_spatial_target,channel_factory=Channel,input_stream=iter(()),output_stream=output,clock=clock)
    return result,[json.loads(line) for line in output.getvalue().splitlines()],calls


def test_worker_preserves_first_ack_across_newer_outputs_and_parent_envelope(monkeypatch):
    pending=dict(update=[1,0,123,[0.,0.,.5]],deadline=10.15)
    result,rows,calls=worker_frames(monkeypatch,[(1.,0,10.),(1.02,1,10.04),(1.04,1,10.14),(1.06,2,10.16)],pending=pending)
    assert result==0
    assert [row['feedback']['ack_sampled_at'] for row in rows]==[10.,10.04,10.04,10.16]
    assert [row['feedback']['worker_sampled_at'] for row in rows]==[10.,10.04,10.14,10.16]
    assert calls==[(pending['update'],10.15),'closed']


def test_worker_repeated_cached_timestamp_cannot_refresh_150ms_guard(monkeypatch):
    result,rows,calls=worker_frames(monkeypatch,[(1.,0,10.),(1.,0,10.1),(1.,0,10.151)])
    assert result==1 and len(rows)==2 and 'stale' in rows[-1]['error']
    assert calls==['closed']


def test_worker_clock_reset_fails_and_closes_without_replay(monkeypatch):
    result,rows,calls=worker_frames(monkeypatch,[(1.,0,10.),(.98,0,10.02)])
    assert result==1 and 'clock reset' in rows[-1]['error'] and calls==['closed']


def test_read_only_worker_emits_joint_speed_without_any_target(monkeypatch):
    result,rows,calls=worker_frames(monkeypatch,[(1.,0,10.),(1.02,0,10.02)],receive_only=True)
    assert result==0 and all(row['feedback']['joint_speed']==[0.]*6 for row in rows)
    assert calls==['closed']


def test_actual_command_reader_rejects_read_only_input_before_any_send():
    adapter=module();calls=[];output=io.StringIO()
    class Channel:
        def __init__(self,*args,**kwargs):calls.append('opened')
        def poll(self,**kwargs):return []
        def send(self,*args):pytest.fail('Read-only worker sent a target')
        def close(self):calls.append('closed')
    result=adapter.worker('unused.invalid',receive_only=True,ack_clock_factory=AckReceiptClock,
        validate_update=validate_spatial_target,channel_factory=Channel,
        input_stream=iter([json.dumps(dict(update=[1,0,123,[0,0,0]],deadline=10.15))+'\n']),output_stream=output)
    assert result==1 and 'Read-only' in json.loads(output.getvalue())['error'] and calls==['opened','closed']
