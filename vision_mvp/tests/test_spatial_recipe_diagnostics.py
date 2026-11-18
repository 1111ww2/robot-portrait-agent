"""手势确认拒绝在离线情况下进行分类，不会保留控制器文本。"""
import json

import pytest

from test_atomic_rtde_transport import INPUT_TYPES,OUTPUT_TYPES,Socket,channel,replies


PREFIX='Spatial RTDE recipe rejected, unavailable or already owned'


@pytest.mark.parametrize('readonly,phase,payload,reason,recipe_id,indices',[
    (False,'input',b'\x00INT32,IN_USE,INT32,DOUBLE,DOUBLE,DOUBLE','in_use',0,[1]),
    (False,'output',b'\x00DOUBLE,INT32,INT32,INT32,NOT_FOUND,VECTOR6D,VECTOR6D','missing',0,[4]),
    (True,'output',b'\x01'+OUTPUT_TYPES+b',NOT_FOUND','missing',1,[7]),
    (False,'output',b'','empty',None,[0,1,2,3,4,5,6]),
    (False,'input',b'\x00'+INPUT_TYPES,'zero',0,[]),
    (False,'input',b'\x02DOUBLE,INT32,INT32,DOUBLE,DOUBLE,DOUBLE','mismatch',2,[0]),
    (False,'input',b'\x02INT32,\xff,INT32,DOUBLE,DOUBLE,DOUBLE','malformed',2,[1]),
    (False,'input',b'\x02INT32','missing',2,[1,2,3,4,5]),
    (False,'input',b'\x02','malformed',2,[0,1,2,3,4,5]),
    (False,'input',b'\x02'+INPUT_TYPES+b',INT32','malformed',2,[6]),
])
def test_recipe_rejection_identifies_exact_handshake_phase_without_retry(
        readonly,phase,payload,reason,recipe_id,indices):
    sock=Socket(replies(readonly,**{phase if phase=='output' else 'inputs':payload}))
    with pytest.raises(RuntimeError,match=PREFIX) as failure:
        channel(readonly,sock=sock)
    message=str(failure.value)
    assert message.startswith(PREFIX+' {'),'Rejected recipes need bounded phase diagnostics'
    details=json.loads(message[len(PREFIX):])
    assert details['phase']==phase
    assert details['mode']==('receive_only' if readonly else 'writer')
    assert details['reason']==reason and details['recipe_id']==recipe_id
    assert details['mismatch_indices']==indices
    assert sock.closed
    assert [value[2] for value in sock.writes]==([86,79] if phase=='output' else [86,79,73])


def test_recipe_diagnostic_redacts_unknown_binary_and_oversized_controller_tokens():
    secret=b'private-controller-text-0123456789'
    for raw in [secret,secret*150,b'\xff\x00'+secret]:
        sock=Socket(replies(inputs=b'\x02INT32,'+raw+b',INT32,DOUBLE,DOUBLE,DOUBLE'))
        with pytest.raises(RuntimeError,match=PREFIX) as failure:
            channel(sock=sock)
        message=str(failure.value)
        assert message.startswith(PREFIX+' {'),'Rejected recipes need bounded phase diagnostics'
        details=json.loads(message[len(PREFIX):])
        assert details['reason']=='malformed' and details['tokens'][1]=='OTHER'
        assert 'private' not in message and len(message)<1536
        assert len(details['tokens'])<=16 and len(details['mismatch_indices'])<=16
        assert sock.closed and not any(value[2]==85 for value in sock.writes)


def test_many_untrusted_tokens_keep_only_bounded_allowlisted_diagnostics():
    sock=Socket(replies(inputs=b'\x02'+b'IN_USE,'*200+b'PRIVATE'))
    with pytest.raises(RuntimeError,match=PREFIX) as failure:
        channel(sock=sock)
    message=str(failure.value)
    assert message.startswith(PREFIX+' {'),'Rejected recipes need bounded phase diagnostics'
    details=json.loads(message[len(PREFIX):])
    assert details['reason']=='in_use' and details['token_count']==201
    assert details['tokens']==['IN_USE']*16 and len(details['mismatch_indices'])==16
    assert 'PRIVATE' not in message and len(message)<1536
