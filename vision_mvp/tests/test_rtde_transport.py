import struct
import pytest

from leica_tracker.rtde_transport import PacketReader, input_packet, parse_feedback, require_recipe


class FragmentedSocket:
    def __init__(self, data):
        self.data = data
    def recv(self, size):
        chunk, self.data = self.data[:min(size, 2)], self.data[min(size, 2):]
        return chunk


def test_fragmented_packet_is_reassembled_and_eof_rejected():
    reader = PacketReader(FragmentedSocket(struct.pack('!HB', 5, 86) + b'\0\1'))
    assert reader.read() == (86, b'\0\1')
    with pytest.raises(EOFError):
        reader.read()


def test_oversize_or_truncated_frame_is_rejected():
    for data in (struct.pack('!HB', 2, 85), struct.pack('!HB', 60000, 85), struct.pack('!HB', 5, 85)):
        with pytest.raises((ValueError, EOFError)):
            PacketReader(FragmentedSocket(data)).read()


def test_input_recipe_requires_exclusive_ownership_and_exact_types():
    assert require_recipe(b'\x01INT32,INT32,INT32,DOUBLE', 'INT32,INT32,INT32,DOUBLE') == 1
    with pytest.raises(RuntimeError):
        require_recipe(b'\x01IN_USE,INT32,INT32,DOUBLE', 'INT32,INT32,INT32,DOUBLE')


def test_input_packet_encodes_finite_target_with_control_and_session():
    data = input_packet(1, 7, 0, 1234, -.135)
    assert struct.unpack('!HB Biiid', data) == (24, 85, 1, 7, 0, 1234, -.135)
    with pytest.raises(ValueError):
        input_packet(1, 7, 1, 1234, float('nan'))


def test_feedback_rejects_wrong_recipe_and_returns_echo():
    body = struct.pack('!Bdiii6d', 1, 10.25, 0, 7, 1234, 0,0,0,0,0,0)
    assert parse_feedback(body, 1) == {'timestamp':10.25,'state':0,'sequence':7,'session':1234,'tcp_speed':[0]*6}
    with pytest.raises(ValueError):
        parse_feedback(body, 2)
