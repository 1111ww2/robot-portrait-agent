"""监测访客控制变化，遇到故障即禁止继续，不发送机器人命令。由显式启动的实机试验监管器使用。
收到停止或重启、服务被替换或心跳丢失时锁定状态；应用仍须检查遥测及常驻看门狗。此功能不代替急停。"""
import math


class ControlGuard:
    def __init__(self):
        self.identity=None;self.stamp=None;self.failure=None

    def _fail(self,reason):
        self.failure=reason
        raise RuntimeError(reason)

    def observe(self,control,now):
        if self.failure:raise RuntimeError(self.failure)
        if (not isinstance(control,dict) or not isinstance(control.get('instance'),str)
                or not 1<=len(control['instance'])<=128 or type(control.get('epoch')) is not int
                or not 0<=control['epoch']<2**53 or not math.isfinite(now)):
            self._fail('Invalid guest control state')
        identity=(control['instance'],control['epoch'])
        if self.identity is not None:
            self.check(now)
            if identity!=self.identity:self._fail('Guest Stop/restart or service change; explicit restart required')
        self.identity=identity;self.stamp=now

    def check(self,now):
        if self.failure:raise RuntimeError(self.failure)
        if self.stamp is None or not math.isfinite(now) or not 0<=now-self.stamp<=.35:
            self._fail('Guest control heartbeat missing; explicit restart required')
