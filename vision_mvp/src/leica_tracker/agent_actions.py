"""仅接受经过验证的少量动作类型。模型输出不能直接作为机器人程序执行。"""
from dataclasses import dataclass
import math

from .gemini_framing import FramingError, _numbers
from .robot_gestures import GesturePlan

DIRECTIONS={'MOVE_LEFT':('horizontal',-1),'MOVE_RIGHT':('horizontal',1),
            'MOVE_UP':('vertical',-1),'MOVE_DOWN':('vertical',1),
            'MOVE_CLOSER':('depth',1),'MOVE_FARTHER':('depth',-1)}
CUES=tuple(DIRECTIONS)+('INVITE','ACKNOWLEDGE','WAIT')
INTENTS=('invite','guide','ask','wait','confirm','finish')
STYLES=('close_up','half_body','full_body','environmental','none')


def agent_gesture(cue,amplitude,repetitions,cycle_ms):
    if (cue not in CUES or type(amplitude) not in (float,int) or
        not math.isfinite(amplitude) or not .2<=amplitude<=.8 or
        type(repetitions) is not int or not 1<=repetitions<=3 or
        type(cycle_ms) is not int or not 2000<=cycle_ms<=4000):
        raise ValueError('Invalid bounded agent gesture')
    if cue=='WAIT':return None
    axis,direction=DIRECTIONS.get(cue,('vertical',1) if cue=='ACKNOWLEDGE' else ('horizontal',1))
    pattern=('agent_invite' if cue=='INVITE' else 'agent_ack' if cue=='ACKNOWLEDGE' else 'agent_sweep')
    return GesturePlan(cue,pattern,'robot_arm',axis,direction,repetitions,
                       repetitions*cycle_ms,amplitude=float(amplitude),profile='agent-coupled')


@dataclass(frozen=True)
class AgentTurn:
    intent:str
    speech:str
    cue:str
    amplitude:float
    repetitions:int
    cycle_ms:int
    pause_ms:int
    wait_for_reply:bool
    subject_visible:bool
    confidence:float
    frame:tuple|None
    style:str
    target:tuple|None=None

    @property
    def gesture(self):
        if not self.subject_visible or self.confidence<.6:return None
        return agent_gesture(self.cue,self.amplitude,self.repetitions,self.cycle_ms)


SCHEMA={'type':'object','additionalProperties':False,'properties':{
    'intent':{'type':'string','enum':list(INTENTS)},
    'speech':{'type':'string','description':'One short English sentence, preferably at most 14 words; at most 240 characters.'},
    'cue':{'type':'string','enum':list(CUES)},
    'amplitude':{'type':'number','description':'Normalized gesture size, .20 to .80.'},
    'repetitions':{'type':'integer','description':'1 to 3 repetitions.'},
    'cycle_ms':{'type':'integer','description':'2000 to 4000 milliseconds per cycle.'},
    'pause_ms':{'type':'integer','description':'1000 to 8000 milliseconds to observe after the gesture.'},
    'wait_for_reply':{'type':'boolean'},'subject_visible':{'type':'boolean'},
    'confidence':{'type':'number','description':'0 to 1 confidence in this decision.'},
    'frame':{'anyOf':[{'type':'null'},{'type':'array','items':{'type':'number'}}],
             'description':'Null or four normalized [left,top,right,bottom] crop bounds within the existing image.'},
    'style':{'type':'string','enum':list(STYLES)},
    'target':{'anyOf':[{'type':'null'},{'type':'array','items':{'type':'number'}}],
              'description':'Optional proposed person goal [center x, head y, shoulder width] in the FULL mirrored image, all in 0..1 and width > 0. Use null when no measured goal is justified. This is image geometry, never robot coordinates.'}}}
SCHEMA['required']=list(SCHEMA['properties'])


def validate_turn(data):
    try:
        if isinstance(data,dict) and 'target' not in data:data={**data,'target':None}
        if not isinstance(data,dict) or set(data)!=set(SCHEMA['required']):raise ValueError()
        if data['intent'] not in INTENTS or data['style'] not in STYLES:raise ValueError()
        speech=data['speech']
        if not isinstance(speech,str) or not 1<=len(speech.strip())<=240:raise ValueError()
        if any(ord(c)<32 and c not in '\n\t' for c in speech):raise ValueError()
        for key in ('wait_for_reply','subject_visible'):
            if type(data[key]) is not bool:raise ValueError()
        confidence=data['confidence']
        if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError()
        if type(data['pause_ms']) is not int or not 1000<=data['pause_ms']<=8000:raise ValueError()
        agent_gesture(data['cue'],data['amplitude'],data['repetitions'],data['cycle_ms'])
        if data['intent']=='finish' and data['cue'] not in ('WAIT','ACKNOWLEDGE'):raise ValueError()
        frame=data['frame']
        if frame is not None:
            frame=_numbers(frame,4)
            if frame[2]-frame[0]<.08 or frame[3]-frame[1]<.08:raise ValueError()
        target=data['target']
        if target is not None:
            target=_numbers(target,3)
            if target[2]<=0:raise ValueError()
        return AgentTurn(**{**data,'speech':speech.strip(),'frame':frame,'target':target})
    except (ValueError,TypeError,KeyError):raise FramingError('AI_INVALID_ACTION') from None
