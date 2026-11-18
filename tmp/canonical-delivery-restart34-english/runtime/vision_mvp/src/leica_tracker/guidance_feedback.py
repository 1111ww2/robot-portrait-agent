"""界面必须区分暂停和正常完成。"""
import cv2
import re
from .person_guidance import GuidanceCue

GREEN = (80, 230, 150)
AMBER = (40, 170, 255)
BLUE = (255, 195, 100)


def position_checks(decision, *, mirrored=True):
    point=getattr(decision,'filtered_point',None) or decision.current_point
    raw=decision.current_point
    tolerances=getattr(decision,'target_tolerances',(.02,.02,.03))
    spans=getattr(decision,'error_spans',(.25,.20,.15))
    rows=[]
    for i,label in enumerate(('CENTER','HEAD HEIGHT','DISTANCE')):
        row=dict(label=label,target=decision.target[i],tolerance=tolerances[i],
                 value=None,state='UNAVAILABLE',action='ENTER VIEW',magnitude=0.)
        if decision.tracking and point is not None and raw is not None:
            error=point[i]-decision.target[i]
            row['value']=point[i]
            if abs(error)<=tolerances[i]:
                row['state']='OK' if abs(raw[i]-decision.target[i])<=tolerances[i] else 'CHECKING'
                row['action']='ALIGNED' if row['state']=='OK' else 'HOLD STILL'
            else:
                row['state']='ADJUST';row['magnitude']=abs(error)/spans[i]
                if i==0:
                    row['action']='MOVE LEFT' if (error>0)==mirrored else 'MOVE RIGHT'
                elif i==1:
                    row['action']='MOVE UP' if error>0 else 'MOVE DOWN'
                else:
                    row['action']='STEP CLOSER' if error<0 else 'STEP BACK'
        rows.append(row)
    return rows


def feedback_text(decision, *, mirrored=True):
    phase = decision.phase
    complete = phase == "ARRIVED"
    coarse=getattr(decision,'interaction_mode','precise')=='coarse-demo'
    frame_guide=getattr(decision,'interaction_mode','precise')=='frame-guide'
    ai=getattr(decision,'interaction_mode','precise') in ('gemini-guide','claude-guide')
    if ai and phase.startswith('AI_'):
        messages={
            'AI_ENTRY_ACK':('I SEE YOU','The arm acknowledges you now; AI plans in the background.'),
            'AI_THINKING':('AI IS CHOOSING A FRAME','You are acquired. The next direction is planned in the background.'),
            'AI_REFRAMING':('ADJUSTING FRAME TO YOUR DISTANCE','Adjusting locally. The scene target stays locked; no new AI request.'),
            'AI_SERVICE_BUSY':('AI SERVICE BUSY','Arm paused. Retrying after a short wait.'),
            'AI_RATE_LIMIT':('AI RATE LIMIT','The provider returned a rate limit. Waiting before retrying.'),
            'AI_AUTH_ERROR':('AI KEY NOT ACCEPTED','Update the local key, then press R to retry.'),
            'AI_KEY_MISSING':('AI KEY REQUIRED','Configure the local API key. Press R to retry.'),
            'AI_MODEL_UNAVAILABLE':('AI MODEL UNAVAILABLE','Check model access in the provider console. R retries.'),
            'AI_BUDGET_REACHED':('LOCAL DEMO LIMIT REACHED','This app reached its request cap. Operator restart required.'),
            'AI_WAITING_FOR_STILLNESS':('CAPTURING A CLEARER VIEW','Choosing a useful retry frame in the background.'),
            'AI_COOLDOWN':('PAUSING BETWEEN MODEL REQUESTS','Camera stays live. Avoiding repeated uploads.'),
            'AI_REPLAN':('SCENE CHANGED - REPLANNING','Keep your head and shoulders in view.'),
            'AI_STALE_RESULT':('YOU MOVED WHILE AI PLANNED','Stand comfortably still. A fresh plan will be requested.'),
            'AI_NO_INTERACTION_TARGET':('CHOOSING A CLEARER INVITATION','The reply proposed no clear move. Retrying shortly.'),
            'AI_INVALID_PLAN_FIELDS':('AI REPLY MISSING FIELDS','Arm paused. Retrying a complete plan shortly.'),
            'AI_INVALID_COORDINATES':('AI COORDINATES INVALID','Arm paused. Retrying a valid plan shortly.'),
            'AI_INVALID_GEOMETRY':('AI FRAME DOES NOT FIT','Arm paused. Retrying a feasible frame shortly.'),
            'AI_INVALID_STYLE':('AI STYLE NOT RECOGNIZED','Arm paused. Retrying a valid plan shortly.'),
            'AI_INVALID_NOTE':('AI EXPLANATION MISSING','Arm paused. Retrying a complete plan shortly.'),
            'AI_NETWORK_ERROR':('AI CONNECTION FAILED','Arm paused. Retrying after a short wait.'),
        }
        title,detail=messages.get(phase,('AI PLAN NOT READY','Arm paused. Waiting to retry; R retries now.'))
        color=BLUE if phase=='AI_THINKING' else AMBER
    elif phase=='DEMO_COMPLETE':
        title,detail,color='DEMO COMPLETE','Framing/capture were simulated. R restarts.',GREEN
    elif phase=='POSITION_READY':
        title,detail,color='YOUR POSITION IS READY','Stand still. The arm will adjust next.',GREEN
    elif phase=='ARM_FRAMING':
        title,detail,color='ARM IS FRAMING  SIM','Hold still. This adjustment is simulated.',BLUE
    elif phase=='SIMULATED_CAPTURE':
        title,detail,color='SIMULATED CAPTURE','No photograph is saved in this test.',BLUE
    elif phase=='COMPLETING':
        title,detail,color='ENDING THE SESSION','The simulated capture step is complete.',BLUE
    elif phase=='REPOSITION':
        title,detail,color='POSITION CHANGED','Arm stopped. Settle in the photo area again.',AMBER
    elif phase=='COARSE_SETTLING':
        title,detail,color='STAND STILL IN THE AREA','Your position is enough. No small point to chase.',GREEN
    elif complete and frame_guide:
        title,detail,color='IN POSITION','Accepted range reached. Arm confirms and ends. R restarts.',GREEN
    elif complete:
        title, detail, color = "IN POSITION", "Guidance complete. Press R to start again.", GREEN
    elif not decision.tracking and decision.reason == 'SUBJECT_TOO_SMALL':
        title, detail, color = "STEP INTO VIEW", "Your head and shoulders need to be clearly visible.", AMBER
    elif not decision.tracking:
        title, detail, color = "PERSON NOT VISIBLE", "Show your head and both shoulders.", AMBER
    elif phase == "SETTLING":
        title, detail, color = "HOLD STILL...", ('Inside the accepted range. Hold for one second.' if frame_guide else "Checking your position for one second."), GREEN
    elif phase in ("WAITING_FOR_RESPONSE", "PAUSED_FOR_PARTICIPANT"):
        checks=position_checks(decision,mirrored=mirrored)
        action=max(checks,key=lambda row:row['magnitude'])['action']
        title, detail, color = ('ENTER PHOTO AREA' if coarse else 'NOT IN POSITION'), f"Next: {action}. The arm is waiting for you.", BLUE
    elif phase == "INVITING":
        title, detail, color = "READY TO GUIDE YOU", "Follow the arm toward the green target.", BLUE
    else:
        title = decision.cue.value.replace("_", " ")
        detail, color = "Swing size is a rough correction, not a step distance.", BLUE
    if ai:
        claude=getattr(decision,'interaction_mode','')=='claude-guide'
        brand='Claude' if claude else 'Gemini'
        title=re.sub(r'\bAI\b',brand.upper(),title)
        detail=re.sub(r'\bAI\b',brand,detail)
        detail=detail.replace('The provider returned','Anthropic returned' if claude else 'Google returned')
    return title,detail,color


def draw_position_checks(frame, decision, *, mirrored=True):
    left,right,top=18,242,88
    cv2.rectangle(frame,(left,top),(right,388),(18,22,24),-1)
    coarse=getattr(decision,'interaction_mode','precise')=='coarse-demo'
    ai=getattr(decision,'interaction_mode','precise') in ('gemini-guide','claude-guide')
    frame_guide=getattr(decision,'interaction_mode','precise') in ('frame-guide','gemini-guide','claude-guide')
    if ai and decision.phase=='AI_REFRAMING':
        for i,text in enumerate(('FRAME UPDATING','Person size changed.','Near: larger area.','Far: smaller area.','Choosing a new frame.')):
            cv2.putText(frame,text,(left+10,top+30+i*28),cv2.FONT_HERSHEY_SIMPLEX,
                        .42,BLUE if i==0 else (230,230,230),1,cv2.LINE_AA)
        return
    if ai and not getattr(decision,'composition_frame',None):
        active=decision.tracking and decision.cue in (GuidanceCue.INVITE,GuidanceCue.WAIT)
        lines=(('PERSON ACQUIRED','Arm responds now.','AI plans the','next direction.','R = replan / retry') if active else
               ('AI COMPOSITION','No target accepted.','No fixed fallback.','Arm is paused.','R = replan / retry'))
        for i,text in enumerate(lines):
            cv2.putText(frame,text,(left+10,top+30+i*28),cv2.FONT_HERSHEY_SIMPLEX,
                        .42,BLUE if i==0 else (230,230,230),1,cv2.LINE_AA)
        return
    if coarse and decision.phase in ('POSITION_READY','ARM_FRAMING','SIMULATED_CAPTURE','COMPLETING','DEMO_COMPLETE'):
        for i,text in enumerate(('PERSON READY','Hold a comfortable','position.','Arm handles the','next simulated steps.')):
            cv2.putText(frame,text,(left+10,top+30+i*28),cv2.FONT_HERSHEY_SIMPLEX,
                        .43,GREEN if i==0 else (230,230,230),1,cv2.LINE_AA)
        return
    if decision.phase=='ARRIVED':
        for i,text in enumerate(('SESSION COMPLETE','Target reached','in this round.','R = start again')):
            cv2.putText(frame,text,(left+10,top+30+i*28),cv2.FONT_HERSHEY_SIMPLEX,
                        .43,GREEN,1,cv2.LINE_AA)
        return
    checks=position_checks(decision,mirrored=mirrored)
    count=sum(row['state']=='OK' for row in checks)
    cv2.putText(frame,f'{"PHOTO AREA" if coarse else "POSITION CHECK"} {count}/3',(left+10,top+24),
                cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(frame,'IN RANGE + HOLD 1 s' if frame_guide else 'IN AREA + STAND STILL' if coarse else 'ALL 3 OK + HOLD 1 s',(left+10,top+44),
                cv2.FONT_HERSHEY_SIMPLEX,.37,(210,210,210),1,cv2.LINE_AA)
    for i,row in enumerate(checks):
        y=top+72+i*77
        color=GREEN if row['state']=='OK' else BLUE if row['state']=='CHECKING' else AMBER
        cv2.putText(frame,f'{i+1} {row["label"]}',(left+10,y),
                    cv2.FONT_HERSHEY_SIMPLEX,.4,(235,235,235),1,cv2.LINE_AA)
        action='IN RANGE' if (coarse or frame_guide) and row['state']=='OK' else row['action']
        cv2.putText(frame,action,(left+10,y+23),
                    cv2.FONT_HERSHEY_SIMPLEX,.50,color,1,cv2.LINE_AA)
        value='--' if row['value'] is None else f'{100*row["value"]:.0f}%'
        text=(f'{value} / range {100*(row["target"]-row["tolerance"]):.0f}-{100*(row["target"]+row["tolerance"]):.0f}%'
              if coarse or frame_guide else f'{value} / goal {100*row["target"]:.0f}% +/-{100*row["tolerance"]:.0f}%')
        cv2.putText(frame,text,(left+10,y+43),
                    cv2.FONT_HERSHEY_SIMPLEX,.33,(210,210,210),1,cv2.LINE_AA)


def draw_guidance_feedback(frame, decision, *, mirrored=True):
    """将目标和头部标记显示在醒目的状态卡片上方。"""
    title,detail,color=feedback_text(decision,mirrored=mirrored)
    phase=decision.phase
    complete=phase in ('ARRIVED','DEMO_COMPLETE')

    height, width = frame.shape[:2]
    card_width = min(620, width-40)
    card_height = 112
    left = (width-card_width)//2
    top = max(12, min(int(height*.64), height-card_height-114))
    right, bottom = left+card_width, top+card_height
    cv2.rectangle(frame,(left,top),(right,bottom),(20,26,22),-1)
    cv2.rectangle(frame,(left,top),(right,bottom),color,2)
    if complete:
        cv2.rectangle(frame,(8,8),(width-9,height-9),GREEN,8)
        center=(left+40,top+43)
        cv2.circle(frame,center,24,GREEN,2,cv2.LINE_AA)
        cv2.line(frame,(center[0]-12,center[1]),(center[0]-3,center[1]+10),GREEN,4,cv2.LINE_AA)
        cv2.line(frame,(center[0]-3,center[1]+10),(center[0]+14,center[1]-12),GREEN,4,cv2.LINE_AA)
        cv2.putText(frame,title,(left+82,top+57),cv2.FONT_HERSHEY_SIMPLEX,1.05,GREEN,2,cv2.LINE_AA)
        cv2.putText(frame,detail,(left+24,top+88),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
    else:
        cv2.putText(frame,title,(left+24,top+40),cv2.FONT_HERSHEY_SIMPLEX,.78,color,1,cv2.LINE_AA)
        cv2.putText(frame,detail,(left+24,top+68),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
        if phase in ('SETTLING','COARSE_SETTLING','ARM_FRAMING','SIMULATED_CAPTURE'):
            progress=max(0.,min(1.,getattr(decision,'arrival_progress',0.) if phase in ('SETTLING','COARSE_SETTLING') else decision.stage_progress))
            bar_left,bar_right=left+24,right-24
            bar_top=bottom-18
            cv2.rectangle(frame,(bar_left,bar_top),(bar_right,bar_top+6),(60,60,60),-1)
            if progress > 0.:
                cv2.rectangle(frame,(bar_left,bar_top),
                              (bar_left+int((bar_right-bar_left)*progress),bar_top+6),GREEN,-1)
    draw_position_checks(frame,decision,mirrored=mirrored)
