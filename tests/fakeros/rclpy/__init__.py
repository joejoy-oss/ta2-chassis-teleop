"""Fake rclpy for offline tests: a simulated TA2 chassis.

Publishes /chassis/joint_states (200 Hz, wrapped motor positions, Gaussian speed noise) and
/api/current_mode (1 Hz); follows /api/chassis/velocity with the real robot's acceleration
limits (0.3 up / 0.6 down) and the motor matrix measured on the robot on 2026-11-14.
Behaviour is tuned with the SC environment variable (JSON), e.g.
  {"noise": 0.08, "no_autostop": 1, "lost_after": 4.0, "mode": {...}, "no_mode": 1,
   "names": [...], "sigint_at": 4.0}
"""
import os, time, math, json, signal
SC = json.loads(os.environ.get("SC", "{}"))
OK = [True]
def init(**kw): OK[0] = True
def shutdown(): OK[0] = False
try_shutdown = shutdown
class Robot:
    def __init__(s):
        s.t0 = time.monotonic(); s.v = [0.0,0.0,0.0]; s.target=[0.0,0.0,0.0]; s.last_cmd=None
        s.p=[3.0,3.0,3.0]; s.last=time.monotonic(); s.next_js=0; s.next_mode=0; s.sent=[]
    def step(s):
        now=time.monotonic(); dt=now-s.last; s.last=now; el=now-s.t0
        if s.last_cmd is not None and now-s.last_cmd>1.0 and not SC.get("no_autostop"): s.target=[0,0,0]
        for i in range(3):
            d=s.target[i]-s.v[i]; lim=(0.3 if abs(s.target[i])>abs(s.v[i]) else 0.6)*dt
            s.v[i]+=max(-lim,min(lim,d))
        J=[[-5.1183,8.68,4.1675],[-5.07,-8.595,4.2625],[10.0667,-0.07,4.18]]
        w=[sum(J[r][c]*s.v[c] for c in range(3)) for r in range(3)]
        for i in range(3):
            s.p[i]+=w[i]*dt; s.p[i]=(s.p[i]+math.pi)%(2*math.pi)-math.pi
        return el,w
R=Robot()


class Arms:
    """Two 7-joint arms with an FSM: PAUSE(0) -> SLOW_START(1, 0.5 s) -> READY(2) while fsm/enable 1.0
    arrives at least every 1 s. In READY each joint follows its joint_cmd at <= 1 rad/s.
    SC "block": {"arm": "right", "joint": 3, "at": -0.6} stops that joint there (simulated contact)."""
    def __init__(s):
        s.q={"left":[-0.05,1.57,0.13,0.25,-0.23,-0.02,-0.06],"right":[0.13,-1.55,-0.08,-0.28,-0.04,-0.02,0.11]}
        s.q0={k:list(v) for k,v in s.q.items()}
        s.target={k:list(v) for k,v in s.q.items()}; s.fsm=0; s.hb=-1e9; s.ss=None; s.last=time.monotonic(); s.pauses=0
    def cmd(s, topic, m):
        if topic=="/api/fsm/enable":
            if m.data>=0.5:
                s.hb=time.monotonic()
                if s.fsm==0: s.fsm=1; s.ss=time.monotonic()
            else:
                s.fsm=0; s.pauses+=1
        elif topic.endswith("_arm/joint_cmd"):
            arm=topic.split("/")[2].split("_")[0]
            mode=SC.get("mode",{})
            if mode.get(f"{arm}_arm_control_mode",1)==1: s.target[arm]=list(m.position)
    def step(s):
        now=time.monotonic(); dt=now-s.last; s.last=now
        if s.fsm and now-s.hb>1.0: s.fsm=0; s.pauses+=1
        if s.fsm==1 and now-s.ss>0.5: s.fsm=2
        if s.fsm==2:
            for arm in s.q:
                for i in range(7):
                    d=s.target[arm][i]-s.q[arm][i]; s.q[arm][i]+=max(-dt,min(dt,d))
            b=SC.get("block")
            if b:   # the joint cannot pass b["at"] coming from its starting side
                a,j,at=b["arm"],b["joint"],b["at"]
                side=s.side if hasattr(s,"side") else None
                if side is None: s.side=side=(1 if s.q0[a][j]>at else -1)
                if (s.q[a][j]-at)*side<0: s.q[a][j]=at


ARM=Arms()
NODE=[None]
def spin_once(node, timeout_sec=0):
    if not OK[0]: raise RuntimeError("context invalid")
    time.sleep(timeout_sec)
    el,w=R.step(); ARM.step(); now=time.monotonic()
    if SC.get("sigint_at") and el>SC["sigint_at"] and not SC.get("_sig"):
        SC["_sig"]=1; os.kill(os.getpid(), signal.SIGINT)
    lost = SC.get("lost_after") is not None and el>SC["lost_after"]
    if now>=R.next_js and not lost:
        R.next_js=now+0.005
        from sensor_msgs.msg import JointState
        m=JointState(); m.name=SC.get("names",["chassis_joint1","chassis_joint2","chassis_joint3"])
        noise=SC.get("noise",0.02)
        import random; m.velocity=[x+random.gauss(0,noise) for i,x in enumerate(w)]; m.position=list(R.p)
        for cb in node._subs.get("/chassis/joint_states",[]): cb(m)
        for arm in ("left","right"):
            if node._subs.get(f"/{arm}_arm/joint_states"):
                a=JointState(); a.name=[f"{arm[0]}_joint{i+1}" for i in range(7)]; a.position=list(ARM.q[arm]); a.velocity=[0.0]*7
                for cb in node._subs[f"/{arm}_arm/joint_states"]: cb(a)
        if node._subs.get("/fsm_state"):
            from std_msgs.msg import Int32
            for cb in node._subs["/fsm_state"]: cb(Int32(ARM.fsm))
    if now>=R.next_mode:
        R.next_mode=now+1.0
        from std_msgs.msg import String
        m=String(); m.data=json.dumps(SC.get("mode",{"meta_mode":1,"enable_chassis":True}))
        if not SC.get("no_mode"):
            for cb in node._subs.get("/api/current_mode",[]): cb(m)
