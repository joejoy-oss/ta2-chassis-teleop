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
NODE=[None]
def spin_once(node, timeout_sec=0):
    if not OK[0]: raise RuntimeError("context invalid")
    time.sleep(timeout_sec)
    el,w=R.step(); now=time.monotonic()
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
    if now>=R.next_mode:
        R.next_mode=now+1.0
        from std_msgs.msg import String
        m=String(); m.data=json.dumps(SC.get("mode",{"meta_mode":1,"enable_chassis":True}))
        if not SC.get("no_mode"):
            for cb in node._subs.get("/api/current_mode",[]): cb(m)
