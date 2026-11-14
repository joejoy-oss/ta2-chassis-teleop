import time, rclpy
class Pub:
    def publish(s,m):
        if not rclpy.OK[0]: raise RuntimeError("context invalid")
        rclpy.R.target=list(m.data); rclpy.R.sent.append(list(m.data))
        import os,sys
        if os.environ.get('FAKE_PUBLOG'): print(f'PUB {time.monotonic():.2f} {list(m.data)}', file=sys.stderr, flush=True)
        if any(m.data): rclpy.R.last_cmd=time.monotonic()
        else: rclpy.R.last_cmd=None
    def get_subscription_count(s): return 1
class Node:
    def __init__(s,name): s._subs={}
    def create_publisher(s,*a): return Pub()
    def create_subscription(s,t,topic,cb,q): s._subs.setdefault(topic,[]).append(cb)
    def destroy_node(s): pass
