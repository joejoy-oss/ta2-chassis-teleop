from pathlib import Path
import datetime, hashlib, json, math, os, re, shutil, signal, subprocess, tempfile, time, urllib.request
E=Path(__file__).parent
R=Path('/root/ta2-chassis-teleop')
MODEL='grasp_v3_step20000_rtc'
SERVER='grasp-v3-step20000-policy'
BASE='http://127.0.0.1:8088'
def get(path):
    return json.load(urllib.request.urlopen(BASE+path,timeout=5))
def idle():
    s=get('/status')
    assert not any(s.get(k) for k in ['moving','arm','vla','estop','seq','route']), 'Robot busy'
    assert not s.get('mission',{}).get('active'), 'Mission active'
    host=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    client=subprocess.check_output(['docker','exec','openpi-xpu-client','ps','-eo','pid,args'],text=True)
    active=[line for line in (host+'\n'+client).splitlines() if re.search(r'(?:^|\s)(?:[^\s]*/)?(?:guarded_grasp\.py|arm_pose\.py|vla_run\.sh)(?:\s|$)',line)]
    assert not active, 'External robot process active: '+str(active)
    return s
status=json.loads((E/'deployment_status.json').read_text())
assert status['stage']=='ready_for_web', status
assert json.loads((E.parent.parent/'grasp_v3_step20000/.download_status.json').read_text())['state']=='complete'
smoke=json.loads((E/'step20000_smoke.json').read_text())
for row in smoke['rows']:
    for key in ['first_deg','mae30_deg']:
        assert math.isfinite(row[key]) and row[key]>=0
assert subprocess.check_output(['docker','inspect','-f','{{.State.Running}}',SERVER],text=True).strip()=='true'
# Check the actual new wrappers with a fake final runner; never invoke robot code.
with tempfile.TemporaryDirectory(prefix='step20000-wrapper-') as tmp:
    tmp=Path(tmp); fake=tmp/'runner.py'
    fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\nprint(json.dumps({"args":sys.argv[1:],"env":{k:os.environ.get(k) for k in ["PORT","PROMPT","RESET","RTC"]}},ensure_ascii=False))\n')
    fake.chmod(0o755)
    base=tmp/'base.sh';base.write_text((R/'vla/vla_grasp_v3_step20000.sh').read_text().replace('/root/vla_run.sh',str(fake)));base.chmod(0o755)
    rtc=tmp/'rtc.sh';rtc.write_text((R/'vla/vla_grasp_v3_step20000_rtc.sh').read_text().replace('/root/vla_grasp_v3_step20000.sh',str(base)));rtc.chmod(0o755)
    result=json.loads(subprocess.check_output([str(rtc),'deployment_check','60','1.0'],text=True))
    assert result=={'args':['deployment_check','60','1.0'],'env':{'PORT':'8005','PROMPT':'抓取红色沙包','RESET':'1','RTC':'1'}},result
    (E/'wrapper_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
config=json.loads((R/'vla_models.json').read_text())
expected={MODEL,'grasp_v3_step13000_rtc','grasp_v3_step10000_rtc','sandbag_v2_step4999_rtc'}
assert {m['id'] for m in config['models']}==expected
assert config['default']==MODEL
assert {m['id'] for m in json.loads(Path('/root/vla_models.json').read_text())['models']}=={'grasp_v3_step5000_rtc','grasp_v3_step13000_rtc','grasp_v3_step10000_rtc','sandbag_v2_step4999_rtc'},'Runtime registry changed; review concurrent changes'
idle()
b=Path('/root/web-v3-backups')/('step20000_cutover_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));b.mkdir()
for name in ['vla_models.json','ta2_vla_selected.json']:
    src=Path('/root')/name
    if src.exists():shutil.copy2(src,b/name)
# Preserve the current web process environment and code.
listener=subprocess.check_output(['ss','-ltnpH','sport','=',':8088'],text=True)
pids=set(re.findall(r'pid=(\d+)',listener));assert len(pids)==1,listener
pid=int(next(iter(pids)));env=dict(entry.split('=',1) for entry in Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in entry)
assert not any(env.get(k) for k in ['VLA_CMD','POLICY_CONTAINER','MISSION_PORT','POLICY_PORT']), 'Policy environment override requires review'
(b/'web_move.sha256').write_text(hashlib.sha256(Path('/root/web_move.py').read_bytes()).hexdigest())
def start_web():
    with Path('/root/web_move.log').open('a') as log:
        p=subprocess.Popen(['python3','/root/web_move.py'],cwd='/root',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(30):
        if p.poll() is not None:raise RuntimeError('Web startup exited')
        try:return get('/api/models')
        except Exception:time.sleep(.5)
    raise TimeoutError('Web startup timeout')
def stop_web(pid):
    os.kill(pid,signal.SIGTERM)
    for _ in range(30):
        if not subprocess.check_output(['ss','-ltnH','sport','=',':8088'],text=True).strip():return
        time.sleep(.2)
    raise TimeoutError('Old web listener did not exit')
try:
    for suffix in ['','_rtc']:
        src=R/f'vla/vla_grasp_v3_step20000{suffix}.sh';dst=Path('/root')/src.name
        assert not dst.exists();shutil.copy2(src,dst)
    idle()
    shutil.copy2(R/'vla_models.json','/root/vla_models.json')
    stop_web(pid)
    models=start_web()
    assert {m['id'] for m in models['models']}==expected,models
    assert all(m['running'] for m in models['models']),models
    request=urllib.request.Request(BASE+'/models/select/'+MODEL,data=b'',method='POST')
    with urllib.request.urlopen(request,timeout=5) as response:selection=json.load(response)
    for _ in range(20):
        s=idle()
        if s.get('connected'):break
        time.sleep(.5)
    assert s['vla_selected']==MODEL and s.get('connected'),s
except BaseException:
    shutil.copy2(b/'vla_models.json','/root/vla_models.json')
    if (b/'ta2_vla_selected.json').exists():shutil.copy2(b/'ta2_vla_selected.json','/root/ta2_vla_selected.json')
    current=subprocess.check_output(['ss','-ltnpH','sport','=',':8088'],text=True)
    for found in set(re.findall(r'pid=(\d+)',current)):stop_web(int(found))
    start_web()
    raise
# The old service is now absent from the registry; leave its container and weights for rollback.
idle()
connections=subprocess.check_output(['ss','-tnH','state','established','sport','=',':8002'],text=True).strip()
assert not connections,'Old policy still has a connected client; leave running and review'
subprocess.run(['docker','stop','-t','15','grasp-v3-step5000-policy'],check=True,timeout=25)
assert subprocess.check_output(['docker','inspect','-f','{{.State.Running}}','grasp-v3-step5000-policy'],text=True).strip()=='false'
status.update(stage='deployed',selected=MODEL,training_steps=20000,source_checkpoint_index=19999,backup=str(b),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_v3_step5000_stopped=True,no_robot_motion=True)
(E/'deployment_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
(E/'web_models_after.json').write_text(json.dumps(get('/api/models'),ensure_ascii=False,indent=2))
(E/'web_status_after.json').write_text(json.dumps(get('/status'),ensure_ascii=False,indent=2))
print(json.dumps(status,ensure_ascii=False))
