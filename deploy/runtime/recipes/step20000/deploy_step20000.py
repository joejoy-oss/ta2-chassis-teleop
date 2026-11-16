"""B70 checkpoint conversion and isolated serving; no robot or web motion commands."""
from pathlib import Path
import datetime
import json
import shutil
import subprocess
import time
import traceback
import urllib.request

ROOT = Path('/data/ta2/model_exports')
CKPT = ROOT / 'grasp_v3_step20000'
PT = ROOT / 'grasp_v3_step20000_pytorch'
E = ROOT / 'grasp_v3_eval' / 'step20000_deployment'
E.mkdir(parents=True, exist_ok=True)
START = time.time()
DEADLINE = START + 1800
SERVER = 'grasp-v3-step20000-policy'
CONVERTER = 'grasp-v3-step20000-convert'
status = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'checkpoint': str(CKPT), 'port': 8005, 'gpu': 1}
created_server = False

def save(stage, **kwargs):
    status.update(stage=stage, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kwargs)
    (E / 'deployment_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
    print(json.dumps(status, ensure_ascii=False), flush=True)

def run(command, limit, **kwargs):
    return subprocess.run(command, check=True, timeout=max(1, min(limit, DEADLINE-time.time())), **kwargs)

def idle():
    data = json.load(urllib.request.urlopen('http://127.0.0.1:8088/status', timeout=4))
    return not any(data.get(key) for key in ['moving','arm','vla','estop','seq','route']) and not data.get('mission', {}).get('active')

try:
    save('waiting_for_download')
    while time.time() < DEADLINE:
        download = json.loads((CKPT / '.download_status.json').read_text())
        if download['state'] == 'failed':
            raise RuntimeError('Checkpoint download failed: '+str(download))
        if download['state'] == 'complete':
            assert download['verified_files'] == download['files']
            break
        time.sleep(5)
    else:
        raise TimeoutError('Waiting for download exceeded deadline')
    for name in [SERVER, CONVERTER]:
        assert subprocess.run(['docker','inspect',name], capture_output=True).returncode != 0, 'Name already exists: '+name
    assert not PT.exists(), 'Converted output already exists; inspect before reusing'
    save('converting', verified_files=download['verified_files'], verified_bytes=download['verified_bytes'])
    command = ['docker','run','--rm','--name',CONVERTER,'--init','--cpus','8','--memory','44g','--memory-swap','44g','--shm-size=8g','-w','/app']
    for entry in ['JAX_PLATFORMS=cpu','PYTHONUNBUFFERED=1','PYTHONDONTWRITEBYTECODE=1','HF_HOME=/hf_home','OMP_NUM_THREADS=8']:
        command += ['-e',entry]
    for entry in ['/data/ta2/software/deploy/server/openpi:/app:ro',str(ROOT)+':/outputs','/root/.cache/openpi:/openpi_assets','/root/.cache/huggingface:/hf_home']:
        command += ['-v',entry]
    command += ['openpi-xpu-env:v0.1.0','python','examples/convert_jax_model_to_pytorch.py','--checkpoint-dir','/outputs/grasp_v3_step20000','--config-name','pi05_teleavatar_v2_chassis','--output-path','/outputs/grasp_v3_step20000_pytorch','--precision','bfloat16']
    with (E/'conversion.log').open('w') as log:
        run(command, 480, stdout=log, stderr=subprocess.STDOUT)
    log_text = (E/'conversion.log').read_text()
    assert 'missing keys: []; unexpected keys: []' in log_text, 'Conversion did not confirm matching tensors'
    assert (PT/'model.safetensors').stat().st_size > 6_000_000_000
    assert (PT/'assets/grasp_20261115_valid/norm_stats.json').read_bytes() == (CKPT/'assets/grasp_20261115_valid/norm_stats.json').read_bytes()
    (PT/'assets/checkout_224_plus_no_scan_224').symlink_to('grasp_20261115_valid', target_is_directory=True)
    save('waiting_for_idle_before_warmup', converted_bytes=(PT/'model.safetensors').stat().st_size)
    for _ in range(60):
        if idle(): break
        time.sleep(5)
    else:
        raise RuntimeError('Robot stays busy; new policy warmup deferred')
    assert not subprocess.check_output(['ss','-ltnH','sport','=',':8005'],text=True).strip(), 'Port 8005 is occupied'
    save('starting_policy_and_warming_up')
    (E/'xpu-cache').mkdir(exist_ok=True)
    command = ['docker','run','-d','--name',SERVER,'--init','--shm-size=8g','-w','/app','--device','/dev/dri','-p','127.0.0.1:8005:8000']
    for entry in ['ZE_AFFINITY_MASK=1','PYTHONUNBUFFERED=1','PYTHONDONTWRITEBYTECODE=1','OPENPI_DATA_HOME=/openpi_assets','HF_HOME=/hf_home','JAX_PLATFORMS=cpu','OMP_NUM_THREADS=8','TORCHINDUCTOR_CACHE_DIR=/xpu-cache/inductor','TRITON_CACHE_DIR=/xpu-cache/triton']:
        command += ['-e',entry]
    for entry in ['/dev/dri/by-path:/dev/dri/by-path','/data/ta2/software/deploy/server/openpi:/app:ro',str(PT)+':/ckpt:ro','/data/ta2/sim_pipeline/openpi-assets:/openpi_assets','/root/.cache/huggingface:/hf_home',str(E/'xpu-cache')+':/xpu-cache',str(E)+':/outputs','/data/ta2:/ta2data:ro']:
        command += ['-v',entry]
    command += ['openpi-xpu-env:v0.1.0','python','scripts/serve_policy.py','--env','TELEAVATAR','--port','8000','--default-prompt','抓取红色沙包','--warmup','--rtc.enabled','--rtc.mode','TRAINED','policy:checkpoint','--policy.config','pi05_teleavatar_v2_chassis','--policy.dir','/ckpt','--policy.pytorch-device','xpu','--policy.pytorch-compile-mode','default']
    run(command,30)
    created_server = True
    warmup_deadline = min(DEADLINE, time.time()+600)
    while time.time() < warmup_deadline:
        output = subprocess.run(['docker','logs',SERVER],capture_output=True,text=True)
        logs = output.stdout + output.stderr
        (E/'server.log').write_text(logs)
        if 'server listening' in logs: break
        running = subprocess.check_output(['docker','inspect','-f','{{.State.Running}}',SERVER],text=True).strip()
        if running != 'true': raise RuntimeError('New policy exited during startup; inspect server.log')
        time.sleep(5)
    else:
        raise TimeoutError('New policy did not become ready within 10 minutes')
    save('offline_smoke_test')
    source = ROOT/'grasp_v3_eval/step5000_train13_old36_20261115T082810Z/eval_ckpt.py'
    shutil.copy2(source,E/'eval_ckpt.py')
    command = ['docker','exec',SERVER,'/opt/venv/bin/python','-u','/outputs/eval_ckpt.py','--ds','/ta2data/review/exports/grasp_20261115_valid_20261115_053753','--episodes','0','--prompt','抓取红色沙包','--port','8000','--tag','step20000_smoke','--out','/outputs/step20000_smoke.json']
    with (E/'smoke.log').open('w') as log:
        run(command,90,stdout=log,stderr=subprocess.STDOUT)
    smoke = json.loads((E/'step20000_smoke.json').read_text())
    assert smoke['summary']['frames'] == 5 and smoke['summary']['episodes'] == 1
    assert all(row['first_deg'] >= 0 and row['mae30_deg'] >= 0 for row in smoke['rows'])
    save('ready_for_web', smoke=smoke['summary'], elapsed_seconds=round(time.time()-START,1))
except BaseException as exc:
    save('failed', error=str(exc))
    traceback.print_exc()
    subprocess.run(['docker','stop','-t','10',CONVERTER],capture_output=True)
    if created_server:
        output=subprocess.run(['docker','logs',SERVER],capture_output=True,text=True)
        (E/'server.log').write_text(output.stdout+output.stderr)
        subprocess.run(['docker','stop','-t','10',SERVER],capture_output=True)
    raise
