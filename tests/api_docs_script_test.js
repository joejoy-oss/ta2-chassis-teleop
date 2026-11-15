// Exercise page request/polling logic without pretending this is a browser rendering test.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../api_docs.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
const context = vm.createContext({assert, URLSearchParams, AbortController,
  location: {origin: 'http://localhost:8088'}, performance,
  setTimeout: (fn, ms) => setTimeout(fn, ms === 500 ? 0 : ms), clearTimeout,
  fetch: async () => { throw Error('Unexpected real network access'); }});
vm.runInContext(script.split('for(const def of definitions)buildCard(def);')[0], context);
vm.runInContext(`(async()=>{
  const makeCard=(id,values={})=>{
    const def=definitions.find(d=>d.id===id);
    const card={def,inputs:{},test:{disabled:false},meta:{},output:{},cancelResult:{},curl:{},python:{}};
    if(def.action)card.cancel={disabled:true};
    for(const f of def.fields||[])card.inputs[f.key]={value:String(values[f.key]??f.value??''),disabled:false};
    cards.set(id,card);return card;
  };
  const query=makeCard('action'),cancel=makeCard('cancel');
  const grasp=makeCard('vla.grasp');
  assert.deepEqual(JSON.parse(JSON.stringify(values(grasp))), {stop_on_grasp:false});
  const mission=makeCard('mission.run',{rtc:'false',attempts:'2',prompt:'抓取沙包'});
  assert.equal(values(mission).rtc,false);
  assert.equal(values(mission).attempts,2);
  assert.ok(!('port' in values(mission)));
  assert.equal(requestFor(query.def,{id:'a/b?x'}).path,'/api/v1/actions/a%2Fb%3Fx');
  assert.equal(requestFor(cards.get('cancel').def,{id:'abc'}).path,'/api/v1/actions/abc/cancel');
  assert.equal(requestFor(definitions.find(d=>d.id==='preflight'),{}).path,'/api/v1/preflight');
  examples(mission);
  assert.ok(mission.python.textContent.includes('rtc=False'));
  assert.ok(mission.curl.textContent.includes('Content-Type: application/json'));
  let calls=[];
  const realRequest=request;
  let fetchCalls=[];
  fetch=async(path,options)=>{fetchCalls.push({path,options});return {status:409,ok:false,text:async()=>JSON.stringify({error:{code:'BUSY',message:'busy'}})}};
  const busy=await realRequest('POST','/api/v1/actions',{type:'odom.reset',params:{}});
  assert.equal(busy.status,409);assert.equal(busy.data.error.code,'BUSY');
  assert.equal(fetchCalls[0].options.method,'POST');
  assert.equal(JSON.parse(fetchCalls[0].options.body).type,'odom.reset');
  await assert.rejects(()=>realRequest('POST','/estop'),/仅允许/);
  const health=makeCard('health');
  request=async(method,path,body)=>{calls.push({method,path,body});return {ok:true,status:200,ms:7,data:{ok:true}}};
  await runCard(health);
  assert.equal(health.meta.textContent,'HTTP 200 · 7 ms');
  assert.equal(JSON.parse(health.output.textContent).ok,true);
  assert.equal(calls[0].path,'/api/v1/health');
  calls=[];
  const arm=makeCard('arm.down',{hand:'left',cm:'2.5'});
  const replies=[{ok:true,status:202,ms:2,data:{action_id:'trial',action:{status:'running'}}},
    {ok:true,status:200,ms:3,data:{id:'trial',status:'running'}},
    {ok:true,status:200,ms:4,data:{id:'trial',status:'succeeded',result:{returncode:0}}}];
  request=async(method,path,body)=>{calls.push({method,path,body});assert.ok(replies.length);return replies.shift()};
  await runCard(arm);
  assert.equal(calls.length,3);
  assert.equal(calls[0].body.type,'arm.down');assert.equal(calls[0].body.params.cm,2.5);
  assert.equal(calls[1].path,'/api/v1/actions/trial');
  assert.equal(JSON.parse(arm.output.textContent).status,'succeeded');
  assert.ok(arm.meta.textContent.includes('启动 HTTP 202'));
  assert.ok(arm.meta.textContent.includes('HTTP 200'));
  assert.equal(arm.cancel.disabled,true);assert.equal(arm.test.disabled,false);
  assert.equal(query.inputs.id.value,'trial');
  request=async()=>({ok:false,status:409,ms:1,data:{error:{code:'BUSY',message:'busy'}}});
  await runCard(arm);
  assert.equal(arm.aid,null);assert.equal(JSON.parse(arm.output.textContent).error.code,'BUSY');
  arm.aid='route';
  request=async(method,path)=>{assert.equal(method,'POST');assert.equal(path,'/api/v1/actions/route/cancel');return {status:409,ms:5,data:{error:{code:'ROUTE_ESTOP_ONLY'}}}};
  await cancelAction(arm);assert.ok(arm.cancelResult.textContent.includes('ROUTE_ESTOP_ONLY'));
  request=async()=>{throw Error('offline')};
  await runCard(health);assert.ok(health.meta.textContent.includes('请求可能已接受'));assert.equal(health.test.disabled,false);
  return 'Page requests, examples, polling, cancellation, HTTP errors and network errors passed.';
})()`, context).then(message => process.stdout.write(message + '\n'), error => {console.error(error);process.exitCode=1;});
