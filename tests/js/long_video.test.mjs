import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

// Exercise the real extension lifecycle. Only the Comfy host and DOM are stubs.
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.style = {}; this.listeners = {}; this.value = ''; this.currentTime = 0; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; this.rebuilds = (this.rebuilds || 0) + 1; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  setAttribute(name, value) { this[name] = value; }
  async fire(type) { for (const fn of this.listeners[type] || []) await fn({target: this, preventDefault() {}}); }
  pause() { this.paused = true; }
}
const all = (root) => [root, ...root.children.flatMap(all)];
const find = (node, key) => all(node.root).find(item => item.dataset.field === key || item.dataset.action === key);
const widget = (node, name) => node.widgets.find(item => item.name === name);
const windowRecord = (index, status = 'awaiting_review') => ({index, status, take_id: 'take'+index,
  start: index ? 12.25 : 0, end: index ? 22.875 : 12.25,
  local_new_start: index ? 1.625 : 0, local_new_end: 12.25,
  preview_url: '/native'+index+'.mp4', can_retry: ['accepted','rejected','invalidated'].includes(status)});
const state = (windows = [], extra = {}) => ({project_name: 'A', status: 'awaiting_review',
  actual_seconds: 90, target_seconds: 90, completed_seconds: 12.25, generated_seconds: 22.875, windows, ...extra});
const contract = {schema_version: 1, fps: 24, total_frames: 2160, beats: [{id: 'x'}], not_ready_for_video: true};

async function install() {
  let extension;
  const calls = [], callbacks = [], timers = [];
  const host = {reply: state(), responder: null};
  const director = {comfyClass: 'MiniMaxH3DirectorPlus', widgets: Object.entries({mode:'FL2VA', performance_preset:'智能画质（自动适配）', sampling_method:'standard',
    postprocess_mode:'ai_upscale', vosr2_profile:'legacy', face_refine_mode:'off', motion_smoothing:'off',
    seed: 314159, voice_mode:'h3_reference', resolution_preset:'2K QHD', reference_image_1_file:'same.png'}).map(([name,value]) => ({name,value,callback() { callbacks.push([name,this.value]); }})),
    onWidgetChanged() {}, graph: {setDirtyCanvas() {}}};
  const app = {graph: {_nodes: [director]}, registerExtension(value) { extension = value; },
    async graphToPrompt() { return {output: {'actual': {class_type:'ActualGraph', inputs:{}}}}; }};
  const api = {apiURL: value => value, async fetchApi(url, options = {}) {
    const call = {url, options, data: options.body ? JSON.parse(options.body) : undefined}; calls.push(call);
    if (host.responder) return host.responder(call);
    return {ok: true, async json() { return url.endsWith('/contract/example') ? contract : host.reply; }};
  }};
  const context = vm.createContext({app, api, document: {createElement: tag => new Element(tag), activeElement: null},
    setTimeout: (fn, ms) => { timers.push({fn,ms}); return timers.length; }, clearTimeout() {}, console});
  const source = readFileSync(new URL('../../js/long_video.js', import.meta.url), 'utf8');
  vm.runInContext(source.replace(/^import .*;\r?\n/gm, ''), context);
  class Node {
    constructor() { this.widgets = Object.entries({enabled:false,target_seconds:30,project_name:'A',scene_contract:'{}',stop_condition:'target'}).map(([name,value]) => ({name,value,options:{},callback() {}})); this.size = [500,760]; }
    addDOMWidget(name, type, element) { this.root = element; const item = {name,type,element}; this.widgets.push(item); return item; }
    setSize(value) { this.size = value; } computeSize() { return [500,760]; } setDirtyCanvas() {}
  }
  await extension.beforeRegisterNodeDef(Node, {name:'MiniMaxH3LongVideoConfig'});
  const node = new Node(); node.onNodeCreated();
  app.graph._nodes.push(node);
  return {node, host, calls, timers, director, callbacks, context, extension};
}

async function installOriginalDirector() {
  const fixture = await install();
  class OriginalDirector {
    constructor() { this.id=42; this.comfyClass='MiniMaxH3DirectorPlus'; this.properties={};
      this.widgets=[{name:'duration',value:90},{name:'prompt',value:'整片时间轴'}]; this.size=[1350,1760]; }
    addDOMWidget() { throw new Error('Progress must mount inside the original Director UI'); }
    setDirtyCanvas() {}
  }
  await fixture.extension.beforeRegisterNodeDef(OriginalDirector,{name:'MiniMaxH3DirectorPlus'});
  const director=new OriginalDirector(); fixture.context.app.graph._nodes=[director];
  director.onNodeCreated?.();
  const container=new Element('section'); director._h3LongMount?.(container); director.root=container;
  class Output {}
  await fixture.extension.beforeRegisterNodeDef(Output,{name:'MiniMaxH3U11Output'});
  return {...fixture,director,output:new Output()};
}

test('original Director has integrated progress without setup/file/preflight controls',async()=>{
  const {director}=await installOriginalDirector();
  assert.ok(find(director,'progress-status'));
  for (const field of ['recommend','contract-file','preflight','target_seconds','enabled']) assert.equal(find(director,field),undefined);
  assert.equal(widget(director,'duration').value,90);
});

test('original progress explains that each candidate awaits actual review and acceptance before continuing',async()=>{
  const {director}=await installOriginalDirector();
  const feedback=find(director,'feedback').textContent;
  assert.match(feedback,/逐窗复核后续拍/);
  assert.match(feedback,/候选.*实际复核.*实际尾态.*接受.*增强.*下一窗/);
  assert.equal(all(director.root).filter(item=>item.tag==='input' && item.type==='file').length,0);
});

test('actual output event binds the automatic project to its original Director and reload preserves it',async()=>{
  const {director,output,host,calls}=await installOriginalDirector();
  host.reply=state([], {project_name:'u11-real-project'});
  output.onExecuted({h3_long_video:[{project_name:'u11-real-project',director_id:'42'}]});
  await director._h3LongRefresh();
  assert.equal(director.properties.h3_long_video_project,'u11-real-project');
  assert.ok(calls.some(call=>call.url.endsWith('/u11-real-project')));
  assert.ok(!calls.some(call=>call.url.endsWith('/contract/preflight')));
});

test('original progress labels the queued whole-audio preparation stage',async()=>{
  const {director,output,host}=await installOriginalDirector();
  host.reply=state([], {project_name:'u11-audio',status:'preparing_audio',busy:true});
  output.onExecuted({h3_long_video:[{project_name:'u11-audio',director_id:'42'}]});
  await director._h3LongRefresh();
  assert.match(find(director,'progress-status').textContent,/准备完整音频中/);
});

test('same local Director id in separate subgraphs binds only the exact full execution path',async()=>{
  const {director,output,context}=await installOriginalDirector();
  const other={id:42,comfyClass:'MiniMaxH3DirectorPlus',properties:{},widgets:[]};
  context.app.graph._nodes=[{id:'A',subgraph:{_nodes:[director]}},{id:'B',subgraph:{_nodes:[other]}}];
  output.onExecuted({h3_long_video:[{project_name:'u11-only-A',director_id:'A:42'}]});
  assert.equal(director.properties.h3_long_video_project,'u11-only-A');
  assert.equal(other.properties.h3_long_video_project,undefined);
});

test('presets and finite custom duration use saved widgets without enabling or queueing', async () => {
  const {node,calls} = await install();
  assert.ok(find(node, 'duration-90'), 'duration presets missing');
  await find(node, 'duration-90').fire('click');
  assert.equal(widget(node,'target_seconds').value,90);
  const input = find(node,'target_seconds'); input.value = '97.125'; await input.fire('change');
  assert.equal(widget(node,'target_seconds').value,97.125);
  input.value = 'Infinity'; await input.fire('change');
  assert.equal(widget(node,'target_seconds').value,97.125);
  assert.equal(widget(node,'enabled').value,false);
  assert.ok(!calls.some(call => call.url.includes('/prompt')));
});

test('import preserves contract/duration, adopt is explicit and preflight never queues', async () => {
  const {node,host,calls} = await install();
  assert.ok(find(node,'scene_contract'), 'contract editor missing');
  const file = find(node,'contract-file'); file.files = [{async text() { return JSON.stringify(contract); }}];
  await file.fire('change');
  assert.equal(JSON.parse(widget(node,'scene_contract').value).not_ready_for_video,true);
  assert.equal(widget(node,'target_seconds').value,30);
  assert.match(find(node,'feedback').textContent,/90.*30|30.*90/);
  await find(node,'adopt-duration').fire('click');
  assert.equal(widget(node,'target_seconds').value,90);
  host.reply = {total_seconds:90,window_count:9,windows:[],warnings:['pending'],binding_inventory:{pictures:[],audios:[]},encoding:{codec:'H.264',container:'MP4',bit_depth:'8-bit',audio_codec:'AAC',crf:16,audio_bitrate:96000}, route:'resolved'};
  await find(node,'preflight').fire('click');
  const call = calls.find(call => call.url.endsWith('/contract/preflight'));
  assert.ok(call && call.data.prompt.actual);
  assert.equal(call.data.config.target_seconds,90);
  await find(node,'example').fire('click');
  assert.equal(JSON.parse(widget(node,'scene_contract').value).total_frames,2160);
  assert.ok(!calls.some(call => /\/control$|\/prompt$/.test(call.url)));
});

test('rich planning bundle is rejected without silently rewriting saved contract', async () => {
  const {node} = await install();
  const file = find(node,'contract-file');
  assert.ok(file, 'file import missing');
  file.files = [{async text() { return JSON.stringify({source_beats:[]}); }}];
  await file.fire('change');
  assert.equal(widget(node,'scene_contract').value,'{}');
  assert.match(find(node,'feedback').textContent,/导出/);
});

test('recommended settings invoke real callbacks and preserve unrelated saved choices', async () => {
  const {node,director,callbacks} = await install();
  const untouched = Object.fromEntries(director.widgets.filter(w => ['seed','voice_mode','resolution_preset','reference_image_1_file'].includes(w.name)).map(w => [w.name,w.value]));
  assert.ok(find(node,'recommend'), 'explicit recommended action missing');
  await find(node,'recommend').fire('click');
  assert.equal(widget(director,'mode').value,'REF2VA');
  assert.equal(widget(director,'performance_preset').value,'质量优先二采样');
  assert.equal(widget(director,'sampling_method').value,'selflift');
  assert.equal(widget(director,'postprocess_mode').value,'vosr2');
  assert.equal(widget(director,'vosr2_profile').value,'manual_baseline');
  assert.equal(callbacks.length,7);
  assert.deepEqual(Object.fromEntries(Object.keys(untouched).map(name => [name,widget(director,name).value])),untouched);
});

test('unsupported voice route requires an explicit original-voice choice before recommendations mutate anything', async () => {
  const {node,director,callbacks} = await install();
  widget(director,'voice_mode').value = 'fish_lock';
  const before = director.widgets.map(w=>w.value);
  await find(node,'recommend').fire('click');
  assert.deepEqual(director.widgets.map(w=>w.value),before);
  assert.equal(callbacks.length,0);
  assert.match(find(node,'feedback').textContent,/音色/);
});

test('polling retains focused drafts, playback and earlier window selection', async () => {
  const {node,host,context} = await install();
  host.reply = state([windowRecord(0,'accepted'),windowRecord(1)]);
  await node._h3LongRefresh();
  const outgoing = find(node,'outgoing'), video = find(node,'native-video');
  outgoing.value = 'actual hand on cup'; await outgoing.fire('input'); context.document.activeElement = outgoing;
  video.currentTime = 7.25;
  await node._h3LongRefresh();
  assert.equal(find(node,'outgoing'),outgoing);
  assert.equal(find(node,'native-video'),video);
  assert.equal(outgoing.value,'actual hand on cup'); assert.equal(video.currentTime,7.25);
  const select = find(node,'window-select'); select.value = '0'; await select.fire('change');
  assert.equal(find(node,'native-video').src,'/native0.mp4');
  await node._h3LongRefresh(); assert.equal(select.value,'0');
  assert.equal(find(node,'native-video').src,'/native0.mp4');
  select.value = '1'; await select.fire('change');
  assert.equal(find(node,'outgoing').value,'actual hand on cup');
  assert.equal(find(node,'native-video').currentTime,7.25);
});

test('seconds repair and accepted retry send current take identity and explicit replacement', async () => {
  const {node,host,calls} = await install();
  host.reply = state([windowRecord(0,'accepted')]); await node._h3LongRefresh();
  assert.ok(find(node,'repair-start'), 'seconds repair controls missing');
  find(node,'repair-start').value = '2.1'; find(node,'repair-end').value = '2.6';
  await find(node,'repair').fire('click');
  const repair = calls.find(c => c.data?.action === 'repair').data;
  assert.deepEqual(repair,{action:'repair',index:0,take_id:'take0',start_seconds:2.1,end_seconds:2.6,stream:'video'});
  await find(node,'retry').fire('click');
  const retry = calls.find(c => c.data?.action === 'retry').data;
  assert.deepEqual(retry,{action:'retry',index:0,take_id:'take0',replace_accepted:true});
});

test('content and forward reviews require actual observed state and scope', async () => {
  const {node,host,calls} = await install();
  const record = windowRecord(0); host.reply = state([record]); await node._h3LongRefresh();
  await find(node,'accept').fire('click'); assert.ok(!calls.some(c => c.data?.action === 'review'));
  find(node,'outgoing').value = 'Actual cup in right hand'; find(node,'scope').value = 'Actual video and audio checked';
  await find(node,'accept').fire('click');
  assert.equal(calls.find(c => c.data?.action === 'review').data.observed_outgoing_state,'Actual cup in right hand');
  host.reply = state([{...record,status:'accepted',forward_review:{source_seconds:[12.25,12.958333],consumed_seconds:[10.625,12.958333],review_receipt_exists:false}}],{status:'awaiting_forward_review'});
  await node._h3LongRefresh();
  assert.match(find(node,'forward-label').textContent,/12\.25/);
  find(node,'forward-state').value = 'Actual fingers released, cup stable'; find(node,'forward-scope').value = 'Actual tail and sound checked';
  await find(node,'review-forward').fire('click');
  const forward = calls.find(c => c.data?.action === 'review_forward').data;
  assert.deepEqual(forward,{action:'review_forward',take_id:'take0',observed_state:'Actual fingers released, cup stable',scope:'Actual tail and sound checked'});
});

test('actual seam preview seeks parent tail and child first net-new independently', async () => {
  const {node,host} = await install();
  const record = windowRecord(1);
  record.seam = {parent:{take_id:'old-parent',url:'/parent.mp4',local_seconds:[12.08333,12.25],film_seconds:[12.08333,12.25]},
    child:{take_id:'take1',url:'/child.mp4',local_seconds:[1.625,1.79167],film_seconds:[12.25,12.41667]},
    free_boundary:{take_id:'take1',url:'/child.mp4',local_seconds:[2.25,2.41667],film_seconds:[12.875,13.04167]}};
  host.reply = state([record]); await node._h3LongRefresh();
  const parent = find(node,'seam-parent'), child = find(node,'seam-child');
  assert.ok(parent && child,'actual delivery seam videos missing');
  await parent.fire('loadedmetadata'); await child.fire('loadedmetadata');
  assert.equal(parent.currentTime,12.08333); assert.equal(child.currentTime,1.625);
  assert.match(find(node,'seam-parent-label').textContent,/old-parent/);
});

test('project switch ignores stale asynchronous response and restores project drafts', async () => {
  const {node,host} = await install();
  host.reply = state([windowRecord(0)]); await node._h3LongRefresh();
  find(node,'outgoing').value = 'project A draft'; await find(node,'outgoing').fire('input');
  let resolveOld;
  host.responder = () => new Promise(resolve => { resolveOld = resolve; });
  const old = node._h3LongRefresh();
  const project = find(node,'project_name'); project.value = 'B';
  host.responder = async () => ({ok:true,json:async () => state([windowRecord(1)],{project_name:'B'})});
  await project.fire('change'); await node._h3LongRefresh();
  resolveOld({ok:true,json:async () => state([windowRecord(0)],{project_name:'A',completed_seconds:999})}); await old;
  assert.equal(find(node,'native-video').src,'/native1.mp4');
  assert.ok(!find(node,'progress-status').textContent.includes('999'));
  project.value = 'A'; host.responder = null; host.reply = state([windowRecord(0)]); await project.fire('change'); await node._h3LongRefresh();
  assert.equal(find(node,'outgoing').value,'project A draft');
  node.onRemoved();
});

test('same-project older poll cannot replace newer state and output keeps one whole file link', async () => {
  const {node,host} = await install();
  let oldResolve;
  host.responder = () => new Promise(resolve => { oldResolve = resolve; });
  const old = node._h3LongRefresh();
  host.responder = async () => ({ok:true,json:async () => state([windowRecord(1)],{status:'complete',completed_seconds:90,output_url:'/whole.mp4'})});
  await node._h3LongRefresh();
  oldResolve({ok:true,json:async () => state([windowRecord(0)],{completed_seconds:1})}); await old;
  assert.equal(find(node,'native-video').src,'/native1.mp4');
  assert.equal(find(node,'whole-output').href,'/whole.mp4');
  assert.equal(find(node,'whole-output').hidden,false);
  assert.match(find(node,'progress-status').textContent,/90/);
});

test('repair prefill uses exact free-start and end and reports actual snapped extent', async () => {
  const {node,host,calls} = await install();
  const record = {...windowRecord(1,'accepted'),start:549/24,end:2332/24,free_start_seconds:566/24,
    repair:{conditioning_extent:{video_seconds:[23.583333333333332,24.291666666666668],frame_isolation_guaranteed:false}}};
  host.reply = state([record]); await node._h3LongRefresh();
  assert.equal(Number(find(node,'repair-start').value),566/24);
  assert.equal(Number(find(node,'repair-end').value),2332/24);
  assert.match(find(node,'repair-extent').textContent,/VAE.*不能保证/);
  await find(node,'repair').fire('click');
  assert.equal(calls.find(c=>c.data?.action === 'repair').data.start_seconds,566/24);
});

test('independent clips describe the original short route and retry only this clip',async()=>{
  const {node,host}=await install();
  const record={...windowRecord(1,'accepted'),kind:'independent_shots',local_new_start:0,
    local_new_end:15,seam:{kind:'editorial_cut',parent:{take_id:'take0',url:'/p.mp4',
    local_seconds:[14.8,15],film_seconds:[14.8,15]},child:{take_id:'take1',url:'/c.mp4',
    local_seconds:[0,.2],film_seconds:[15,15.2]}}};
  host.reply=state([record],{entry_mode:'independent_shots'}); await node._h3LongRefresh();
  assert.match(find(node,'route').textContent,/原短片/);
  assert.match(find(node,'retry').textContent,/只重做本段/);
  assert.doesNotMatch(find(node,'retry').textContent,/后续需重做/);
  assert.equal(all(node.root).find(e=>e.tag==='details' && e.children.some(c=>c.textContent?.includes('局部返修'))).hidden,true);
  assert.ok(all(node.root).some(e=>e.textContent?.includes('剧情切镜')));
  assert.doesNotMatch(all(node.root).find(e=>e.dataset.field==='clip-title').textContent,/保护历史/);
});

test('original output execution hook refreshes the mounted long panel', async () => {
  const {node,host,extension,calls} = await install();
  host.reply = state([windowRecord(0)]);
  class Output {}
  await extension.beforeRegisterNodeDef(Output,{name:'MiniMaxH3U11Output'});
  new Output().onExecuted({h3_long_video:[{project_name:'A'}]});
  await Promise.resolve(); await Promise.resolve();
  assert.ok(calls.some(c=>c.url.endsWith('/A')));
});
