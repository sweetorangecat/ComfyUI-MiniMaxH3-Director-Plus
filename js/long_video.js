import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const BASE = "/h3-director-plus/long-video";
const LABELS = {new:"尚未开始",preparing:"核对项目和素材中",preparing_audio:"准备完整音频中",queued:"等待本窗开始",generating:"原生生成中",decoding:"保存原生预览中",awaiting_review:"待确认本窗",awaiting_forward_review:"待审查衔接素材",enhancing:"沿原 U11 增强中",assembling:"装配完整视频中",complete:"完整成片已完成",awaiting_extension:"已到剧情结尾，可补充后继续",stopped:"已停止续拍",stopping:"本窗保存后停止",failed:"运行失败，已保留检查点",rejected:"本窗需要修正",ready:"等待继续",accepted:"已接受",invalidated:"来源已撤回"};
const number = value => Number(value || 0).toFixed(3).replace(/\.?0+$/, "");
function widget(node,name) {
  const actual = node.widgets?.find(item=>item.name === name); if (actual) return actual;
  if (!node._h3LongAutomatic) return undefined;
  if (name === "target_seconds") return node.widgets?.find(item=>item.name === "duration");
  return {get value() {
    if (name === "project_name") return node.properties?.h3_long_video_project || "";
    if (name === "enabled") return Number(widget(node,"duration")?.value) > 15;
    return name === "scene_contract" ? "{}" : "target";
  }, set value(value) { if (name === "project_name") { node.properties ||= {}; node.properties.h3_long_video_project = value; } }};
}
function el(tag,text,field) { const item = document.createElement(tag); if (text !== undefined) item.textContent = text; if (field) item.dataset.field = field; return item; }
function button(parent,label,action,callback) { const item = el("button",label); item.type = "button"; item.dataset.action = action; item.addEventListener("click",callback); parent.append(item); return item; }
function writeWidget(node,name,value) {
  const item = widget(node,name); if (!item) throw new Error("缺少原节点控件："+name);
  const previous = item.value; item.value = value; item.callback?.call(item,value);
  node.onWidgetChanged?.(name,item.value,previous,item); node.graph?.setDirtyCanvas?.(true,true); node.setDirtyCanvas?.(true,true);
}
function canonical(text) {
  let value; try { value = JSON.parse(text); } catch (error) { throw new Error("剧情 JSON 无法解析："+error.message); }
  if (!value || Array.isArray(value) || value.schema_version !== 1 || value.fps !== 24 || !Array.isArray(value.beats) || !value.beats.length) throw new Error("请导入 H3 技能的 U11 单语 scene_contract 导出；{}、source_beats、分镜规划或提示词包不能直接使用。");
  return value;
}
function directorPaths(graph,prefix = "",seen = new Set()) {
  if (!graph || seen.has(graph)) return []; seen.add(graph);
  const result = (graph._nodes || []).flatMap(node=>{
    const path = prefix+node.id;
    return [...([node.comfyClass,node.type].includes("MiniMaxH3DirectorPlus") ? [{node,path}] : []),...directorPaths(node.subgraph,path+":",seen)];
  });
  seen.delete(graph); return result;
}
const directors = graph => directorPaths(graph).map(item=>item.node);

function attachProgress(node,automatic = false) {
  if (node._h3LongInstalled) return; node._h3LongInstalled = true;
  node._h3LongAutomatic = automatic;
  // Keep real serialized widgets; hide only their duplicate native display.
  for (const item of node.widgets || []) if (["enabled","target_seconds","project_name","scene_contract","stop_condition"].includes(item.name)) {
    item.hidden = true; item.options ||= {}; item.options.hidden = true; item.computeSize = ()=>[0,-4]; item.draw = ()=>{};
  }
  const panel = el("section"), compatibility = el("details"); compatibility.append(el("summary","高级兼容设置（旧长片画布）"));
  let root = compatibility;
  const feedback = el("p",automatic ? "使用导演台原有时长、提示词和素材直接排队；超过 15 秒自动续拍。" : "旧配置仍可恢复；普通生成请直接使用原导演台的总时长。","feedback"); feedback.setAttribute("role","status");
  function input(tag,field,placeholder) { const item = el(tag,undefined,field); if (placeholder) item.placeholder = placeholder; item.style.cssText = "max-width:100%;box-sizing:border-box;margin:4px 0"; return item; }
  function row(label,control) { const item = el("label",label+" "); item.append(control); root.append(item,el("br")); }
  const enabled = input("input","enabled"); enabled.type = "checkbox"; row("开启原生长视频",enabled);
  const duration = input("input","target_seconds"); duration.type = "number"; duration.min = "0.05"; duration.step = "any"; row("总时长（秒，可自定义）",duration);
  const presets = el("div"); root.append(presets);
  function setDuration(value) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed <= 0 || Math.round(parsed*24) < 1) { feedback.textContent = "总时长请输入正的有限秒数；没有 30/60/90 秒总时长上限。"; duration.value = widget(node,"target_seconds").value; return; }
    writeWidget(node,"target_seconds",parsed); duration.value = parsed; checkMismatch();
  }
  for (const seconds of [30,60,90]) button(presets,seconds+" 秒","duration-"+seconds,()=>setDuration(seconds));
  const projectInput = input("input","project_name"); row("项目名（同名恢复）",projectInput);
  const stop = input("select","stop_condition");
  for (const [value,label] of [["target","到目标时长结束"],["manual","到当前剧情末尾暂停，可扩展"]]) { const option = el("option",label); option.value = value; stop.append(option); }
  row("结束方式",stop);
  const route = el("p","长片路线：REF2VA / 质量优先二采样 / SelfLift / VOSR2 manual_baseline。","route"); root.append(route);
  button(root,"应用长片推荐设置","recommend",()=>{
    try {
      const choices = directors(app.graph); if (choices.length !== 1) throw new Error("请在原 U11 中保留一个实际导演台，再应用长片设置。");
      const director = choices[0];
      if (!["none","h3_reference"].includes(widget(director,"voice_mode")?.value || "none")) throw new Error("长片当前支持无音色或 H3 原生参考音色；请在原导演台明确选择音色模式后再应用推荐设置。");
      const values = {mode:"REF2VA",performance_preset:"质量优先二采样",sampling_method:"selflift",postprocess_mode:"vosr2",vosr2_profile:"manual_baseline",face_refine_mode:"off",motion_smoothing:"off"};
      const changes = Object.entries(values).map(([name,value])=>{
        const item = widget(director,name); if (!item) throw new Error("原导演台缺少 "+name+"，请更新节点定义。");
        if (name === "performance_preset" && item.options?.values?.includes("quality_two_stage") && !item.options.values.includes(value)) value = "quality_two_stage";
        return {name,value,item,previous:item.value};
      });
      // Assign the complete route before callbacks render; no intermediate invalid route.
      for (const change of changes) change.item.value = change.value;
      for (const {name,value,item,previous} of changes) { item.callback?.call(item,value); director.onWidgetChanged?.(name,item.value,previous,item); }
      director.graph?.setDirtyCanvas?.(true,true);
      feedback.textContent = "已应用长片推荐路线。请核对实际素材与依赖，再预检；音色、种子、画幅、素材、模型及编码选择由原 U11 保留。";
    } catch (error) { feedback.textContent = error.message; }
  });
  const file = input("input","contract-file"); file.type = "file"; file.accept = ".json,application/json"; row("导入 H3 技能 U11 剧情 JSON",file);
  const details = el("details"); details.append(el("summary","粘贴 / 查看剧情 JSON"));
  const script = input("textarea","scene_contract","粘贴 H3 技能导出的 U11 scene_contract JSON"); script.style.cssText += ";width:100%;min-height:140px"; details.append(script);
  button(details,"检查粘贴内容","import",()=>{ try { canonical(script.value); writeWidget(node,"scene_contract",script.value); checkMismatch(); } catch (error) { feedback.textContent = error.message; } });
  button(details,"格式化 JSON","format",()=>{ try { script.value = JSON.stringify(canonical(script.value),null,2); writeWidget(node,"scene_contract",script.value); checkMismatch(); } catch (error) { feedback.textContent = error.message; } });
  const exampleText = input("textarea","example-text"); exampleText.readOnly = true; exampleText.hidden = true; exampleText.style.cssText += ";width:100%;height:180px";
  button(details,"查看格式示例（未审素材）","example",async()=>{
    const capturedEpoch = epoch;
    try { const response = await api.fetchApi(BASE+"/contract/example"); const result = await response.json(); if (capturedEpoch !== epoch || removed) return; if (!response.ok) throw new Error(result.error || response.statusText); exampleText.value = JSON.stringify(result,null,2); exampleText.hidden = false; feedback.textContent = "示例仅供对照，保留来源未审批/未挂载标记；未替换当前剧情，也未排队。"; }
    catch (error) { if (capturedEpoch === epoch) feedback.textContent = error.message; }
  }); details.append(exampleText); root.append(details);
  const adopt = button(root,"采用剧情时长","adopt-duration",()=>{ try { const value = canonical(script.value); if (!Number.isInteger(value.total_frames) || value.total_frames < 1) throw new Error("剧情没有明确 total_frames。"); setDuration(value.total_frames/24); } catch (error) { feedback.textContent = error.message; } }); adopt.hidden = true;
  const preflightActions = el("div"), preflightResult = el("div",undefined,"preflight-result"); root.append(preflightActions,preflightResult);
  root = panel;
  root.style.cssText = "padding:10px;box-sizing:border-box;color:var(--input-text);background:var(--comfy-input-bg);font:13px sans-serif;max-height:680px;overflow:auto;line-height:1.5";
  root.append(el("strong","视频生成进度"),feedback);
  if (!automatic) root.append(compatibility);
  const progressStatus = el("p","尚未排队。","progress-status"), actions = el("div"), selector = input("select","window-select"), content = el("div");
  actions.style.cssText = "display:flex;gap:8px;flex-wrap:wrap"; selector.style.width = "100%";
  const output = el("a","查看 / 下载一个完整 MP4","whole-output"); output.target = "_blank"; output.hidden = true;
  root.append(progressStatus,actions,selector,content,output);
  if (automatic) {
    node._h3LongMount = parent => {
      root.hidden = Number(widget(node,"duration")?.value) <= 15 && !widget(node,"project_name")?.value;
      parent.append(root);
    };
  } else {
    const progressWidget = node.addDOMWidget("long_video_progress","div",root,{serialize:false,hideOnZoom:false,getHeight:()=>860}); progressWidget.computeSize = ()=>[540,860]; node.setSize([Math.max(node.size[0],560),Math.max(node.size[1],900)]);
  }
  let timer, removed = false, activeProject, epoch = 0, sequence = 0, latestApplied = 0, visibleView;
  const projects = new Map(), project = ()=>String(widget(node,"project_name")?.value || (automatic ? "" : "h3-native-av")), endpoint = key=>BASE+"/"+encodeURIComponent(key);
  function ensureProject() {
    const key = project();
    if (activeProject !== key) { activeProject = key; epoch++; latestApplied = 0; clearTimeout(timer); visibleView?.video?.pause(); content.replaceChildren(); visibleView = null; selector.replaceChildren(); selector._signature = undefined; output.hidden = true; }
    if (!projects.has(key)) projects.set(key,{selected:null,views:new Map(),data:null}); return projects.get(key);
  }
  function syncConfig() {
    for (const [name,control] of [["target_seconds",duration],["project_name",projectInput],["stop_condition",stop],["scene_contract",script]]) if (document.activeElement !== control) control.value = widget(node,name)?.value ?? "";
    enabled.checked = Boolean(widget(node,"enabled")?.value); ensureProject();
  }
  function checkMismatch() {
    try { const value = canonical(script.value), seconds = value.total_frames/24, mismatch = Number.isInteger(value.total_frames) && Math.round(Number(widget(node,"target_seconds").value)*24) !== value.total_frames; adopt.hidden = !mismatch;
      feedback.textContent = mismatch ? `剧情为 ${number(seconds)} 秒，目标为 ${number(widget(node,"target_seconds").value)} 秒。可明确采用剧情时长，或重新导出；不会拉伸剧情或删对白。` : "剧情已保存。请预检实际接线；导入不会自动排队或批准素材。";
    } catch (error) { adopt.hidden = true; feedback.textContent = error.message; }
  }
  enabled.addEventListener("change",()=>writeWidget(node,"enabled",enabled.checked)); duration.addEventListener("change",()=>setDuration(duration.value)); stop.addEventListener("change",()=>writeWidget(node,"stop_condition",stop.value));
  projectInput.addEventListener("change",async()=>{ writeWidget(node,"project_name",projectInput.value); ensureProject(); await refresh(); });
  script.addEventListener("input",()=>{ writeWidget(node,"scene_contract",script.value); adopt.hidden = true; });
  file.addEventListener("change",async()=>{
    const capturedEpoch = epoch;
    try { if (!file.files?.[0]) return; const text = await file.files[0].text(); canonical(text); if (capturedEpoch !== epoch || removed) return; script.value = text; writeWidget(node,"scene_contract",text); checkMismatch(); }
    catch (error) { if (capturedEpoch === epoch) feedback.textContent = error.message; } finally { file.value = ""; }
  });
  button(preflightActions,"预检剧情 / 实际接线（不排队）","preflight",async()=>{
    ensureProject(); const capturedEpoch = epoch;
    try {
      const value = canonical(script.value), prompt = await app.graphToPrompt();
      const config = Object.fromEntries(["enabled","target_seconds","project_name","scene_contract","stop_condition"].map(name=>[name,widget(node,name)?.value])); config.scene_contract = value;
      const response = await api.fetchApi(BASE+"/contract/preflight",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:prompt.output,config})}); const result = await response.json();
      if (capturedEpoch !== epoch || removed) return; if (!response.ok) throw new Error(result.error || response.statusText);
      preflightResult.replaceChildren(el("p",`预检完成：${number(result.total_seconds)} 秒，${result.window_count} 窗。实际路线：${result.route}`));
      if (result.encoding) preflightResult.append(el("p",encodingText(result.encoding)));
      for (const text of result.warnings || []) preflightResult.append(el("p",text));
      const inventory = el("details"); inventory.append(el("summary","实际绑定 / 来源声明 / 每窗计划"));
      for (const kind of ["pictures","audios"]) for (const record of result.binding_inventory?.[kind] || []) inventory.append(el("p",`${kind === "pictures" ? "Picture" : "Audio"} ${record.id} · ${record.slot} · ${JSON.stringify(record.binding)}`));
      for (const asset of result.source_facts?.reference_assets || []) inventory.append(el("p",`${asset.slot} 职责：${asset.responsibility}；来源 approved=${asset.approved}, mounted=${asset.mounted}`));
      for (const w of result.windows || []) inventory.append(el("p",`第 ${w.index+1} 窗 · 成片 ${number(w.start_seconds)}–${number(w.end_seconds)} 秒 · 净新增 ${w.net_new_frames} 帧${w.forward_review_required ? " · 需要实际衔接素材审查" : ""}`));
      preflightResult.append(inventory); feedback.textContent = "静态预检完成，未排队。核对来源仍待审的资产与依赖后，使用原 ComfyUI 普通排队；续窗实际 AV 审查仍必需。";
    } catch (error) { if (capturedEpoch === epoch) feedback.textContent = error.message; }
  });
  function encodingText(value) { return `完整成片：${value.codec} / ${value.container} / ${value.bit_depth} / ${value.audio_codec} ${value.audio_bitrate/1000}k · CRF ${value.crf} · BT.709 limited`; }
  async function request(action,data = {}) {
    ensureProject(); const key = activeProject, capturedEpoch = epoch, order = ++sequence;
    if (!key) { feedback.textContent = "使用原导演台正常排队后，这里会显示续拍进度。"; return; }
    try { const response = await api.fetchApi(endpoint(key)+"/control",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action,...data})}); const result = await response.json(); if (capturedEpoch !== epoch || key !== project() || removed) return; if (!response.ok) throw new Error(result.error || response.statusText); if (order >= latestApplied) { latestApplied = order; render(result); } }
    catch (error) { if (capturedEpoch === epoch && key === project()) feedback.textContent = error.message; }
  }
  async function refresh() {
    ensureProject(); clearTimeout(timer); if (removed) return;
    const key = activeProject, capturedEpoch = epoch, order = ++sequence;
    if (!key) return;
    root.hidden = false;
    try { const response = await api.fetchApi(endpoint(key)); const result = await response.json(); if (removed || capturedEpoch !== epoch || key !== project()) return; if (!response.ok) throw new Error(result.error || response.statusText); if (order >= latestApplied) { latestApplied = order; render(result); } }
    catch (error) { if (!removed && capturedEpoch === epoch && key === project()) progressStatus.textContent = error.message; }
    finally { if (!removed && capturedEpoch === epoch && key === project()) timer = setTimeout(refresh,3000); }
  }
  function video(parent,source,field,seek) {
    const item = el("video",undefined,field); item.controls = true; item.preload = "metadata"; item.src = api.apiURL(source); item.style.width = "100%";
    if (seek !== undefined) item.addEventListener("loadedmetadata",()=>{ item.currentTime = seek; }); parent.append(item); return item;
  }
  function buildView(record) {
    const view = {root:el("div"),record}, box = view.root; view.title = el("p"); box.append(view.title);
    if (record.preview_url) view.video = video(box,record.preview_url,"native-video");
    const seam = record.seam;
    if (seam?.parent && seam?.child) {
      const detail = el("details"); detail.open = true; detail.append(el("summary","实际交付接缝：父窗尾部 → 子窗首批净新增帧"));
      for (const [role,label] of [["parent","实际父窗交付尾部"],["child","子窗首批净新增"],["free_boundary","子窗条件出口 / 自由起点"]]) {
        const item = seam[role]; if (!item) continue;
        detail.append(el("p",`${label} · 来源 ${item.take_id} · 预览局部 ${number(item.local_seconds[0])}–${number(item.local_seconds[1])} 秒 · 成片 ${number(item.film_seconds[0])}–${number(item.film_seconds[1])} 秒`,"seam-"+role+"-label")); video(detail,item.url,"seam-"+role,item.local_seconds[0]);
      }
      button(detail,"生成实际接缝对照图","seam-contact",async()=>{
        const capturedEpoch = epoch, key = project();
        try { const response = await api.fetchApi(endpoint(key)+"/seam/"+encodeURIComponent(record.take_id)); const result = await response.json(); if (capturedEpoch !== epoch || key !== project() || removed) return; if (!response.ok) throw new Error(result.error || response.statusText); if (!view.contact) { view.contact = el("img"); view.contact.style.width = "100%"; detail.append(view.contact); } view.contact.src = api.apiURL(result.image_url); }
        catch (error) { if (capturedEpoch === epoch) feedback.textContent = error.message; }
      }); box.append(detail);
    } else if (seam?.error) box.append(el("p",seam.error));
    view.review = el("div"); view.review.append(el("p","检查完整原生动作、口型、声音、手和道具及实际交付接缝。填写实际尾态与已检查范围。"));
    const outgoing = input("textarea","outgoing","实际尾态：人物位置、手持物、动作阶段、机位和声音状态"), scope = input("textarea","scope","实际检查范围与发现（动作、声音、自由边界和收尾）");
    for (const field of [outgoing,scope]) { field.style.cssText += ";width:100%;min-height:65px"; view.review.append(field); }
    // Elements persist for the take across polls, selection and project switches.
    button(view.review,"接受本窗，增强并继续","accept",()=>{ if (!outgoing.value.trim() || !scope.value.trim()) { feedback.textContent = "请填写实际尾态和实际审查范围。"; return; } return request("review",{take_id:record.take_id,accepted:true,observed_outgoing_state:outgoing.value,scope:scope.value}); });
    button(view.review,"退回本窗","reject",()=>{ if (!scope.value.trim()) { feedback.textContent = "请记录退回原因与实际检查范围。"; return; } return request("review",{take_id:record.take_id,accepted:false,observed_outgoing_state:outgoing.value,scope:scope.value}); }); box.append(view.review);
    view.forward = el("div"); view.forwardLabel = el("p",undefined,"forward-label"); view.forward.append(view.forwardLabel);
    const observed = input("textarea","forward-state","实际衔接素材末端 / 下一窗自由起点状态，不能填写预期结果"), forwardScope = input("textarea","forward-scope","实际检查的源尾部画面、声音、接触与动作范围");
    for (const field of [observed,forwardScope]) { field.style.cssText += ";width:100%;min-height:65px"; view.forward.append(field); }
    button(view.forward,"定位实际来源尾段","forward-seek",()=>{ if (view.video) view.video.currentTime = view.record.forward_review.source_seconds[0]; });
    button(view.forward,"确认实际衔接素材，继续","review-forward",()=>{ if (!observed.value.trim() || !forwardScope.value.trim()) { feedback.textContent = "请填写实际衔接状态和检查范围。"; return; } return request("review_forward",{take_id:record.take_id,observed_state:observed.value,scope:forwardScope.value}); }); box.append(view.forward);
    view.repair = el("details"); view.repair.append(el("summary","局部返修（视频；锁定原音频）"));
    const start = input("input","repair-start"), end = input("input","repair-end");
    for (const item of [start,end]) { item.type = "number"; item.step = "any"; } start.value = record.free_start_seconds ?? record.start; end.value = record.end;
    view.repair.append(el("span","成片秒数：从 "),start,el("span"," 到 "),end);
    view.repairButton = button(view.repair,"按成片秒数返修视频","repair",()=>{
      const a = Number(start.value), b = Number(end.value);
      if (!start.value.toString().trim() || !end.value.toString().trim() || !Number.isFinite(a) || !Number.isFinite(b) || !(a < b)) { feedback.textContent = "返修起止秒数必须为有限数字且结束大于开始。"; return; }
      return request("repair",{index:record.index,take_id:record.take_id,start_seconds:a,end_seconds:b,stream:"video"});
    }); view.extent = el("p",undefined,"repair-extent"); box.append(view.repair,view.extent);
    view.retry = button(box,"重新生成本窗","retry",()=>request("retry",{index:record.index,take_id:record.take_id,...(view.record.status === "accepted" ? {replace_accepted:true} : {})})); return view;
  }
  function render(data) {
    const projectState = ensureProject(); projectState.data = data;
    progressStatus.textContent = `${LABELS[data.status] || data.status} · 目标 ${number(data.target_seconds ?? data.actual_seconds)} 秒 / 连续已确认 ${number(data.completed_seconds)} 秒 / 已生成候选至 ${number(data.generated_seconds)} 秒`+(data.error ? "\n"+data.error : "")+(data.forward_review_error ? "\n"+data.forward_review_error : "");
    if (data.encoding) route.textContent = "长片路线：REF2VA / 质量优先二采样 / SelfLift / VOSR2 manual_baseline。"+encodingText(data.encoding);
    const windows = data.windows || [];
    if (!windows.some(w=>w.index === projectState.selected)) projectState.selected = (windows.find(w=>w.status === "awaiting_review" || w.status === "rejected" || (w.forward_review && !w.forward_review.review_receipt_exists)) || windows.at(-1))?.index ?? null;
    const signature = JSON.stringify(windows.map(w=>[w.index,w.take_id,w.start,w.end,w.status]));
    if (selector._signature !== signature) { selector.replaceChildren(...windows.map(w=>{ const option = el("option",`第 ${w.index+1} 窗 · ${number(w.start)}–${number(w.end)} 秒 · ${LABELS[w.status] || w.status}`); option.value = String(w.index); return option; })); selector._signature = signature; }
    selector.value = projectState.selected === null ? "" : String(projectState.selected);
    const record = windows.find(w=>w.index === projectState.selected);
    if (record) {
      // A preview may arrive after queued metadata: rebuild once at arrival.
      let view = projectState.views.get(record.take_id); if (view && !view.video && record.preview_url) { projectState.views.delete(record.take_id); view = null; }
      if (!view) { view = buildView(record); projectState.views.set(record.take_id,view); } view.record = record;
      if (visibleView !== view) { visibleView?.video?.pause(); content.replaceChildren(view.root); visibleView = view; }
      view.title.textContent = `第 ${record.index+1} 窗 · 成片交付 ${number(record.start)}–${number(record.end)} 秒 · 来源 ${record.take_id}。完整预览含保护历史和生成余量；实际交付为预览局部 ${number(record.local_new_start)}–${number(record.local_new_end)} 秒。`;
      view.review.hidden = record.status !== "awaiting_review";
      const forward = record.forward_review; view.forward.hidden = !forward || (forward.review_receipt_exists && data.status !== "awaiting_forward_review");
      if (forward) view.forwardLabel.textContent = `衔接素材审查：实际来源 ${record.take_id} 的预览局部 ${number(forward.source_seconds[0])}–${number(forward.source_seconds[1])} 秒。检查这段未交付尾部的实际动作和声音，记录末端实际自由起点状态；已有交付审批不会自动批准此尾段。`;
      view.repair.hidden = !["accepted","rejected"].includes(record.status); view.repairButton.disabled = Boolean(data.busy);
      view.retry.hidden = !record.can_retry; view.retry.disabled = Boolean(data.busy); view.retry.textContent = record.status === "accepted" ? "整窗重新生成（后续需重做，保留旧文件）" : (record.repair ? "放弃局部返修，重新生成本窗" : "重新生成本窗（新种子）");
      const extent = record.repair?.conditioning_extent;
      view.extent.textContent = extent ? `实际返修条件范围：${extent.video_seconds ? "视频成片 "+extent.video_seconds.map(number).join("–")+" 秒" : ""}${extent.audio_seconds ? "；音频成片 "+extent.audio_seconds.map(number).join("–")+" 秒" : "；原音频锁定"}。按原生时间网格向外对齐；VAE 存在时间耦合，不能保证逐帧隔离。` : "";
    } else if (visibleView) { visibleView.video?.pause(); content.replaceChildren(); visibleView = null; }
    output.hidden = !data.output_url; if (data.output_url) output.href = api.apiURL(data.output_url); node.setDirtyCanvas?.(true,true);
  }
  selector.addEventListener("change",()=>{ const state = ensureProject(); state.selected = Number(selector.value); if (state.data) render(state.data); });
  button(actions,"刷新进度","refresh",refresh); button(actions,"继续 / 恢复","resume",()=>request("resume")); button(actions,"停止续拍","stop",()=>request("stop"));
  node._h3LongRefresh = refresh;
  const originalConfigure = node.onConfigure; node.onConfigure = function () { originalConfigure?.apply(this,arguments); syncConfig(); };
  const originalRemoved = node.onRemoved; node.onRemoved = function () { removed = true; epoch++; clearTimeout(timer); visibleView?.video?.pause(); originalRemoved?.apply(this,arguments); };
  syncConfig(); timer = setTimeout(refresh,0);
}
app.registerExtension({name:"MiniMaxH3.NativeAVLongVideo",async beforeRegisterNodeDef(nodeType,nodeData) {
  if (nodeData.name === "MiniMaxH3DirectorPlus") { const original = nodeType.prototype.onNodeCreated; nodeType.prototype.onNodeCreated = function () { original?.apply(this,arguments); attachProgress(this,true); }; }
  if (nodeData.name === "MiniMaxH3LongVideoConfig") { const original = nodeType.prototype.onNodeCreated; nodeType.prototype.onNodeCreated = function () { original?.apply(this,arguments); attachProgress(this); }; }
  if (nodeData.name === "MiniMaxH3U11Output") { const original = nodeType.prototype.onExecuted; nodeType.prototype.onExecuted = function (message) {
    original?.apply(this,arguments);
    if (!message.h3_long_video) return;
    for (const state of message.h3_long_video) {
      const choices = directorPaths(app.graph), identity = String(state.director_id ?? "");
      const matches = choices.filter(item=>item.path === identity);
      for (const {node} of matches.length ? matches : choices.length === 1 ? choices : []) {
        node.properties ||= {}; node.properties.h3_long_video_project = state.project_name; node._h3LongRefresh?.();
      }
    }
    for (const node of app.graph?._nodes || []) if (!node._h3LongAutomatic) node._h3LongRefresh?.();
  }; }
}});
