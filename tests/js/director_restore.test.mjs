import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { clearSlot, compactBoundSlots, compactSlots } from "../../js/media_slot_state.mjs";

// Exercise the actual extension's install/configure/render lifecycle without
// ComfyUI or a GPU. Only the DOM and host services are stubbed.
class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.listeners = {};
    this.style = { setProperty() {} };
    this.classList = { add() {}, remove() {}, toggle() {} };
  }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.append(child); }
  replaceChildren(...children) {
    this.replaceChildrenCount = (this.replaceChildrenCount || 0) + 1;
    this.children = children;
  }
  setAttribute(name, value) { this[name] = value; }
  setCustomValidity(message) { this.validationMessage = message; }
  closest(selector) {
    if (selector === "[data-h3p-value-widget]"
      || selector === ".h3p [data-h3p-value-widget]"
      || selector === '.h3p [data-h3p-value-widget]:not(select)') return this;
    return null;
  }
  matches(selector) { return selector === "select" && this.tag === "select"; }
  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  get innerHTML() { return this.html || ""; }
  set innerHTML(value) {
    this.html = value;
    if (value.includes('class="h3p-fish-body"')) {
      const body = new Element("div");
      body.className = "h3p-fish-body";
      this.append(body);
    }
  }
  querySelector(selector) {
    return this.children.find(child => selector.startsWith(".")
      ? child.className === selector.slice(1) : child.tag === selector) || null;
  }
  querySelectorAll() { return []; }
}

function installedDirector(initialValue = "off", extraWidgets = [], nativeWidgets = false, hostApi = {}) {
  let extension;
  const frames = [];
  const changes = [];
  const documentListeners = {};
  const context = vm.createContext({
    app: { registerExtension(value) { extension = value; } },
    api: hostApi,
    FormData,
    URLSearchParams,
    clearSlot, compactBoundSlots, compactSlots,
    document: {
      createElement: (tag) => new Element(tag),
      head: new Element("head"),
      body: { contains: () => false },
      documentElement: new Element("html"),
      addEventListener(type, listener) { (documentListeners[type] ||= []).push(listener); },
      querySelectorAll: () => [],
    },
    window: {},
    MutationObserver: class { observe() {} disconnect() {} },
    requestAnimationFrame: (fn) => frames.push(fn),
    setTimeout() {},
  });
  const source = readFileSync(new URL("../../js/minimax_h3_director_plus_v11.js", import.meta.url), "utf8");
  vm.runInContext(source.replace(/^import .*;\r?\n/gm, ""), context);
  const face = { name: "face_refine_mode", value: initialValue, options: { values: ["off", "auto"] } };
  const node = {
    comfyClass: "MiniMaxH3DirectorPlus",
    widgets: nativeWidgets ? extraWidgets : [face, { name: "mode", value: "T2VA" }, ...extraWidgets],
    size: [1350, 1760],
    setSize(value) { this.size = value; },
    graph: { setDirtyCanvas() {} },
    onWidgetChanged(name, value, previous, item) { changes.push({ name, value, previous, item }); },
    // Simulates the host's restoration callback, before our wrapper runs.
    onConfigure(data) { if (!nativeWidgets) face.value = data.face_refine_mode; },
    serialize() { return nativeSave(this, false, true); },
    addDOMWidget(name, type, element) {
      this.root = element;
      const item = { name, type, element, value: "", options: { serialize: false } };
      if (nativeWidgets) this.widgets.push(item);
      return item;
    },
  };
  extension.nodeCreated(node);
  return { node, face, frames, extension, changes, documentListeners };
}

for (const mode of ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"]) {
  for (const voiceMode of ["none", "h3_reference", "fish_lock", "audio_reuse"]) {
    test(`${mode} / ${voiceMode} restores the low-VRAM selection without replacing it`, () => {
      const preset = { name: "performance_preset", value: "低显存" };
      const { node } = installedDirector("off", [preset,
        { name: "voice_mode", value: voiceMode },
        { name: "postprocess_mode", value: "ai_upscale" },
        { name: "resolution_preset", value: "768p H3" },
      ]);
      node.widgets.find(item => item.name === "mode").value = mode;
      node.widgets.find(item => item.name === "mode").callback(mode);
      assert.equal(preset.value, "低显存");
      assert.equal(node.widgets.find(item => item.name === "postprocess_mode").value, "ai_upscale");
    });
  }
}

test("rendering an empty legacy face mode writes off into the submitted widget", () => {
  const { face } = installedDirector("");
  assert.equal(face.value, "off");
});

for (const legacyValue of [undefined, null, ""]) {
  test(`old workflow restores ${String(legacyValue)} to off before the next animation frame`, () => {
    const { node, face } = installedDirector();
    node.onConfigure({ face_refine_mode: legacyValue });
    // A queue/save may happen before the deferred render callback.
    assert.equal(face.value, "off");
  });
}

test("legacy face mode is disabled and no face control is rendered", () => {
  const { node, face } = installedDirector("auto");
  node.onConfigure({ face_refine_mode: "auto" });
  assert.equal(face.value, "off");
  assert.ok(!allElements(node.root).some(el => el.dataset?.h3pValueWidget === "face_refine_mode"));
});

function allElements(root) {
  return [root, ...(root.children || []).flatMap(allElements)];
}

test("original duration accepts long totals and rendering never clamps them to a physical window", () => {
  for (const preset of ["稳定质量", "质量优先二采样", "低显存二采"]) {
    const {node} = installedDirector("off", [
      {name:"duration",value:90}, {name:"performance_preset",value:preset},
      {name:"resolution_preset",value:"768p H3"},
    ]);
    assert.equal(node.widgets.find(item => item.name === "duration").value,90);
    const input = allElements(node.root).find(el => el.dataset?.h3pValueWidget === "duration");
    assert.ok(input);
    assert.ok(!input.max || Number(input.max) > 90);
    assert.ok(allElements(node.root).some(el => String(el.textContent || "").includes("续拍")));
  }
});

test("refreshing the original cached U11 retires only the untouched generated long panel", () => {
  const {extension}=installedDirector();
  const director={id:42,type:'MiniMaxH3DirectorPlus',widgets_values:['saved prompt',90],properties:{}};
  const output={id:43,type:'MiniMaxH3StreamingVideoCombine',properties:{},inputs:[{name:'long_video',link:8}]};
  const generated={id:44,type:'MiniMaxH3LongVideoConfig',properties:{cnr_id:'ComfyUI-MiniMaxH3-Director-Plus'},
    widgets_values:[false,30,'h3-native-av','{}','target'],inputs:[],outputs:[{links:[8]}]};
  const graph={nodes:[director,output,generated],links:[[8,44,0,43,0,'MINIMAX_H3_LONG_VIDEO']],groups:[{id:'u11-long-video'}]};
  extension.beforeConfigureGraph(graph);
  assert.equal(graph.nodes.length,2);
  assert.equal(output.type,'MiniMaxH3U11Output');
  assert.equal(output.inputs[0].link,null);
  assert.equal(graph.links.length,0);
  assert.equal(graph.groups.length,0);
});

test("cached expert config controlled by a linked boolean remains untouched", () => {
  const {extension}=installedDirector();
  const config={id:44,type:'MiniMaxH3LongVideoConfig',properties:{cnr_id:'ComfyUI-MiniMaxH3-Director-Plus'},
    widgets_values:[false,30,'h3-native-av','{}','target'],inputs:[{name:'enabled',link:9}],outputs:[{links:[8]}]};
  const graph={nodes:[{id:42,type:'MiniMaxH3DirectorPlus'},config,
    {id:43,type:'MiniMaxH3U11Output',inputs:[{name:'long_video',link:8}]}],
    links:[[9,45,0,44,0,'BOOLEAN'],[8,44,0,43,0,'MINIMAX_H3_LONG_VIDEO']]};
  extension.beforeConfigureGraph(graph);
  assert.ok(graph.nodes.includes(config));
  assert.equal(graph.links.length,2);
});

test("Director capture handler lets integrated progress buttons reach their own listeners", () => {
  const {node}=installedDirector();
  const button=new Element('button'); button.dataset.action='resume';
  button.closest=selector=>selector === '.h3p button' ? button : null;
  let stopped=false,prevented=false,targetCalls=0;
  button.addEventListener('click',()=>{targetCalls++;});
  const event={target:button,stopPropagation(){stopped=true;},preventDefault(){prevented=true;}};
  for (const fn of node.root.listeners.click || []) fn(event);
  if (!stopped) for (const fn of button.listeners.click) fn(event);
  assert.equal(stopped,false);
  assert.equal(prevented,false);
  assert.equal(targetCalls,1);
});

for (const method of ["h3_two_stage", "vosr2", "video_sr"]) {
  test(`smart upscale selector is enabled and includes ${method}`, () => {
    const { node } = installedDirector("off", [
      { name: "postprocess_mode", value: method },
      { name: "performance_preset", value: "智能画质（自动适配）" },
    ]);
    const select = allElements(node.root).find(el => el.dataset?.h3pValueWidget === "postprocess_mode");
    assert.ok(select);
    assert.equal(select.disabled, undefined);
    assert.ok(select.children.some(el => el.value === method));
  });
}

for (const preset of ["稳定质量", "质量优先加速", "质量优先二采样", "低显存"]) {
  test(`${preset} exposes three SR engines and preserves selected VOSR2`, () => {
    const output = { name: "postprocess_mode", value: "vosr2" };
    const { node } = installedDirector("off", [output,
      { name: "performance_preset", value: preset },
    ]);
    const select = allElements(node.root).find(el => el.dataset?.h3pValueWidget === "postprocess_mode");
    assert.deepEqual(select.children.map(el => el.value), ["vosr2", "video_sr", "ai_upscale"]);
    assert.equal(output.value, "vosr2");
  });
}

test("SelfLift and motion profiles restore and reset when leaving two-stage", () => {
  const method = { name: "sampling_method", value: "selflift" };
  const profile = { name: "community_lora_profile", value: "continuity" };
  const preset = { name: "performance_preset", value: "质量优先二采样" };
  const { node } = installedDirector("off", [method, profile, preset,
    { name: "voice_mode", value: "none" }, { name: "duration", value: 5 },
  ]);
  // Host restore can restore these before the next render.
  node.widgets.find(el => el.name === "mode").value = "REF2VA";
  method.value = "selflift";
  node.widgets.find(el => el.name === "mode").callback("REF2VA");
  assert.equal(method.value, "selflift");
  assert.equal(profile.value, "continuity");
  assert.ok(allElements(node.root).some(el => el.dataset?.h3pValueWidget === "sampling_method"));
  preset.value = "低显存";
  preset.callback("低显存");
  assert.equal(method.value, "standard");
  assert.equal(profile.value, "off");
});

for (const voiceMode of ["h3_reference", "audio_reuse"]) {
  test(`REF2VA ${voiceMode} exposes the SelfLift choice`, () => {
    const { node } = installedDirector("off", [
      { name: "performance_preset", value: "质量优先二采样" },
      { name: "sampling_method", value: "standard" },
      { name: "community_lora_profile", value: "off" },
      { name: "voice_mode", value: voiceMode },
      { name: "resolution_preset", value: "2K QHD" },
    ]);
    const mode = node.widgets.find(item => item.name === "mode");
    mode.value = "REF2VA";
    mode.callback("REF2VA");
    const select = allElements(node.root).find(el => el.dataset?.h3pValueWidget === "sampling_method");
    assert.deepEqual(select.children.map(el => el.value), ["standard", "selflift"]);
  });
}

function advancedPanel(node) {
  return allElements(node.root).find(el => el.tag === "details"
    && el.children.some(child => child.textContent === "二采高级设置"));
}

function voiceDirector(duration, { mode = "REF2VA", voiceMode = "fish_lock", profile = "off" } = {}) {
  const values = { mode, duration, voice_mode: voiceMode, performance_preset: "质量优先二采样",
    sampling_method: "selflift", community_lora_profile: profile };
  const widgets = nativeWidgets();
  for (const item of widgets) if (Object.hasOwn(values, item.name)) item.value = values[item.name];
  return installedDirector("off", widgets, true);
}

function renderedControl(node, name) {
  return allElements(node.root).find(el => el.dataset?.h3pValueWidget === name);
}

function editValue(installed, scope, type, name, value) {
  const input = renderedControl(installed.node, name);
  assert.ok(input, `missing editor for ${name}`);
  const closest = input.closest.bind(input);
  input.closest = selector => selector === ".h3p[data-h3p-controller]" ? installed.node.root : closest(selector);
  input.value = value;
  const listeners = scope === "document" ? installed.documentListeners : installed.node.root.listeners;
  for (const handler of listeners[type] || []) handler({ target: input, stopPropagation() {} });
  return input;
}

for (const scope of ["root", "document"]) {
  for (const eventType of ["input", "change"]) {
    test(`${scope} ${eventType} preserves a fractional duration instead of submitting the previous valid long total`, () => {
      const installed = voiceDirector(24);
      const { node } = installed;
      assert.equal(valueOf(node, "performance_preset"), "质量优先二采样");
      assert.equal(valueOf(node, "sampling_method"), "selflift");
      const preserved = ["mode", "voice_mode", "prompt", "seed", "community_lora_profile",
        ...STATE_FIELDS.filter(name => name.endsWith("_file") || name.startsWith("voice_reference_name_"))];
      const before = Object.fromEntries(preserved.map(name => [name, valueOf(node, name)]));
      const input = editValue(installed, scope, eventType, "duration", "24.5");
      assert.equal(valueOf(node, "duration"), 24.5);
      assert.equal(nativeSave(node).widgets_values_named.duration, 24.5);
      assert.match(input.validationMessage, /至少 4 秒的整数/);
      if (eventType === "input") editValue(installed, scope, "change", "duration", "24.5");
      assert.equal(valueOf(node, "duration"), 24.5);
      assert.notEqual(valueOf(node, "performance_preset"), "质量优先二采样");
      assert.equal(valueOf(node, "sampling_method"), "standard");
      assert.equal(renderedControl(node, "duration").value, 24.5);
      assert.match(renderedControl(node, "duration").validationMessage, /至少 4 秒的整数/);
      const text = allElements(node.root).map(el => el.textContent || "").join(" ");
      assert.match(text, /视频总时长请输入至少 4 秒的整数/);
      assert.doesNotMatch(text, /计划后端：REF2VA 原生续拍/);
      for (const name of preserved) assert.equal(valueOf(node, name), before[name], name);
      assert.deepEqual(Object.keys(nativeSave(node).widgets_values_named), STATE_FIELDS);
    });
  }

  test(`${scope} duration editor keeps other invalid drafts invalid for submission and displays the error`, () => {
    for (const draft of ["Infinity", "", "NaN", "3", "15.5", "24suffix"]) {
      const installed = voiceDirector(24);
      const expected = draft.trim() ? Number(draft) : NaN;
      editValue(installed, scope, "input", "duration", draft);
      assert.equal(valueOf(installed.node, "duration"), expected, draft);
      editValue(installed, scope, "change", "duration", draft);
      assert.equal(valueOf(installed.node, "duration"), expected, draft);
      assert.equal(nativeSave(installed.node).widgets_values_named.duration,
        Number.isFinite(expected) ? expected : null, draft);
      assert.match(renderedControl(installed.node, "duration").validationMessage, /至少 4 秒的整数/, draft);
      assert.notEqual(valueOf(installed.node, "performance_preset"), "质量优先二采样", draft);
    }
  });

  test(`${scope} duration editor preserves valid original long and short transitions`, () => {
    const installed = voiceDirector(24);
    for (const draft of ["15", "24", "15", "24.0"]) {
      editValue(installed, scope, "input", "duration", draft);
      editValue(installed, scope, "change", "duration", draft);
      const seconds = Number(draft);
      assert.equal(valueOf(installed.node, "duration"), seconds);
      assert.equal(nativeSave(installed.node).widgets_values_named.duration, seconds);
      assert.equal(renderedControl(installed.node, "duration").validationMessage, "");
      assert.equal(renderedControl(installed.node, "performance_preset").children
        .some(el => el.value === "质量优先二采样"), seconds > 15);
      assert.equal(valueOf(installed.node, "mode"), "REF2VA");
      assert.equal(valueOf(installed.node, "voice_mode"), "fish_lock");
    }
  });

  test(`${scope} numeric editors retain existing seed and dimension parsing`, () => {
    const installed = installedDirector("off", [
      { name: "seed", value: 1 }, { name: "aspect_ratio", value: "CUSTOM" },
      { name: "custom_width", value: 16 }, { name: "custom_height", value: 9 },
    ]);
    for (const name of ["seed", "custom_width"]) {
      editValue(installed, scope, "input", name, "42.5");
      assert.equal(valueOf(installed.node, name), 42);
      editValue(installed, scope, "change", name, "42.5");
      assert.equal(valueOf(installed.node, name), 42);
    }
  });
}

test("saved 24-second Fish restores SelfLift synchronously and preserves original bindings and slots", () => {
  const values = { duration: 24, voice_mode: "fish_lock", performance_preset: "质量优先二采样",
    sampling_method: "selflift", community_lora_profile: "continuity",
    prompt: "0–24 秒：<Picture 1> 人物使用 <Audio 1> 音色，<Audio 2> 和 <Audio 3> 保留原角色职责。",
    target_dialogue: "保留原对白。", reference_transcript: "保留样本文本。",
    voice_reference_name_3: "旁白", voice_reference_audio_3_file: "input/旁白.wav" };
  const widgets = nativeWidgets();
  for (const item of widgets) if (Object.hasOwn(values, item.name)) item.value = values[item.name];
  const saved = nativeSave({ widgets }, false);
  const original = { ...saved.widgets_values_named };
  const preserved = ["mode", "prompt", "duration", "seed", "control_after_generate", "voice_mode",
    "fish_model_path", "target_dialogue", "reference_transcript", "community_lora_profile",
    ...STATE_FIELDS.filter(name => name.endsWith("_file") || name.startsWith("voice_reference_name_"))];
  const { node, extension, frames } = installedDirector("off", nativeWidgets(), true);
  extension.beforeConfigureGraph({ nodes: [saved] });
  nativeRestore(node, saved);
  // Queueing and saving may occur before the deferred DOM render.
  assert.equal(valueOf(node, "sampling_method"), "selflift");
  assert.equal(valueOf(node, "performance_preset"), "质量优先二采样");
  for (const callback of frames.splice(0)) callback();
  for (const name of preserved) assert.equal(valueOf(node, name), original[name], name);
  const roundTrip = nativeSave(node);
  assert.deepEqual(Object.keys(roundTrip.widgets_values_named), STATE_FIELDS);
  assert.equal(roundTrip.widgets_values.length, STATE_FIELDS.length);
  for (const name of preserved) assert.equal(roundTrip.widgets_values_named[name], original[name], name);
  assert.equal(roundTrip.widgets_values_named.sampling_method, "selflift");
  assert.equal(roundTrip.widgets_values_named.performance_preset, "质量优先二采样");
});

for (const mode of ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"]) {
  test(`24-second ${mode} Fish exposes existing quality and SelfLift choices`, () => {
    const { node } = voiceDirector(24, { mode, profile: "continuity" });
    assert.equal(valueOf(node, "mode"), mode);
    assert.equal(valueOf(node, "voice_mode"), "fish_lock");
    assert.equal(valueOf(node, "duration"), 24);
    assert.equal(valueOf(node, "performance_preset"), "质量优先二采样");
    assert.equal(valueOf(node, "sampling_method"), "selflift");
    assert.equal(valueOf(node, "community_lora_profile"), "continuity");
    assert.ok(renderedControl(node, "performance_preset").children.some(el => el.value === "质量优先二采样"));
    assert.deepEqual(renderedControl(node, "sampling_method").children.map(el => el.value), ["standard", "selflift"]);
  });
}

test("24-second REF2VA Fish keeps the author recipe eligible with complete target audio requirements", () => {
  const { node } = voiceDirector(24, { profile: "combat" });
  assert.equal(valueOf(node, "performance_preset"), "质量优先二采样");
  assert.equal(valueOf(node, "sampling_method"), "selflift");
  assert.equal(valueOf(node, "community_lora_profile"), "combat");
  assert.equal(advancedPanel(node).open, false);
  assert.deepEqual(renderedControl(node, "sampling_method").children.map(el => el.value), ["selflift"]);
  const text = allElements(advancedPanel(node)).map(el => el.textContent || "").join(" ");
  assert.match(text, /simple.*12.*6\+6/);
  assert.match(text, /Fish.*完整目标音频/);
  assert.doesNotMatch(text, /作者配方需要/);
});

for (const mode of ["T2VA", "I2VA", "FL2VA", "L2VA"]) {
  test(`24-second Fish author recipe preserves incompatible original ${mode} with an explanation`, () => {
    const { node } = voiceDirector(24, { mode, profile: "combat" });
    assert.equal(valueOf(node, "mode"), mode);
    assert.equal(valueOf(node, "voice_mode"), "fish_lock");
    assert.equal(valueOf(node, "performance_preset"), "质量优先二采样");
    assert.equal(valueOf(node, "sampling_method"), "selflift");
    assert.equal(valueOf(node, "community_lora_profile"), "combat");
    assert.equal(advancedPanel(node).open, true);
    const text = allElements(advancedPanel(node)).map(el => el.textContent || "").join(" ");
    assert.match(text, /作者配方需要.*REF2VA/);
    assert.match(text, /Fish.*完整目标音频/);
    assert.doesNotMatch(text, /不支持 Fish/);
  });
}

for (const duration of [4, 15, 15.5, 24.5, NaN, Infinity, -Infinity, "24", null, undefined, true, [24]]) {
  test(`Fish duration ${String(duration)} (${typeof duration}) keeps short-route restrictions`, () => {
    const { node } = voiceDirector(duration);
    assert.equal(valueOf(node, "duration"), duration);
    assert.notEqual(valueOf(node, "performance_preset"), "质量优先二采样");
    assert.equal(valueOf(node, "sampling_method"), "standard");
    assert.ok(!renderedControl(node, "performance_preset").children.some(el => el.value === "质量优先二采样"));
    assert.equal(renderedControl(node, "sampling_method"), undefined);
    const text = allElements(node.root).map(el => el.textContent || "").join(" ");
    assert.doesNotMatch(text, /计划后端：REF2VA 原生续拍/);
  });
}

test("15-second Fish keeps an explicit incompatible author recipe visible", () => {
  const { node } = voiceDirector(15, { profile: "combat" });
  assert.equal(valueOf(node, "community_lora_profile"), "combat");
  assert.equal(valueOf(node, "sampling_method"), "selflift");
  assert.equal(advancedPanel(node).open, true);
  assert.match(allElements(advancedPanel(node)).map(el => el.textContent || "").join(" "), /作者配方需要.*不支持 Fish/);
});

test("long audio reuse retains the original voice choice without claiming SelfLift support", () => {
  const { node } = voiceDirector(24, { voiceMode: "audio_reuse" });
  assert.equal(valueOf(node, "voice_mode"), "audio_reuse");
  assert.equal(valueOf(node, "sampling_method"), "standard");
  assert.deepEqual(renderedControl(node, "sampling_method").children.map(el => el.value), ["standard"]);
  assert.match(allElements(node.root).map(el => el.textContent || "").join(" "), /超过 15 秒.*不支持完整复用/);
});

test("long audio reuse keeps an explicit incompatible author recipe and explains the unsupported voice", () => {
  const { node } = voiceDirector(24, { voiceMode: "audio_reuse", profile: "combat" });
  assert.equal(valueOf(node, "voice_mode"), "audio_reuse");
  assert.equal(valueOf(node, "community_lora_profile"), "combat");
  assert.equal(valueOf(node, "sampling_method"), "selflift");
  assert.equal(advancedPanel(node).open, true);
  assert.match(allElements(advancedPanel(node)).map(el => el.textContent || "").join(" "), /作者配方需要.*不支持完整复用/);
});

test("long Fish help describes review-gated continuation and the existing picture quota without extra uploads", () => {
  const { node } = voiceDirector(24);
  const { node: short } = voiceDirector(15);
  const text = allElements(node.root).map(el => el.textContent || "").join(" ");
  assert.match(text, /逐窗复核后续拍/);
  assert.match(text, /候选.*实际复核.*实际尾态.*接受.*增强.*下一窗/);
  assert.match(text, /8 张静态.*1 张父窗.*9 张/);
  assert.match(allElements(advancedPanel(node)).map(el => el.textContent || "").join(" "), /Fish.*对白.*环境声.*音效.*完整目标音频/);
  const uploadFields = director => allElements(director.root).filter(el => el.dataset?.h3pUpload).map(el => el.dataset.h3pUpload);
  assert.deepEqual(uploadFields(node), uploadFields(short));
  assert.ok(uploadFields(node).every(name => STATE_FIELDS.includes(name)));
});

test("VOSR tuning is collapsed by default and preserves saved values", () => {
  const profile = { name: "vosr2_profile", value: "manual_baseline" };
  const { node } = installedDirector("off", [profile,
    { name: "performance_preset", value: "均衡高质（768p 8步）" },
    { name: "postprocess_mode", value: "vosr2" },
    { name: "resolution_preset", value: "1080p FHD" },
  ]);
  const panel = () => allElements(node.root).find(el => el.tag === "details"
    && el.children.some(child => child.textContent === "超分高级设置"));
  assert.equal(panel().open, false);
  assert.ok(allElements(panel()).some(el => el.dataset?.h3pValueWidget === "vosr2_profile"));
  assert.equal(profile.value, "manual_baseline");
  panel().open = true;
  profile.callback(profile.value);
  assert.equal(panel().open, true);
});

test("SelfLift VOSR2 panel upgrades the legacy speed choice to the quality baseline", () => {
  const profile = { name: "vosr2_profile", value: "legacy" };
  const { node } = installedDirector("off", [profile,
    { name: "performance_preset", value: "质量优先二采样" },
    { name: "postprocess_mode", value: "vosr2" },
    { name: "sampling_method", value: "selflift" },
    { name: "community_lora_profile", value: "off" },
    { name: "resolution_preset", value: "2K QHD" },
  ]);
  const mode = node.widgets.find(item => item.name === "mode");
  mode.value = "REF2VA";
  mode.callback("REF2VA");
  const sampling = node.widgets.find(item => item.name === "sampling_method");
  sampling.value = "selflift";
  sampling.callback("selflift");
  const panel = () => allElements(node.root).find(el => el.tag === "details"
    && el.children.some(child => child.textContent === "超分高级设置"));
  assert.equal(profile.value, "manual_baseline");
  const select = allElements(panel()).find(el => el.dataset?.h3pValueWidget === "vosr2_profile");
  assert.deepEqual(select.children.map(el => el.value), ["manual_baseline", "experimental_32gb"]);
});

function directorWithAdvancedOptions() {
  const installed = installedDirector("off", [
    { name: "performance_preset", value: "质量优先二采样" },
    { name: "sampling_method", value: "standard" },
    { name: "community_lora_profile", value: "off" },
    { name: "voice_mode", value: "none" },
    { name: "resolution_preset", value: "1080p FHD" },
  ]);
  const mode = installed.node.widgets.find(item => item.name === "mode");
  mode.value = "REF2VA";
  mode.callback("REF2VA");
  return installed;
}

for (const voiceMode of ["none", "h3_reference", "audio_reuse"]) {
  test(`author Wenwu selection chooses its sampler for ${voiceMode}`, () => {
    const { node } = directorWithAdvancedOptions();
    const voice = node.widgets.find(item => item.name === "voice_mode");
    voice.value = voiceMode;
    voice.callback(voiceMode);
    const profile = node.widgets.find(item => item.name === "community_lora_profile");
    profile.value = "combat";
    profile.callback("combat");
    assert.equal(node.widgets.find(item => item.name === "sampling_method").value, "selflift");
    const profileControl = allElements(node.root).find(el => el.dataset?.h3pValueWidget === "community_lora_profile");
    assert.equal(profileControl.children.find(el => el.value === "combat").textContent, "文武双全（作者配方）");
    const samplerControl = allElements(node.root).find(el => el.dataset?.h3pValueWidget === "sampling_method");
    assert.deepEqual(samplerControl.children.map(el => el.value), ["selflift"]);
    assert.ok(allElements(node.root).some(el => /simple.*12.*6\+6/.test(el.textContent || "")));
    const quick = allElements(node.root).find(el => el.innerHTML?.includes("快速设置"));
    assert.match(quick.innerHTML, /simple.*12.*6\+6/);
    assert.doesNotMatch(quick.innerHTML, /首采 8 步/);
  });
}

test("author Wenwu cannot silently downgrade its sampler and can be switched off", () => {
  const { node } = directorWithAdvancedOptions();
  const profile = node.widgets.find(item => item.name === "community_lora_profile");
  const method = node.widgets.find(item => item.name === "sampling_method");
  profile.value = "combat";
  profile.callback("combat");
  method.value = "standard";
  method.callback("standard");
  assert.equal(method.value, "selflift");
  profile.value = "off";
  profile.callback("off");
  method.value = "standard";
  method.callback("standard");
  assert.equal(method.value, "standard");
});

test("incompatible author selection remains visible with a clear recovery path", () => {
  const { node } = directorWithAdvancedOptions();
  const profile = node.widgets.find(item => item.name === "community_lora_profile");
  profile.value = "combat";
  profile.callback("combat");
  const preset = node.widgets.find(item => item.name === "performance_preset");
  preset.value = "低显存";
  preset.callback("低显存");
  assert.equal(profile.value, "combat");
  const profileControl = allElements(node.root).find(el => el.dataset?.h3pValueWidget === "community_lora_profile");
  assert.ok(profileControl, "the user can still turn the profile off");
  assert.ok(allElements(node.root).some(el => /作者配方.*需要.*质量优先二采样.*REF2VA/.test(el.textContent || "")));
  profile.value = "off";
  profile.callback("off");
  assert.equal(node.widgets.find(item => item.name === "sampling_method").value, "standard");
  assert.equal(preset.value, "低显存");
});

test("saved author selection preserves long duration, prompt, media and seed", () => {
  const saved = nativeWidgets();
  const values = { performance_preset: "质量优先二采样", community_lora_profile: "combat",
    sampling_method: "standard", duration: 90 };
  for (const item of saved) if (item.name in values) item.value = values[item.name];
  const preservedNames = ["duration", "prompt", "seed", "first_image_file", "last_image_file", "reference_image_1_file", "voice_mode"];
  const preserved = preservedNames.map(name => [name, saved.find(item => item.name === name).value]);
  const { node } = installedDirector("off", saved, true);
  assert.equal(node.widgets.find(item => item.name === "sampling_method").value, "selflift");
  for (const [name, value] of preserved) assert.equal(node.widgets.find(item => item.name === name).value, value, name);
});

test("advanced settings stay open across sampling, motion and output changes", () => {
  const { node } = directorWithAdvancedOptions();
  advancedPanel(node).open = true;
  for (const [name, value] of [["sampling_method", "selflift"],
    ["community_lora_profile", "combat"], ["resolution_preset", "4K UHD"]]) {
    const item = node.widgets.find(widget => widget.name === name);
    item.value = value;
    item.callback(value);
    assert.equal(advancedPanel(node).open, true, name);
    assert.equal(item.value, value);
  }
  advancedPanel(node).open = false;
  const method = node.widgets.find(widget => widget.name === "sampling_method");
  method.callback(method.value);
  assert.equal(advancedPanel(node).open, false);
});

test("advanced panel state belongs to each director and survives preset changes", () => {
  const { node } = directorWithAdvancedOptions();
  advancedPanel(node).open = true;
  const { node: other } = directorWithAdvancedOptions();
  assert.ok(!advancedPanel(other).open);
  const preset = node.widgets.find(widget => widget.name === "performance_preset");
  for (const value of ["稳定质量", "质量优先二采样"]) {
    preset.value = value;
    preset.callback(value);
  }
  assert.equal(advancedPanel(node).open, true);
});

test("loading a legacy U11 removes the face branch and preserves audio and prompts", () => {
  const { extension } = installedDirector();
  const graph = {
    nodes: [
      { id: 1, type: "MiniMaxH3DirectorPlus", widgets_values: ["user prompt"] },
      { id: 2, type: "MiniMaxH3ColorGuard", outputs: [{ links: [10] }] },
      { id: 3, type: "MiniMaxH3FaceRefineSwitch", properties: { director_plus_face_refine: true },
        inputs: [{ name: "original_images", link: 10 }], outputs: [{ links: [11] }] },
      { id: 4, type: "MiniMaxH3StreamingVideoCombine", inputs: [{ name: "images", link: 11 }, { name: "audio", link: 12 }] },
      { id: 5, type: "VAEDecodeAudio", outputs: [{ links: [12] }] },
      { id: 6, type: "PreviewImage", properties: { director_plus_face_refine: true }, inputs: [{ link: 13 }] },
    ],
    links: [[10, 2, 0, 3, 0, "IMAGE"], [11, 3, 0, 4, 0, "IMAGE"], [12, 5, 0, 4, 1, "AUDIO"], [13, 3, 0, 6, 0, "IMAGE"]],
    groups: [{ title: "FaceRefine 人脸修复" }, { title: "Output" }],
  };
  extension.beforeConfigureGraph(graph);
  assert.deepEqual(graph.nodes.map(n => n.id), [1, 2, 4, 5]);
  assert.deepEqual(JSON.parse(JSON.stringify(graph.links)), [[11, 2, 0, 4, 0, "IMAGE"], [12, 5, 0, 4, 1, "AUDIO"]]);
  assert.deepEqual(graph.nodes[1].outputs[0].links, [11]);
  assert.deepEqual(graph.nodes[0].widgets_values, ["user prompt"]);
  assert.deepEqual(graph.groups, [{ title: "Output" }]);
  const once = JSON.stringify(graph);
  extension.beforeConfigureGraph(graph);
  assert.equal(JSON.stringify(graph), once);
});

test("afterQueued seed update retains the native numeric callback receiver", () => {
  const seed = {
    name: "seed", value: 0, options: { step2: 1 },
    callback(value) {
      const step = this.options.step2 || 1;
      this.value = Math.round(value / step) * step;
    },
  };
  installedDirector("off", [seed]);
  // Same method call as ComfyUI's applyWidgetControl after randomize/increment.
  seed.callback(12345.6);
  assert.equal(seed.value, 12346);
});

test("wrapped widget callbacks preserve host arguments and return values", () => {
  const calls = [];
  const seed = {
    name: "seed", value: 0,
    callback(...args) { calls.push({ receiver: this, args }); return "host-result"; },
  };
  const { node } = installedDirector("off", [seed]);
  const canvas = {};
  assert.equal(seed.callback(42, canvas, node), "host-result");
  assert.equal(calls[0].receiver, seed);
  assert.deepEqual(calls[0].args, [42, canvas, node]);
});

test("typing a seed keeps the editor mounted until the value is committed", () => {
  let callbackCalls = 0;
  const seed = {
    name: "seed", value: 0,
    callback(value) {
      callbackCalls += 1;
      this.value = value;
    },
  };
  const { node } = installedDirector("off", [seed]);
  const input = allElements(node.root).find(el => el.dataset?.h3pValueWidget === "seed");
  assert.ok(input, "seed editor is rendered");
  const beforeRender = node.root.replaceChildrenCount;
  input.value = "100372459382532";
  const event = { target: input, stopPropagation() {} };
  for (const handler of node.root.listeners.input || []) handler(event);
  assert.equal(seed.value, 100372459382532);
  assert.equal(callbackCalls, 0, "typing must not invoke the native callback");
  assert.equal(node.root.replaceChildrenCount, beforeRender, "typing must not rebuild the editor");

  for (const handler of node.root.listeners.change || []) handler(event);
  assert.equal(callbackCalls, 1, "committing the field still invokes the native callback");
});

test("smart mode retains 480p selection and synchronizes actual output dimensions", () => {
  const preset = { name: "resolution_preset", value: "480p" };
  const width = { name: "width", value: 1920 };
  const height = { name: "height", value: 1080 };
  installedDirector("off", [preset, width, height,
    { name: "aspect_ratio", value: "16:9" },
    { name: "performance_preset", value: "智能画质（自动适配）" },
  ]);
  assert.equal(preset.value, "480p");
  assert.equal(width.value, 854);
  assert.equal(height.value, 480);
});
for (const method of ["h3_two_stage", "vosr2", "video_sr"]) {
  test(`smart FHD render preserves selected ${method}`, () => {
    const output = { name: "postprocess_mode", value: method };
    installedDirector("off", [output,
      { name: "resolution_preset", value: "1080p FHD" },
      { name: "performance_preset", value: "智能画质（自动适配）" },
    ]);
    assert.equal(output.value, method);
  });
}


test("15-second low-VRAM panel shows sampling size separately from final output", () => {
  const { node } = installedDirector("off", [
    { name: "performance_preset", value: "低显存" },
    { name: "voice_mode", value: "none" },
    { name: "postprocess_mode", value: "video_sr" },
    { name: "duration", value: 15 },
    { name: "aspect_ratio", value: "16:9" },
    { name: "resolution_preset", value: "768p H3" },
  ]);
  const text = allElements(node.root).map(el => el.textContent || "").join(" ");
  assert.ok(text.includes("实际 H3 采样 704×384"));
  assert.ok(text.includes("输出目标 1344×768"));
  assert.ok(text.includes("放大无法准确补回底图缺失的五官细节"));
});


function serverUploadHelpers() {
  return ["first_image_file", "last_image_file",
    ...Array.from({ length: 9 }, (_, i) => `reference_image_${i + 1}_file`)]
    .flatMap(name => ["input", "output"].map(direction => ({
      name: `${direction}:${name}`, value: `keep-${name}`,
      computeSize: () => [200, 20], draw() {},
    })));
}

test("server image helper rows collapse without dropping uploads or unrelated controls", () => {
  const helpers = serverUploadHelpers();
  const unrelated = { name: "input:unrelated_file", value: "custom.png", computeSize: () => [200, 20] };
  const upload = { name: "reference_image_1_file", value: "user.png", options: { values: ["user.png"] } };
  const { node } = installedDirector("off", [upload, unrelated, ...helpers]);
  assert.equal(helpers.length, 22);
  for (const helper of helpers) {
    assert.equal(helper.hidden, true, helper.name);
    assert.ok(helper.computeSize()[1] <= 0, helper.name);
    assert.ok(node.widgets.includes(helper));
    assert.ok(helper.value.startsWith("keep-"));
  }
  assert.equal(upload.value, "user.png");
  assert.equal(unrelated.hidden, undefined);
  assert.equal(unrelated.computeSize()[1], 20);
});

test("server helpers added after node creation and during restore are hidden on mount and render", () => {
  const { node, frames } = installedDirector();
  const late = serverUploadHelpers();
  node.widgets.push(...late);
  // Host extensions can create their widgets after DirectorPlus.nodeCreated.
  const mountingFrames = frames.splice(0);
  for (const callback of mountingFrames) callback();
  for (const helper of late) assert.equal(helper.hidden, true, helper.name);
  const restored = { name: "input:reference_image_9_file", value: "restored.png" };
  node.widgets.push(restored);
  node.onConfigure({ face_refine_mode: "off" });
  for (const callback of frames.splice(0)) callback();
  assert.equal(restored.hidden, true);
  assert.equal(restored.value, "restored.png");
});


// Native frontend 1.52 saves values at widget indices, but restores them with
// a compact cursor that skips serialize=false previews. Reproduce both sides.
const STATE_FIELDS = [
  "mode", "prompt", "duration", "width", "height", "aspect_ratio", "resolution_preset",
  "custom_width", "custom_height", "seed", "control_after_generate", "voice_mode",
  "fish_model_path", "ref_image_size", "performance_preset", "postprocess_mode",
  "rtx_quality", "ai_upscale_model", "timeline_data", "target_dialogue", "reference_transcript",
  "voice_reference_name_1", "voice_reference_name_2", "voice_reference_name_3",
  "first_image_file", "last_image_file", "voice_reference_audio_file",
  "voice_reference_audio_2_file", "voice_reference_audio_3_file",
  ...Array.from({ length: 9 }, (_, i) => "reference_image_" + (i + 1) + "_file"),
  "motion_smoothing", "audio_loudness", "voice_gender", "face_refine_mode",
  "sampling_method", "community_lora_profile", "vosr2_profile",
];
function nativeWidgets() {
  const values = {
    mode: "REF2VA", prompt: "保留用户提示词 <Picture 1>", duration: 15,
    width: 1920, height: 1080, aspect_ratio: "16:9", resolution_preset: "1080p FHD",
    custom_width: 16, custom_height: 9, seed: 1015586840945481,
    control_after_generate: "randomize", voice_mode: "h3_reference", fish_model_path: "w4a16",
    ref_image_size: "max", performance_preset: "质量优先加速", postprocess_mode: "vosr2",
    rtx_quality: "ULTRA", ai_upscale_model: "auto", motion_smoothing: "off",
    audio_loudness: "auto", voice_gender: "auto", face_refine_mode: "off",
    sampling_method: "standard", community_lora_profile: "off", vosr2_profile: "legacy",
    first_image_file: "input/角色.png", last_image_file: "input/衣服.png",
    reference_image_1_file: "input/房间.png", reference_image_2_file: "input/道具.png",
    reference_image_3_file: "input/锁.png",
    voice_reference_audio_file: "input/南宫阙.wav", voice_reference_name_1: "南宫阙",
    voice_reference_audio_2_file: "input/南宫婉.wav", voice_reference_name_2: "南宫婉",
  };
  return STATE_FIELDS.flatMap(name => {
    const item = { name, value: values[name] ?? "", options: { values: ["", values[name] ?? ""] } };
    return /^(first_image|last_image|reference_image_\d+)_file$/.test(name)
      ? [item, ...["input", "output"].map(prefix => ({ name: prefix + ":" + name, serialize: false }))]
      : [item];
  });
}
function nativeSave(node, invokeHook = true, nativeHook = false) {
  if (invokeHook && typeof node.serialize === "function") {
    return JSON.parse(JSON.stringify(node.serialize()));
  }
  const data = { id: 2693, type: "MiniMaxH3DirectorPlus", properties: {}, widgets_values: [], widgets_values_named: {} };
  node.widgets.forEach((item, index) => {
    if (item.serialize === false) return;
    const value = item.value ?? null;
    data.widgets_values[index] = value;
    data.widgets_values_named[item.name] = value;
  });
  if (invokeHook || nativeHook) node.onSerialize?.(data);
  return JSON.parse(JSON.stringify(data));
}
function legacySave() {
  return nativeSave({ widgets: [...nativeWidgets(), { name: "minimax_h3_director_plus_ui", value: "" }] }, false);
}

test("old sparse saves without trailing VOSR control migrate to legacy profile", () => {
  const old = nativeSave({ widgets: nativeWidgets().filter(item => item.name !== "vosr2_profile") }, false);
  const { node, extension } = installedDirector("off", nativeWidgets(), true);
  extension.beforeConfigureGraph({ nodes: [old] });
  nativeRestore(node, old);
  assert.equal(valueOf(node, "vosr2_profile"), "legacy");
  assert.equal(valueOf(node, "first_image_file"), "input/角色.png");
  assert.equal(valueOf(node, "voice_reference_audio_file"), "input/南宫阙.wav");
});

for (const voiceMode of ["none", "h3_reference", "fish_lock", "audio_reuse"]) {
  test(`balanced eight-step survives route render with ${voiceMode}`, () => {
    const preset = { name: "performance_preset", value: "均衡高质（768p 8步）" };
    const { node } = installedDirector("off", [preset,
      { name: "voice_mode", value: voiceMode }, { name: "postprocess_mode", value: "vosr2" },
      { name: "vosr2_profile", value: "manual_baseline" },
    ]);
    for (const mode of ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"]) {
      node.widgets.find(item => item.name === "mode").value = mode;
      node.widgets.find(item => item.name === "mode").callback(mode);
      assert.equal(preset.value, "均衡高质（768p 8步）");
      assert.equal(valueOf(node, "vosr2_profile"), "manual_baseline");
    }
  });
}
function nativeRestore(node, data) {
  let cursor = 0;
  for (const item of node.widgets) {
    if (item.serialize !== false && cursor < data.widgets_values.length) item.value = data.widgets_values[cursor++];
  }
  node.onConfigure(data);
}
function valueOf(node, name) { return node.widgets.find(item => item.name === name)?.value; }
function clickRemove(node, field) {
  const button = allElements(node.root).find(el => el.dataset?.h3pRemoveFile === field);
  assert.ok(button, "missing remove button for " + field);
  const event = { target: { closest: () => button }, preventDefault() {}, stopPropagation() {} };
  for (const handler of node.root.listeners.click || []) handler(event);
}

test("native sparse preview save recovers all image/audio values without shifting settings", () => {
  const legacy = legacySave();
  assert.equal(legacy.widgets_values.length, 68);
  assert.equal(legacy.widgets_values[39], "input/锁.png");
  const { node, extension } = installedDirector("off", nativeWidgets(), true);
  extension.beforeConfigureGraph({ nodes: [legacy] });
  nativeRestore(node, legacy);
  for (const name of STATE_FIELDS) assert.equal(valueOf(node, name), legacy.widgets_values_named[name], name);
  assert.equal(valueOf(node, "audio_loudness"), "auto");
  assert.equal(valueOf(node, "voice_reference_audio_file"), "input/南宫阙.wav");
});

test("saving excludes all preview/DOM helpers and keeps exact native control order", () => {
  const { node } = installedDirector("off", nativeWidgets(), true);
  const saved = nativeSave(node);
  assert.equal(saved.widgets_values.length, STATE_FIELDS.length);
  assert.deepEqual(Object.keys(saved.widgets_values_named), STATE_FIELDS);
  for (const [index, name] of STATE_FIELDS.entries()) assert.equal(saved.widgets_values[index], valueOf(node, name));
  assert.equal(node.widgets.find(item => item.name === "minimax_h3_director_plus_ui").serialize, false);
});

test("removing a reloaded middle image compacts references and stays removed after save/reload", () => {
  const { node, extension } = installedDirector("off", nativeWidgets(), true);
  const legacy = legacySave();
  extension.beforeConfigureGraph({ nodes: [legacy] });
  nativeRestore(node, legacy);
  node.widgets.find(item => item.name === "mode").callback("REF2VA");
  clickRemove(node, "last_image_file");
  assert.equal(valueOf(node, "last_image_file"), "input/房间.png");
  assert.equal(valueOf(node, "reference_image_1_file"), "input/道具.png");
  assert.equal(valueOf(node, "reference_image_2_file"), "input/锁.png");
  assert.equal(valueOf(node, "reference_image_3_file"), "");
  const saved = nativeSave(node);
  const restored = installedDirector("off", nativeWidgets(), true);
  restored.extension.beforeConfigureGraph({ nodes: [saved] });
  nativeRestore(restored.node, saved);
  assert.equal(valueOf(restored.node, "last_image_file"), "input/房间.png");
  assert.equal(valueOf(restored.node, "reference_image_1_file"), "input/道具.png");
  assert.ok(!saved.widgets_values.includes("input/衣服.png"));
  assert.equal(valueOf(restored.node, "audio_loudness"), "auto");
});

for (const dynamic of [false, true]) {
  test("saved image absent from " + (dynamic ? "dynamic" : "cached") + " choices remains visible and removable", () => {
    const selected = { name: "first_image_file", value: "input/保留.png", options: { values: dynamic ? () => [""] : [""] } };
    const { node } = installedDirector("off", [selected]);
    node.widgets.find(item => item.name === "mode").value = "I2VA";
    node.widgets.find(item => item.name === "mode").callback("I2VA");
    assert.equal(selected.value, "input/保留.png");
    const choices = typeof selected.options.values === "function" ? selected.options.values() : selected.options.values;
    assert.ok(choices.includes(selected.value));
    assert.ok(allElements(node.root).some(el => el.tag === "img" && el.src.includes("%E4%BF%9D%E7%95%99.png")));
    const preview = allElements(node.root).find(el => el.tag === "img");
    for (const handler of preview.listeners.error || []) handler();
    assert.equal(preview.hidden, true);
    clickRemove(node, "first_image_file");
    assert.equal(selected.value, "");
    assert.ok(!allElements(node.root).some(el => el.dataset?.h3pRemoveFile === "first_image_file"));
  });
}

test("old compact controls override stale named metadata in main graph and subgraphs", () => {
  const { extension } = installedDirector();
  const full = legacySave();
  const compact = STATE_FIELDS.map(name => full.widgets_values_named[name]);
  const old = { ...full, widgets_values: compact.slice(0, 42), widgets_values_named: { ...full.widgets_values_named, first_image_file: "stale.png" } };
  const nested = structuredClone(old);
  nested.id = 8;
  const other = { id: 1, type: "Other", widgets_values: [null, "untouched"], pos: [100, 200] };
  const graph = { nodes: [old, other], definitions: { subgraphs: [{ nodes: [nested], links: [] }] } };
  extension.beforeConfigureGraph(graph);
  for (const data of [old, nested]) {
    assert.equal(data.widgets_values_named.first_image_file, "input/角色.png");
    assert.equal(data.widgets_values.length, STATE_FIELDS.length);
    assert.equal(data.widgets_values_named.sampling_method, "standard");
    assert.equal(data.widgets_values_named.community_lora_profile, "off");
  }
  assert.deepEqual(other, { id: 1, type: "Other", widgets_values: [null, "untouched"], pos: [100, 200] });
});

test("onConfigure recovers named legacy values even without graph pre-configuration hook", () => {
  const { node } = installedDirector("off", nativeWidgets(), true);
  nativeRestore(node, legacySave());
  assert.equal(valueOf(node, "reference_image_1_file"), "input/房间.png");
  assert.equal(valueOf(node, "audio_loudness"), "auto");
  assert.equal(valueOf(node, "seed"), 1015586840945481);
});

test("replacing a restored image through upload survives another save and reload", async () => {
  const api = { async fetchApi() { return { ok: true, async json() { return { ok: true, asset: "input/替换.png" }; } }; } };
  const { node, extension } = installedDirector("off", nativeWidgets(), true, api);
  const data = legacySave();
  extension.beforeConfigureGraph({ nodes: [data] });
  nativeRestore(node, data);
  node.widgets.find(item => item.name === "mode").callback("REF2VA");
  const input = allElements(node.root).find(el => el.dataset?.h3pUploadFile === "last_image_file");
  input.files = [new Blob(["image"], { type: "image/png" })];
  input.files[0].name = "替换.png";
  input.closest = () => null;
  const event = { target: { closest: selector => selector === ".h3p input[type=file]" ? input : null }, stopPropagation() {} };
  for (const handler of node.root.listeners.change || []) handler(event);
  await new Promise(setImmediate);
  assert.equal(valueOf(node, "last_image_file"), "input/替换.png");
  const restored = installedDirector("off", nativeWidgets(), true);
  const saved = nativeSave(node);
  restored.extension.beforeConfigureGraph({ nodes: [saved] });
  nativeRestore(restored.node, saved);
  assert.equal(valueOf(restored.node, "last_image_file"), "input/替换.png");
  assert.equal(valueOf(restored.node, "first_image_file"), "input/角色.png");
  assert.equal(valueOf(restored.node, "voice_reference_audio_file"), "input/南宫阙.wav");
});

test("native Vue overlay receives hidden options for controls and late upload helpers", () => {
  const widgets = nativeWidgets();
  const { node, frames } = installedDirector("off", widgets, true);
  const late = serverUploadHelpers();
  node.widgets.push(...late);
  for (const callback of frames.splice(0)) callback();
  for (const item of [...widgets, ...late]) {
    if (item.name === "minimax_h3_director_plus_ui") continue;
    // Frontend 1.52.7 extractWidgetDisplayOptions reads options.hidden.
    // Its Vue layer does not read the canvas-only item.hidden flag.
    assert.equal(item.options?.hidden, true, item.name);
  }
  const panel = node.widgets.find(item => item.name === "minimax_h3_director_plus_ui");
  assert.notEqual(panel.options.hidden, true);
});

for (const previous of ["", null, undefined, "input/房间.png"]) {
  test("invalid audio loudness " + String(previous) + " is repaired before queue/save and notifies host validation", () => {
    const audio = { name: "audio_loudness", value: previous, options: { values: ["auto", "original"] } };
    const { node, changes } = installedDirector("off", [audio]);
    assert.equal(audio.value, "auto");
    const change = changes.find(change => change.name === "audio_loudness");
    assert.ok(change);
    assert.equal(change.previous, previous);
    assert.equal(change.value, "auto");
    assert.equal(change.item, audio);
    assert.equal(node.onSerialize ? nativeSave(node).widgets_values_named.audio_loudness : audio.value, "auto");
  });
}

test("configured valid loudness is published to clear errors retained by the host", () => {
  const { node, changes } = installedDirector("off", nativeWidgets(), true);
  changes.length = 0;
  const saved = nativeSave(node);
  nativeRestore(node, saved);
  assert.equal(valueOf(node, "audio_loudness"), "auto");
  const change = changes.find(change => change.name === "audio_loudness");
  assert.ok(change);
  assert.equal(change.value, "auto");
  assert.equal(change.previous, "auto");
});

test("loudness UI edits notify native validation without changing a valid original setting", () => {
  const audio = { name: "audio_loudness", value: "original", options: { values: ["auto", "original"] } };
  const { node, changes } = installedDirector("off", [audio]);
  assert.equal(audio.value, "original");
  const control = allElements(node.root).find(el => el.dataset?.h3pValueWidget === "audio_loudness");
  control.value = "auto";
  const event = {
    target: { closest: selector => selector === "[data-h3p-value-widget]" ? control : null },
    stopPropagation() {},
  };
  for (const listener of node.root.listeners.change || []) listener(event);
  assert.equal(audio.value, "auto");
  const change = changes.find(change => change.name === "audio_loudness");
  assert.ok(change);
  assert.equal(change.previous, "original");
  assert.equal(change.value, "auto");
});

test("removing an image notifies host input validation with the cleared filename", () => {
  const selected = { name: "first_image_file", value: "input/原图.png", options: { values: ["", "input/原图.png"] } };
  const { node, changes } = installedDirector("off", [selected]);
  const mode = node.widgets.find(item => item.name === "mode");
  mode.value = "I2VA";
  mode.callback("I2VA");
  clickRemove(node, "first_image_file");
  const change = changes.find(change => change.name === "first_image_file");
  assert.ok(change);
  assert.equal(change.previous, "input/原图.png");
  assert.equal(change.value, "");
});

test("compact browser draft repairs empty audio before queue while retaining all current images", () => {
  const saved = legacySave();
  saved.widgets_values = STATE_FIELDS.map(name => saved.widgets_values_named[name]);
  saved.widgets_values[STATE_FIELDS.indexOf("audio_loudness")] = "";
  saved.widgets_values_named.audio_loudness = "";
  const { node, extension, changes } = installedDirector("off", nativeWidgets(), true);
  extension.beforeConfigureGraph({ nodes: [saved] });
  nativeRestore(node, saved);
  assert.equal(valueOf(node, "audio_loudness"), "auto");
  assert.ok(changes.some(change => change.name === "audio_loudness" && change.value === "auto"));
  const repaired = nativeSave(node);
  assert.equal(repaired.widgets_values_named.audio_loudness, "auto");
  assert.equal(repaired.widgets_values[39], "auto");
  for (const name of STATE_FIELDS.filter(name => name.endsWith("_file"))) {
    assert.equal(valueOf(node, name), saved.widgets_values_named[name], name);
  }
});

test("native serialize stays compact when a later extension replaces onSerialize", () => {
  const { node } = installedDirector("off", nativeWidgets(), true);
  let calls = 0;
  node.onSerialize = data => {
    calls++;
    data.extension_metadata = { preserved: true };
    data.widgets_values.push(null);
    data.widgets_values_named.audio_loudness = "";
  };
  const saved = nativeSave(node);
  assert.equal(calls, 1);
  assert.equal(saved.widgets_values.length, STATE_FIELDS.length);
  assert.equal(saved.widgets_values_named.audio_loudness, "auto");
  assert.deepEqual(saved.extension_metadata, { preserved: true });
  const restored = installedDirector("off", nativeWidgets(), true);
  restored.extension.beforeConfigureGraph({ nodes: [saved] });
  nativeRestore(restored.node, saved);
  assert.deepEqual(nativeSave(restored.node).widgets_values, saved.widgets_values);
});

for (const raw of ["", null, undefined, "invalid", "original"]) {
  test("graph normalization repairs audio before native validation: " + String(raw), () => {
    const { extension } = installedDirector();
    const saved = legacySave();
    saved.widgets_values_named.audio_loudness = raw;
    // Native validation reads the JSON values before onConfigure can run.
    extension.beforeConfigureGraph({ nodes: [saved] });
    const expected = raw === "original" ? "original" : "auto";
    assert.equal(saved.widgets_values[39], expected);
    assert.equal(saved.widgets_values_named.audio_loudness, expected);
    for (const field of STATE_FIELDS.filter(field => field.endsWith("_file"))) {
      assert.equal(saved.widgets_values[STATE_FIELDS.indexOf(field)], saved.widgets_values_named[field]);
    }
  });
}

for (const field of ["reference_image_8_file", "reference_image_9_file"]) {
  test("retained legacy reference is visible and removable after reload: " + field, () => {
    const saved = legacySave();
    saved.widgets_values_named[field] = "input/旧故事板.png";
    const { node, extension, frames } = installedDirector("off", nativeWidgets(), true);
    extension.beforeConfigureGraph({ nodes: [saved] });
    nativeRestore(node, saved);
    for (const frame of frames.splice(0)) frame();
    assert.equal(valueOf(node, field), "input/旧故事板.png");
    assert.ok(allElements(node.root).some(el => el.tag === "img" && el.src.includes("%E6%97%A7%E6%95%85%E4%BA%8B%E6%9D%BF.png")));
    clickRemove(node, field);
    const exported = nativeSave(node);
    assert.equal(exported.widgets_values_named[field], "");
    assert.equal(exported.widgets_values_named.first_image_file, "input/角色.png");
    assert.equal(exported.widgets_values_named.reference_image_3_file, "input/锁.png");
    assert.ok(!exported.widgets_values.includes("input/旧故事板.png"));
    const restored = installedDirector("off", nativeWidgets(), true);
    restored.extension.beforeConfigureGraph({ nodes: [exported] });
    nativeRestore(restored.node, exported);
    assert.equal(valueOf(restored.node, field), "");
    assert.ok(!allElements(restored.node.root).some(el => el.dataset?.h3pRemoveFile === field));
  });
}

test("shipped U11 example uses the current control order through two native save cycles", () => {
  const workflow = JSON.parse(readFileSync(new URL("../../examples/U11-MiniMaxH3-导演台Plus-中文增强版-AutoDL混合底模版-清晰度增强版.json", import.meta.url), "utf8"));
  const saved = workflow.nodes.find(node => node.type === "MiniMaxH3DirectorPlus");
  const { extension: migration } = installedDirector();
  migration.beforeConfigureGraph(workflow);
  assert.equal(saved.widgets_values.length, STATE_FIELDS.length);
  assert.deepEqual(Object.keys(saved.widgets_values_named), STATE_FIELDS);
  assert.deepEqual(saved.widgets_values, STATE_FIELDS.map(field => saved.widgets_values_named[field]));
  let data = saved;
  for (let cycle = 0; cycle < 2; cycle++) {
    const { node, extension } = installedDirector("off", nativeWidgets(), true);
    extension.beforeConfigureGraph({ nodes: [data] });
    nativeRestore(node, data);
    data = nativeSave(node);
  }
  for (const field of ["prompt", "seed", "control_after_generate", "audio_loudness", ...STATE_FIELDS.filter(field => field.endsWith("_file"))]) {
    assert.equal(data.widgets_values_named[field], saved.widgets_values_named[field], field);
  }
});
