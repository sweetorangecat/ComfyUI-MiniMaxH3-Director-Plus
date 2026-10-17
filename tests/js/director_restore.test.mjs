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
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  set innerHTML(value) {
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
      addEventListener() {},
      querySelectorAll: () => [],
    },
    window: {},
    MutationObserver: class { observe() {} disconnect() {} },
    requestAnimationFrame: (fn) => frames.push(fn),
    setTimeout() {},
  });
  const source = readFileSync(new URL("../../js/minimax_h3_director_plus_v10.js", import.meta.url), "utf8");
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
  return { node, face, frames, extension, changes };
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

function advancedPanel(node) {
  return allElements(node.root).find(el => el.tag === "details"
    && el.children.some(child => child.textContent === "二采高级设置"));
}

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
  "sampling_method", "community_lora_profile",
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
    sampling_method: "standard", community_lora_profile: "off",
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
  assert.equal(legacy.widgets_values.length, 67);
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
    assert.equal(data.widgets_values.length, 44);
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
  assert.equal(saved.widgets_values.length, 44);
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
