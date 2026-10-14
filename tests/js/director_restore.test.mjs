import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

// Exercise the actual extension's install/configure/render lifecycle without
// ComfyUI or a GPU. Only the DOM and host services are stubbed.
class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.style = { setProperty() {} };
    this.classList = { add() {}, remove() {}, toggle() {} };
  }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.append(child); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this[name] = value; }
  addEventListener() {}
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

function installedDirector(initialValue = "off", extraWidgets = []) {
  let extension;
  const frames = [];
  const context = vm.createContext({
    app: { registerExtension(value) { extension = value; } },
    api: {},
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
  const source = readFileSync(new URL("../../js/minimax_h3_director_plus_v9.js", import.meta.url), "utf8");
  vm.runInContext(source.replace(/^import .*;\r?\n/gm, ""), context);
  const face = { name: "face_refine_mode", value: initialValue, options: { values: ["off", "auto"] } };
  const node = {
    comfyClass: "MiniMaxH3DirectorPlus",
    widgets: [face, { name: "mode", value: "T2VA" }, ...extraWidgets],
    size: [1350, 1760],
    setSize(value) { this.size = value; },
    graph: { setDirtyCanvas() {} },
    // Simulates the host's restoration callback, before our wrapper runs.
    onConfigure(data) { face.value = data.face_refine_mode; },
    addDOMWidget(name, type, element) {
      this.root = element;
      return { element };
    },
  };
  extension.nodeCreated(node);
  return { node, face, frames, extension };
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
