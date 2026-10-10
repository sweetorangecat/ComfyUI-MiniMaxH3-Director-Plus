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
  querySelector(selector) { return this.children.find(child => child.tag === selector) || null; }
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
