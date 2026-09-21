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
  setAttribute() {}
  addEventListener() {}
  querySelector() { return null; }
  querySelectorAll() { return []; }
}

function installedDirector(initialValue = "off") {
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
    widgets: [face, { name: "mode", value: "T2VA" }],
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
  return { node, face, frames };
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

test("restoring an enabled face mode preserves the user's selection", () => {
  const { node, face } = installedDirector();
  node.onConfigure({ face_refine_mode: "auto" });
  assert.equal(face.value, "auto");
  face.callback("auto");
  assert.equal(face.value, "auto");
});

test("unexpected nonempty face modes remain visible to validation", () => {
  const { node, face } = installedDirector();
  node.onConfigure({ face_refine_mode: "not-a-mode" });
  assert.equal(face.value, "not-a-mode");
});
