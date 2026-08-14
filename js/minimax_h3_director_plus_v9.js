/**
 * MiniMax H3 Director Plus UI.
 * Adapted from the visual language of DaSiWa MiniMax H3 Director under GPL-3.0.
 * This implementation is independent and intentionally exposes only H3 reference semantics.
 */
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS = "MiniMaxH3DirectorPlus";
// Keep node geometry independent from the browser sidebar width.
const DIRECTOR_UI_WIDTH = 1350;
const DIRECTOR_UI_HEIGHT = 1510;
const DIRECTOR_DOM_HEIGHT = 1050;
const MODES = ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"];
const PRESETS = ["稳定质量", "极速4步", "参考图加速", "低显存", "自定义"];
const PERFORMANCE_PRESETS_BY_ROUTE = {
  t2va: ["稳定质量", "低显存"],
  endpoint: ["稳定质量", "极速4步", "低显存"],
  reference: ["稳定质量", "参考图加速", "极速4步", "低显存"],
};
const ASPECTS = {
  "1:1": [1, 1], "3:2": [3, 2], "2:3": [2, 3], "4:3": [4, 3], "3:4": [3, 4],
  "8:5": [8, 5], "5:8": [5, 8], "16:9": [16, 9], "9:16": [9, 16], "21:9": [21, 9], "9:21": [9, 21],
};
const RESOLUTIONS = [
  "0.26 MP", "0.30 MP", "0.36 MP", "0.40 MP", "0.50 MP", "0.52 MP", "0.60 MP", "0.65 MP", "0.70 MP",
  "0.80 MP", "0.83 MP", "0.90 MP", "1.00 MP", "1.05 MP", "1.10 MP", "1.20 MP", "1.30 MP", "1.35 MP",
  "1.40 MP", "1.50 MP", "1.55 MP", "1.60 MP", "1.65 MP", "1.70 MP", "1.75 MP", "1.80 MP", "1.90 MP",
  "2.00 MP", "2.10 MP",
];
const SEED_MODES = [
  ["fixed", "固定"],
  ["increment", "每次递增"],
  ["decrement", "每次递减"],
  ["randomize", "每次随机"],
];
const REF2VA_IMAGE_SLOTS = [
  ["参考图 1（起始构图建议）", "first_image_file"],
  ["参考图 2（结束构图建议）", "last_image_file"],
  ["参考图 3", "reference_image_1_file"], ["参考图 4", "reference_image_2_file"],
  ["参考图 5", "reference_image_3_file"], ["参考图 6", "reference_image_4_file"],
  ["参考图 7", "reference_image_5_file"], ["参考图 8", "reference_image_6_file"],
  ["参考图 9", "reference_image_7_file"],
];
const VOICE_MODES = [
  ["none", "不使用音色"],
  ["h3_reference", "H3 原生音色参考"],
  ["fish_lock", "Fish S2 高级锁定"],
];
const VOICE_MODE_LABELS = {none: "不使用音色", h3_reference: "H3 原生音色参考", fish_lock: "Fish S2 高级锁定"};
const VOICE_REFERENCE_LABELS = ["音色参考 1", "音色参考 2", "音色参考 3"];
const AUDIO_REFERENCE_HINT = "<Audio 1> / <Audio 2> / <Audio 3>";
const FISH_MODELS = [
  ["s2-pro-w4a16 (auto download)", "S2 Pro 量化版（约 8GB 显存）"],
  ["s2-pro (auto download)", "S2 Pro 完整版（约 24GB 显存）"],
];

let stylesInstalled = false;
let nextControllerId = 0;
const controllers = new Map();
let documentControlsBound = false;

function installStyles() {
  if (stylesInstalled) return;
  stylesInstalled = true;
  const style = document.createElement("style");
  style.textContent = `
    .h3p{box-sizing:border-box;width:${DIRECTOR_UI_WIDTH}px;min-width:${DIRECTOR_UI_WIDTH}px;max-width:none;flex:0 0 ${DIRECTOR_UI_WIDTH}px;padding:8px;background:#0f151b;color:#d9e4eb;font:12px system-ui,sans-serif;display:flex;flex-direction:column;gap:8px}
    .h3p *{box-sizing:border-box;letter-spacing:0}.h3p-head{display:flex;align-items:center;justify-content:space-between;gap:10px}.h3p-title{font-size:14px;font-weight:700;color:#fff}.h3p-badge{padding:2px 7px;border:1px solid #4d6372;border-radius:4px;color:#9fc5d8;background:#17232c}
    .h3p-section{border-top:1px solid #2d3b46;padding-top:8px}.h3p-section-title{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;color:#fff;font-weight:650}.h3p-hint{color:#8fa5b4;font-weight:400;font-size:11px}
    .h3p-segments{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:4px}.h3p-segments.preset{grid-template-columns:repeat(5,minmax(0,1fr))}.h3p button,.h3p-segment{min-height:29px;padding:4px 6px;border:1px solid #3c4d59;border-radius:4px;background:#18232c;color:#aebfca;cursor:pointer;text-align:center;display:flex;align-items:center;justify-content:center}.h3p button:hover,.h3p-segment:hover{background:#22323e;color:#fff}.h3p-segment.active{border-color:#78bce0;background:#17384a;color:#fff;box-shadow:inset 0 0 0 1px rgba(120,188,224,.2)}.h3p-segment input{position:absolute;opacity:0;pointer-events:none;width:1px;height:1px}
    .h3p-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}.h3p-field{display:flex;flex-direction:column;gap:4px;min-width:0}.h3p label{color:#9fb2bf}.h3p select,.h3p input,.h3p textarea{width:100%;max-width:100%;min-width:0;box-sizing:border-box;border:1px solid #3c4d59;border-radius:4px;background:#0b1116;color:#e6edf2;padding:6px}.h3p textarea{min-height:104px;resize:vertical;line-height:1.5}.h3p-readonly{min-height:30px;padding:6px;border:1px solid #344956;border-radius:4px;background:#111d24;color:#8fd2f3}.h3p-spec-note{margin-top:5px;color:#8298a6;font-size:11px}
    .h3p-prompt-tools{margin-top:6px;padding:7px;border:1px solid #344956;border-radius:4px;background:#111d24}.h3p-prompt-tools-title{display:flex;align-items:center;justify-content:space-between;gap:8px;color:#fff;font-weight:650}.h3p-prompt-tools-hint{color:#8fa5b4;font-weight:400;font-size:11px}.h3p-prompt-actions{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}.h3p-prompt-actions button{min-height:26px;padding:3px 8px}.h3p-prompt-help{margin-top:6px;color:#9fb2bf;font-size:11px;line-height:1.45}
    .h3p-route{display:grid;grid-template-columns:auto 1fr;gap:5px 10px;padding:7px;border:1px solid #344956;border-radius:4px;background:#111d24}.h3p-route strong{color:#8fd2f3}.h3p-route .warning{grid-column:1/-1;color:#e9c176;line-height:1.45}.h3p-route .ok{grid-column:1/-1;color:#93cda8;line-height:1.45}
    .h3p-voice{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px}.h3p-audio-lane{margin-top:6px;padding:7px;border:1px dashed #4e7182;border-radius:4px;background:#101b21;color:#adc2ce;line-height:1.45}.h3p-audio-lane b{color:#dfeaf0}.h3p-audio-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin-top:7px}.h3p-voice-slot{display:flex;flex-direction:column;gap:5px;min-width:0}.h3p-fish{margin-top:6px;border:1px solid #3b4a55;border-radius:4px;background:#111920}.h3p-fish summary{padding:7px;cursor:pointer;color:#d5dee4;font-weight:600}.h3p-fish-body{padding:0 7px 7px;color:#99aebc;line-height:1.5}
    .h3p-materials{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}.h3p-material{min-height:52px;padding:7px;border:1px solid #344956;border-radius:4px;background:#111a20}.h3p-material b{display:block;color:#e0e8ed;margin-bottom:3px}.h3p-material span{color:#8298a6;font-size:11px;line-height:1.35}.h3p-upload-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin-top:6px}.h3p-upload{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:6px;padding:7px;border:1px dashed #4e7182;border-radius:4px;background:#101b21;color:#adc2ce}.h3p-upload>span{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.h3p-upload input{display:none}.h3p-upload button{min-height:25px;padding:3px 8px}.h3p-upload.busy{opacity:.6;pointer-events:none}.h3p-upload.error{border-color:#d87979;color:#ffb1b1}.h3p-media-preview{grid-column:1/-1;min-width:0}.h3p-image-link{display:flex;width:100%;height:132px;border:1px solid #2f424e;border-radius:3px;overflow:hidden;background:#0b1116;align-items:center;justify-content:center}.h3p-image-preview{display:block;width:100%;height:100%;object-fit:contain}.h3p-audio-preview{display:block;width:100%;height:34px}.h3p-preview-status{display:block;color:#e9c176;font-size:11px;padding:4px 0}
    @media(max-width:640px){.h3p-segments,.h3p-segments.preset{grid-template-columns:repeat(3,minmax(0,1fr))}.h3p-grid,.h3p-materials,.h3p-audio-grid{grid-template-columns:1fr}.h3p-voice{grid-template-columns:1fr}}
  `;
  document.head.appendChild(style);
}

function widget(node, name) {
  return node.widgets?.find((item) => item.name === name);
}

function hideWidget(item) {
  if (!item) return;
  item.hidden = true;
  item.computeSize = () => [0, -4];
  item.draw = () => {};
}

function setWidget(node, name, value, notify = true) {
  const item = widget(node, name);
  if (!item || item.value === value) return;
  item.value = value;
  if (notify) item.callback?.(value);
  node.graph?.setDirtyCanvas(true, true);
}

function allowedPerformancePresets(mode, voiceMode) {
  if (voiceMode !== "none" || mode === "REF2VA") return PERFORMANCE_PRESETS_BY_ROUTE.reference;
  if (mode === "T2VA") return PERFORMANCE_PRESETS_BY_ROUTE.t2va;
  return PERFORMANCE_PRESETS_BY_ROUTE.endpoint;
}

function keepPromptAssistantReadable(anchor) {
  if (!anchor || anchor._h3pAssistantScaleLoop) return;
  anchor._h3pAssistantScaleLoop = true;

  // LiteGraph scales the entire DOM widget with the canvas. The native
  // Prompt Assistant is mounted inside that tree, so compensate only its
  // own UI to keep the icon and toolbar at a usable screen size.
  const update = () => {
    const assistant = anchor.querySelector?.(".prompt-assistant-container");
    const canvasScale = Number(app.canvas?.ds?.scale) || 1;
    if (assistant) {
      assistant.style.setProperty("--assistant-scale", String(1 / canvasScale));
    }
    if (!document.body.contains(anchor)) {
      anchor._h3pAssistantScaleLoop = false;
      return false;
    }
    return true;
  };

  const tick = () => {
    if (update()) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function syncUploadWidget(node, widgetName, value) {
  const item = widget(node, widgetName);
  if (!item || !value) return;
  const values = item.options?.values;
  if (Array.isArray(values) && !values.includes(value)) item.options.values.push(value);
  item.value = value;
  item.callback?.(value);
  node.graph?.setDirtyCanvas(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function controlButton(label, action, value, current) {
  const segment = document.createElement("label");
  segment.className = "h3p-segment";
  segment.classList.toggle("active", value === current);
  const input = document.createElement("input");
  input.type = "radio";
  input.name = action;
  input.value = value;
  input.checked = value === current;
  input.dataset.h3pValueWidget = action;
  const caption = document.createElement("span");
  caption.textContent = label;
  segment.append(input, caption);
  return segment;
}

function valueControl(label, widgetName, options, current, type = "select") {
  const field = document.createElement("label");
  field.className = "h3p-field";
  const caption = document.createElement("span");
  caption.textContent = label;
  const input = document.createElement(type === "select" ? "select" : type === "textarea" ? "textarea" : "input");
  input.setAttribute("aria-label", label);
  input.dataset.h3pValueWidget = widgetName;
  if (type === "select") {
    options.forEach((raw) => {
      const [value, display] = Array.isArray(raw) ? raw : [raw, raw];
      const option = document.createElement("option");
      option.value = value;
      option.textContent = display;
      option.selected = value === String(current);
      input.append(option);
    });
  } else if (type === "textarea") {
    input.value = current ?? "";
  } else {
    input.type = type;
    input.value = current ?? "";
    if (type === "number") {
      input.min = widgetName === "duration" ? "4" : widgetName === "seed" ? "0" : "1";
      input.max = widgetName === "duration" ? "15" : widgetName === "seed" ? "18446744073709551615" : "8192";
      input.step = "1";
      input.dataset.h3pNumber = "true";
    }
  }
  field.append(caption, input);
  return field;
}

function calculatedResolution(node) {
  const preset = String(widget(node, "resolution_preset")?.value || "0.83 MP");
  const aspect = String(widget(node, "aspect_ratio")?.value || "16:9");
  const megapixels = Number.parseFloat(preset) || 0.83;
  const ratio = aspect === "CUSTOM"
    ? [Math.max(1, Number(widget(node, "custom_width")?.value) || 16), Math.max(1, Number(widget(node, "custom_height")?.value) || 9)]
    : (ASPECTS[aspect] || ASPECTS["16:9"]);
  const area = megapixels * 1024 * 1024;
  const snap = (value) => Math.max(32, Math.round(value / 32) * 32);
  return [snap(Math.sqrt(area * ratio[0] / ratio[1])), snap(Math.sqrt(area * ratio[1] / ratio[0]))];
}

function syncResolution(node) {
  const [width, height] = calculatedResolution(node);
  setWidget(node, "width", width);
  setWidget(node, "height", height);
  return [width, height];
}

function material(title, description) {
  const box = document.createElement("div");
  box.className = "h3p-material";
  const heading = document.createElement("b");
  heading.textContent = title;
  const body = document.createElement("span");
  body.textContent = description;
  box.append(heading, body);
  return box;
}

function mediaViewUrl(filename) {
  if (!filename) return "";
  const normalized = String(filename).replaceAll("\\", "/");
  const slash = normalized.lastIndexOf("/");
  const subfolder = slash >= 0 ? normalized.slice(0, slash) : "";
  const basename = slash >= 0 ? normalized.slice(slash + 1) : normalized;
  const query = new URLSearchParams({ filename: basename, type: "input" });
  if (subfolder) query.set("subfolder", subfolder);
  return `/view?${query.toString()}`;
}

function mediaPreview(filename, accept) {
  const url = mediaViewUrl(filename);
  if (!url) return null;
  const preview = document.createElement("div");
  preview.className = "h3p-media-preview";
  const status = document.createElement("span");
  status.className = "h3p-preview-status";
  status.hidden = true;
  if (accept.startsWith("image/")) {
    const link = document.createElement("a");
    link.className = "h3p-image-link";
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener";
    link.title = "点击查看原图";
    const image = document.createElement("img");
    image.className = "h3p-image-preview";
    image.src = url;
    image.alt = "已选择图片预览";
    image.addEventListener("error", () => {
      image.hidden = true;
      status.hidden = false;
      status.textContent = "图片预览不可用";
    }, { once: true });
    link.append(image);
    preview.append(link, status);
    return preview;
  }
  if (accept.startsWith("audio/")) {
    const audio = document.createElement("audio");
    audio.className = "h3p-audio-preview";
    audio.src = url;
    audio.controls = true;
    audio.preload = "metadata";
    audio.addEventListener("error", () => {
      status.hidden = false;
      status.textContent = "音频预览不可用";
    }, { once: true });
    preview.append(audio, status);
    return preview;
  }
  return null;
}

function replaceMediaPreview(wrapper, filename, accept) {
  wrapper?.querySelector(".h3p-media-preview")?.remove();
  const preview = mediaPreview(filename, accept);
  if (preview) wrapper?.append(preview);
}

function uploadControl(node, label, widgetName, accept) {
  const wrapper = document.createElement("div");
  wrapper.className = "h3p-upload";
  const name = document.createElement("span");
  const filename = widget(node, widgetName)?.value || "";
  name.textContent = filename || `${label}：未上传`;
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "选择文件";
  const input = document.createElement("input");
  input.type = "file";
  input.accept = accept;
  input.dataset.h3pUpload = widgetName;
  input.dataset.h3pUploadLabel = label;
  input.dataset.h3pUploadFile = widgetName;
  input.dataset.h3pAccept = accept;
  button.dataset.h3pUploadButton = widgetName;
  wrapper.append(name, button, input);
  replaceMediaPreview(wrapper, filename, accept);
  return wrapper;
}

async function uploadFile(node, input) {
  const file = input.files?.[0];
  if (!file) return;
  const wrapper = input.closest(".h3p-upload");
  const name = wrapper?.querySelector("span");
  wrapper?.classList.add("busy");
  if (name) name.textContent = "上传中…";
  try {
    const form = new FormData();
    form.append("asset", file, file.name);
    form.append("filename", file.name);
    const response = await api.fetchApi("/h3-director-plus/assets", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result?.error?.message || `HTTP ${response.status}`);
    syncUploadWidget(node, input.dataset.h3pUploadFile, result.asset);
    if (name) name.textContent = result.asset;
    replaceMediaPreview(wrapper, result.asset, input.dataset.h3pAccept);
    wrapper?.classList.remove("error");
  } catch (error) {
    if (name) name.textContent = `上传失败：${error.message}`;
    wrapper?.classList.add("error");
  } finally {
    wrapper?.classList.remove("busy");
    input.value = "";
  }
}

function bindDomControls(element, node, render) {
  if (!element || element.__h3pControlsBound) return;
  element.__h3pControlsBound = true;
  element.addEventListener("pointerdown", (event) => event.stopPropagation(), true);
  element.addEventListener("click", (event) => {
    const button = event.target.closest?.(".h3p button");
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    if (button.dataset.h3pAction) {
      setWidget(node, button.dataset.h3pAction, button.dataset.h3pControl);
      render();
    }
    if (button.dataset.h3pUploadButton) {
      element.querySelector(`input[data-h3p-upload-file="${button.dataset.h3pUploadButton}"]`)?.click();
    }
  }, true);
  element.addEventListener("change", (event) => {
    const input = event.target.closest?.(".h3p input[type=file]");
    if (input) {
      event.stopPropagation();
      uploadFile(node, input);
      return;
    }
    const valueInput = event.target.closest?.("[data-h3p-value-widget]");
    if (!valueInput) return;
    event.stopPropagation();
    const value = valueInput.dataset.h3pNumber === "true" ? Number.parseInt(valueInput.value, 10) : valueInput.value;
    setWidget(node, valueInput.dataset.h3pValueWidget, value);
    if (["aspect_ratio", "resolution_preset", "custom_width", "custom_height"].includes(valueInput.dataset.h3pValueWidget)) syncResolution(node);
    render();
  }, true);
  element.addEventListener("input", (event) => {
    const valueInput = event.target.closest?.("[data-h3p-value-widget]");
    if (!valueInput || valueInput.matches("select")) return;
    event.stopPropagation();
    const value = valueInput.dataset.h3pNumber === "true" ? Number.parseInt(valueInput.value, 10) : valueInput.value;
    if (Number.isNaN(value)) return;
    setWidget(node, valueInput.dataset.h3pValueWidget, value);
  }, true);
}

function bindRenderedControllers() {
  document.querySelectorAll(".h3p[data-h3p-controller]").forEach((element) => {
    const controller = controllers.get(element.dataset.h3pController);
    if (controller) bindDomControls(element, controller.node, controller.render);
  });
}

function controllerForTarget(target) {
  const root = target?.closest?.(".h3p[data-h3p-controller]");
  return root ? controllers.get(root.dataset.h3pController) : null;
}

function bindDocumentControls() {
  if (documentControlsBound) return;
  documentControlsBound = true;
  document.addEventListener("click", (event) => {
    const button = event.target.closest?.(".h3p button");
    const controller = controllerForTarget(button);
    if (!button || !controller) return;
    if (button.dataset.h3pAction) {
      event.preventDefault();
      event.stopPropagation();
      setWidget(controller.node, button.dataset.h3pAction, button.dataset.h3pControl);
      controller.render();
      return;
    }
    if (button.dataset.h3pUploadButton) {
      event.preventDefault();
      event.stopPropagation();
      button.closest(".h3p")?.querySelector(`input[data-h3p-upload-file="${button.dataset.h3pUploadButton}"]`)?.click();
    }
  }, true);
  document.addEventListener("change", (event) => {
    const controller = controllerForTarget(event.target);
    if (!controller) return;
    const fileInput = event.target.closest?.(".h3p input[type=file]");
    if (fileInput) {
      event.stopPropagation();
      uploadFile(controller.node, fileInput);
      return;
    }
    const valueInput = event.target.closest?.(".h3p [data-h3p-value-widget]");
    if (!valueInput) return;
    event.stopPropagation();
    const value = valueInput.dataset.h3pNumber === "true" ? Number.parseInt(valueInput.value, 10) : valueInput.value;
    setWidget(controller.node, valueInput.dataset.h3pValueWidget, value);
    if (["aspect_ratio", "resolution_preset", "custom_width", "custom_height"].includes(valueInput.dataset.h3pValueWidget)) syncResolution(controller.node);
    controller.render();
  }, true);
  document.addEventListener("input", (event) => {
    const controller = controllerForTarget(event.target);
    const valueInput = event.target.closest?.('.h3p [data-h3p-value-widget]:not(select)');
    if (!controller || !valueInput) return;
    event.stopPropagation();
    const value = valueInput.dataset.h3pNumber === "true" ? Number.parseInt(valueInput.value, 10) : valueInput.value;
    if (!Number.isNaN(value)) setWidget(controller.node, valueInput.dataset.h3pValueWidget, value);
  }, true);
}

new MutationObserver(bindRenderedControllers).observe(document.documentElement, {
  childList: true,
  subtree: true,
});

function install(node) {
  if (node.__h3DirectorPlusInstalled) return;
  node.__h3DirectorPlusInstalled = true;
  installStyles();
  bindDocumentControls();

  const hidden = [
    "mode", "prompt", "duration", "width", "height", "aspect_ratio", "resolution_preset", "custom_width", "custom_height", "seed", "seed_mode", "voice_mode", "fish_model_path",
    "ref_image_size", "performance_preset", "timeline_data",
    "target_dialogue", "reference_transcript", "voice_reference_name_1", "voice_reference_name_2", "voice_reference_name_3",
    "first_image_file", "last_image_file", "voice_reference_audio_file", "voice_reference_audio_2_file", "voice_reference_audio_3_file",
    "reference_image_1_file", "reference_image_2_file", "reference_image_3_file", "reference_image_4_file", "reference_image_5_file",
    "reference_image_6_file", "reference_image_7_file", "reference_image_8_file", "reference_image_9_file",
  ];
  hidden.forEach((name) => hideWidget(widget(node, name)));

  const root = document.createElement("div");
  root.className = "h3p";
  const controllerId = `h3p-${++nextControllerId}`;
  root.dataset.h3pController = controllerId;
  let fishPanel;

  function render() {
    const mode = widget(node, "mode")?.value || "FL2VA";
    const voiceMode = widget(node, "voice_mode")?.value || "none";
    const performanceOptions = allowedPerformancePresets(mode, voiceMode);
    let preset = widget(node, "performance_preset")?.value || "稳定质量";
    if (!performanceOptions.includes(preset)) {
      preset = "稳定质量";
      setWidget(node, "performance_preset", preset, false);
    }
    const aspect = widget(node, "aspect_ratio")?.value || "16:9";
    const resolutionPreset = widget(node, "resolution_preset")?.value || "0.83 MP";
    const [resolvedWidth, resolvedHeight] = syncResolution(node);
    const resolvedBackend = mode === "REF2VA" || voiceMode !== "none"
      ? "ref2va_model"
      : "fl2va_model";

    root.replaceChildren();

    const head = document.createElement("div");
    head.className = "h3p-head";
    const title = document.createElement("div");
    title.className = "h3p-title";
    title.textContent = "MiniMax H3 导演台 Plus";
    const badge = document.createElement("span");
    badge.className = "h3p-badge";
    badge.textContent = "reference 专用";
    head.append(title, badge);
    root.append(head);

    const quick = document.createElement("section");
    quick.className = "h3p-section";
    quick.innerHTML = '<div class="h3p-section-title"><span>快速设置</span><span class="h3p-hint">模式与加速预设</span></div>';
    const quickGrid = document.createElement("div");
    quickGrid.className = "h3p-grid";
    quickGrid.append(
      valueControl("生成模式", "mode", MODES, mode),
      valueControl("性能预设", "performance_preset", performanceOptions, preset),
    );
    quick.append(quickGrid);
    root.append(quick);

    const specification = document.createElement("section");
    specification.className = "h3p-section";
    specification.innerHTML = '<div class="h3p-section-title"><span>生成规格</span><span class="h3p-hint">H3 原生生成参数</span></div>';
    const specGrid = document.createElement("div");
    specGrid.className = "h3p-grid";
    specGrid.append(
      valueControl("视频时长（秒）", "duration", [], widget(node, "duration")?.value || 5, "number"),
      valueControl("画面比例", "aspect_ratio", [...Object.keys(ASPECTS), "CUSTOM"], aspect),
      valueControl("分辨率档位", "resolution_preset", RESOLUTIONS, resolutionPreset),
    );
    const finalSize = document.createElement("label");
    finalSize.className = "h3p-field";
    finalSize.innerHTML = `<span>最终尺寸</span><div class="h3p-readonly">${resolvedWidth} × ${resolvedHeight}</div>`;
    specGrid.append(
      finalSize,
      valueControl("噪音种子", "seed", [], widget(node, "seed")?.value ?? 0, "number"),
      valueControl("种子模式", "seed_mode", SEED_MODES, widget(node, "seed_mode")?.value || "randomize"),
    );
    if (aspect === "CUSTOM") {
      specGrid.append(
        valueControl("自定义宽比", "custom_width", [], widget(node, "custom_width")?.value || 16, "number"),
        valueControl("自定义高比", "custom_height", [], widget(node, "custom_height")?.value || 9, "number"),
      );
    }
    specification.append(specGrid);
    const specNote = document.createElement("div");
    specNote.className = "h3p-spec-note";
    specNote.textContent = "时长支持 4–15 秒；尺寸按 32 对齐。2K/4K 属于生成后的超分选项，不冒充 H3 原生分辨率。";
    specification.append(specNote);
    root.append(specification);

    const director = document.createElement("section");
    director.className = "h3p-section";
    director.innerHTML = '<div class="h3p-section-title"><span>导演与素材</span><span class="h3p-hint">选择模式后直接上传</span></div>';
    const materials = document.createElement("div");
    materials.className = "h3p-materials";
    materials.append(
      material("首帧图片", mode === "T2VA" ? "当前模式不需要" : "I2VA / FL2VA 可用"),
      material("尾帧图片", ["FL2VA", "L2VA", "REF2VA"].includes(mode) ? "FL2VA / L2VA / REF2VA 可用" : "当前模式不需要"),
      material("音色参考", ["I2VA", "FL2VA", "REF2VA"].includes(mode) ? "启用音色后上传样本" : "切换到图生/参考模式后使用"),
    );
    director.append(materials);
    const uploadGrid = document.createElement("div");
    uploadGrid.className = "h3p-upload-grid";
    if (mode === "REF2VA") {
      // REF2VA slots are all generic references; timing/keyframe roles belong in the prompt.
      const genericPictureLabels = ["参考图 1", "参考图 2", "参考图 3", "参考图 4", "参考图 5", "参考图 6", "参考图 7", "参考图 8", "参考图 9"];
      REF2VA_IMAGE_SLOTS.forEach(([label, field], index) => uploadGrid.append(uploadControl(node, genericPictureLabels[index] || label, field, "image/*")));
      const pictureOrderHint = document.createElement("div");
      pictureOrderHint.textContent = "REF2VA 参考图均为普通参考图，不自动指定首帧或尾帧；请在提示词中用 <Picture N> 指定时间位置。";
      pictureOrderHint.className = "h3p-spec-note";
      pictureOrderHint.textContent = "图片编号与 <Picture N> 一一对应，请从参考图 1 开始连续上传，不要跳过中间编号。";
      director.append(pictureOrderHint);
    } else {
      if (["I2VA", "FL2VA"].includes(mode)) uploadGrid.append(uploadControl(node, "首帧图片", "first_image_file", "image/*"));
      if (["FL2VA", "L2VA"].includes(mode)) uploadGrid.append(uploadControl(node, "尾帧图片", "last_image_file", "image/*"));
    }
    director.append(uploadGrid);
    const prompt = document.createElement("textarea");
    prompt.placeholder = "视频提示词：描述角色、动作、镜头、对白和时间点";
    prompt.value = widget(node, "prompt")?.value || "";
    prompt.dataset.h3pValueWidget = "prompt";
    ["pointerdown", "mousedown"].forEach((name) => prompt.addEventListener(name, (event) => event.stopPropagation()));
    prompt.style.marginTop = "6px";
    // Give the native Prompt Assistant a local positioning anchor. Without
    // this wrapper it anchors to the whole U11 DOM widget and can land outside
    // the visible prompt field when the canvas is zoomed or a sidebar is open.
    const promptAnchor = document.createElement("div");
    promptAnchor.className = "h3p-prompt-anchor dom-widget";
    promptAnchor.append(prompt);
    director.append(promptAnchor);
    keepPromptAssistantReadable(promptAnchor);

    // Bind this visible textarea to the installed Prompt Assistant plugin.
    // The original widget remains hidden for layout, but the assistant now
    // sees the real editor and can mount its full toolbar and popups here.
    const promptWidget = widget(node, "prompt");
    if (promptWidget) {
      promptWidget.inputEl = prompt;
      promptWidget.element = prompt;
      prompt._promptAssistantInput = true;
      // Prompt Assistant can finish loading after this node is created. Retry
      // briefly so the native toolbar mounts on the visible U11 editor too.
      let assistantAttempts = 0;
      const mountPromptAssistant = () => {
        const assistant = window.promptAssistant || app.promptAssistant;
        if (assistant && typeof assistant.checkAndSetupNode === "function") {
          // Prompt Assistant may have auto-created an instance for the hidden
          // native widget before this DOM editor was mounted. Drop that stale
          // instance so the visible textarea becomes the anchor.
          assistant.cleanup?.(node.id, true);
          promptWidget.inputEl = prompt;
          promptWidget.element = prompt;
          assistant.checkAndSetupNode(node);
          return;
        }
        if (++assistantAttempts < 20) setTimeout(mountPromptAssistant, 250);
      };
      requestAnimationFrame(mountPromptAssistant);
    }

    // Keep a small fallback only when the real Prompt Assistant plugin is not
    // loaded. When it is installed, its full toolbar owns this input.
    if (!window.PromptAssistant_Version) {
    // The raw prompt widget is hidden to avoid duplicate controls. Keep the
    // prompt assistant visible inside the director surface instead.
    const promptTools = document.createElement("div");
    promptTools.className = "h3p-prompt-tools";
    const promptToolsTitle = document.createElement("div");
    promptToolsTitle.className = "h3p-prompt-tools-title";
    promptToolsTitle.innerHTML = '<span>提示词小助手</span><span class="h3p-prompt-tools-hint">按当前模式插入 H3 格式</span>';
    const promptActions = document.createElement("div");
    promptActions.className = "h3p-prompt-actions";
    const promptHelp = document.createElement("div");
    promptHelp.className = "h3p-prompt-help";
    const promptTemplate = mode === "REF2VA"
      ? "integrated_multimodal_description: [Shot 1] <Picture 1> 的主体、动作、镜头与时间变化。\noverall_soundscape: 环境声、对白与同步音效。\nnon_diegetic_music: N/A"
      : "integrated_multimodal_description: [Shot 1] 描述主体、动作、镜头、光线与对白。\noverall_soundscape: 环境声、对白与同步音效。\nnon_diegetic_music: N/A";
    const promptHint = mode === "REF2VA"
      ? "REF2VA 的图片都是普通 reference；用 <Picture 1> 到 <Picture 9> 在提示词中指定首尾、角色或场景作用。"
      : "先写主体、动作、镜头和时间点；I2VA/FL2VA 可用 <Picture 1> 指定首帧，音色参考用 <Audio 1> 到 <Audio 3>。";
    const insertPromptText = (text) => {
      const current = String(prompt.value || "").trim();
      prompt.value = current ? `${current}\n\n${text}` : text;
      setWidget(node, "prompt", prompt.value);
      prompt.focus();
    };
    const promptButton = (label, text) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.title = `插入 ${label}`;
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        insertPromptText(text);
      });
      return button;
    };
    promptActions.append(
      promptButton("插入 H3 结构", promptTemplate),
      promptButton("插入 <Picture 1>", "<Picture 1>"),
      promptButton("插入 <Audio 1>", "<Audio 1>"),
    );
    promptHelp.textContent = promptHint;
    promptTools.append(promptToolsTitle, promptActions, promptHelp);
    director.append(promptTools);
    }
    root.append(director);

    const voice = document.createElement("section");
    voice.className = "h3p-section";
    voice.innerHTML = '<div class="h3p-section-title"><span>音色参考</span><span class="h3p-hint">只提取音色与表达方式</span></div>';
    const voiceBar = document.createElement("div");
    voiceBar.className = "h3p-grid";
    voiceBar.append(valueControl("音色模式", "voice_mode", VOICE_MODES, voiceMode));
    voice.append(voiceBar);
    const audioLane = document.createElement("div");
    audioLane.className = "h3p-audio-lane";
    const audioHintTitle = document.createElement("b");
    audioHintTitle.textContent = "编号音色入口：";
    audioLane.append(audioHintTitle, `提示词中使用 ${AUDIO_REFERENCE_HINT} 指定人物音色。样本只提取音色，不使用原音频完整内容。`);
    if (voiceMode !== "none" && ["I2VA", "FL2VA", "REF2VA"].includes(mode)) {
      const audioGrid = document.createElement("div");
      audioGrid.className = "h3p-audio-grid";
      const audioFields = voiceMode === "h3_reference"
        ? [[1, "voice_reference_audio_file"], [2, "voice_reference_audio_2_file"], [3, "voice_reference_audio_3_file"]]
        : [[1, "voice_reference_audio_file"]];
      audioFields.forEach(([index, field]) => {
        const slot = document.createElement("div");
        slot.className = "h3p-voice-slot";
        slot.append(
          valueControl(`${VOICE_REFERENCE_LABELS[index - 1]} 对应角色`, `voice_reference_name_${index}`, [], widget(node, `voice_reference_name_${index}`)?.value || "", "text"),
          uploadControl(node, VOICE_REFERENCE_LABELS[index - 1], field, "audio/*"),
        );
        audioGrid.append(slot);
      });
      audioLane.append(audioGrid);
    }
    voice.append(audioLane);
    fishPanel = document.createElement("details");
    fishPanel.className = "h3p-fish";
    fishPanel.open = false;
    fishPanel.innerHTML = '<summary>高级音色锁定 · Fish S2（可选）</summary><div class="h3p-fish-body">Fish S2 先用参考音色生成新的目标对白，再把生成结果送入 H3 reference 路线。不会复用样本里的原对白，也不会在失败时静默回退。</div>';
    if (voiceMode === "fish_lock") {
      fishPanel.open = true;
      const fishBody = fishPanel.querySelector(".h3p-fish-body");
      const fishGrid = document.createElement("div");
      fishGrid.className = "h3p-grid";
      fishGrid.style.marginTop = "7px";
      fishGrid.append(
        valueControl("Fish 模型", "fish_model_path", FISH_MODELS, widget(node, "fish_model_path")?.value || FISH_MODELS[0][0]),
        valueControl("新的目标对白", "target_dialogue", [], widget(node, "target_dialogue")?.value || "", "textarea"),
        valueControl("音色样本文本（建议填写）", "reference_transcript", [], widget(node, "reference_transcript")?.value || "", "textarea"),
      );
      fishBody.append(fishGrid);
    }
    voice.append(fishPanel);
    root.append(voice);

    const backend = document.createElement("section");
    backend.className = "h3p-section";
    backend.innerHTML = '<div class="h3p-section-title"><span>实际后端</span><span class="h3p-hint">自动路由结果</span></div>';
    const route = document.createElement("div");
    route.className = "h3p-route";
    const backendLabel = document.createElement("span");
    backendLabel.textContent = "模型：";
    const backendValue = document.createElement("strong");
    backendValue.textContent = resolvedBackend;
    const routeNote = document.createElement("div");
    routeNote.className = resolvedBackend === "ref2va_model" && mode !== "REF2VA" ? "warning" : "ok";
    routeNote.textContent = resolvedBackend === "ref2va_model" && mode !== "REF2VA"
      ? "启用音色后自动走 REF2VA。首尾图片改为提示词级端点参考，不再是 FL2VA 的硬首尾帧。"
      : resolvedBackend === "fl2va_model"
        ? "未启用音色参考，I2VA/FL2VA 保持硬首尾帧约束。"
        : "REF2VA 使用图片、视频与音频参考素材生成。";
    route.append(backendLabel, backendValue, routeNote);
    backend.append(route);
    root.append(backend);

    node.setSize?.([DIRECTOR_UI_WIDTH, DIRECTOR_UI_HEIGHT]);
    node.graph?.setDirtyCanvas(true, true);
    requestAnimationFrame(bindRenderedControllers);
  }

  const domWidget = node.addDOMWidget?.("minimax_h3_director_plus_ui", "custom", root, {
    serialize: false,
    hideOnZoom: false,
    getHeight: () => DIRECTOR_DOM_HEIGHT,
  });
  if (domWidget) domWidget.computeSize = () => [DIRECTOR_UI_WIDTH, DIRECTOR_DOM_HEIGHT];
  controllers.set(controllerId, { node, render });
  const bindMountedControls = () => {
    bindDomControls(domWidget?.element || domWidget?.inputEl || root, node, render);
    bindRenderedControllers();
    const mountedPrompt = root.querySelector('textarea[data-h3p-value-widget="prompt"]');
    const promptWidget = widget(node, "prompt");
    const assistant = window.promptAssistant || app.promptAssistant;
    if (mountedPrompt && promptWidget && assistant?.checkAndSetupNode) {
      assistant.cleanup?.(node.id, true);
      promptWidget.inputEl = mountedPrompt;
      promptWidget.element = mountedPrompt;
      assistant.checkAndSetupNode(node);
    }
  };
  bindDomControls(root, node, render);
  requestAnimationFrame(bindMountedControls);
  setTimeout(bindMountedControls, 250);

  ["mode", "voice_mode", "performance_preset", "aspect_ratio", "resolution_preset", "seed_mode"].forEach((name) => {
    const item = widget(node, name);
    if (!item) return;
    const original = item.callback;
    item.callback = (value) => {
      original?.(value);
      render();
    };
  });

  const originalConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    originalConfigure?.apply(this, args);
    requestAnimationFrame(render);
  };
  render();
}

app.registerExtension({
  name: "MiniMaxH3.DirectorPlus",
  nodeCreated(node) {
    if (node.comfyClass === NODE_CLASS) install(node);
  },
  loadedGraphNode(node) {
    if (node.comfyClass === NODE_CLASS) install(node);
  },
});
