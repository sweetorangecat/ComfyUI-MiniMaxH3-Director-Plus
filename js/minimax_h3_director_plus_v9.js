/**
 * MiniMax H3 Director Plus UI.
 * Adapted from the visual language of DaSiWa MiniMax H3 Director under GPL-3.0.
 * This implementation is independent and intentionally exposes only H3 reference semantics.
 */
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { clearSlot, compactBoundSlots, compactSlots } from "./media_slot_state.mjs";

const NODE_CLASS = "MiniMaxH3DirectorPlus";
const SMART_PRESET = "免费智能 1080p";
const SMART_UPSCALE_MODEL = "RealESRGAN_x2plus.pth";
// Keep node geometry independent from the browser sidebar width.
const DIRECTOR_UI_WIDTH = 1350;
const DIRECTOR_MIN_CONTENT_WIDTH = 1180;
const DIRECTOR_UI_HEIGHT = 1760;
const DIRECTOR_DOM_HEIGHT = 1280;
const DIRECTOR_CONTENT_INSET = 48;
const DIRECTOR_VIEWPORT_HEIGHT = DIRECTOR_DOM_HEIGHT;
const MODES = ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"];
const PRESETS = ["免费智能 1080p", "稳定质量", "质量优先加速", "质量优先二采样", "高清快速（v4 8步）", "极速4步", "参考图加速", "参考高清（原生20步）", "参考极速（官方4步）", "低显存", "低显存二采", "自定义"];
const PERFORMANCE_PRESET_KEYS = {
  "高清快速（v4 8步）": "fl_quality_fast_v4",
  "参考高清（原生20步）": "ref_quality_native",
  "参考极速（官方4步）": "ref_fast_4step",
};
const PERFORMANCE_PRESETS_BY_ROUTE = {
  t2va: ["免费智能 1080p", "稳定质量"],
  endpoint: ["免费智能 1080p", "稳定质量"],
  reference: ["免费智能 1080p", "参考高清（原生20步）"],
};
const ASPECTS = {
  "1:1": [1, 1], "3:2": [3, 2], "2:3": [2, 3], "4:3": [4, 3], "3:4": [3, 4],
  "8:5": [8, 5], "5:8": [5, 8], "16:9": [16, 9], "9:16": [9, 16], "21:9": [21, 9], "9:21": [9, 21],
};
const EXACT_OUTPUT_TARGETS = {
  "1080p FHD|16:9": [1920, 1080],
  "1080p FHD|9:16": [1080, 1920],
  "2K QHD|16:9": [2560, 1440],
  "2K QHD|9:16": [1440, 2560],
  "4K UHD|16:9": [3840, 2160],
  "4K UHD|9:16": [2160, 3840],
};
const RESOLUTION_MEGAPIXELS = {
  "1080p FHD": 1920 * 1080 / (1024 * 1024),
  "2K QHD": 3.6864,
  "4K UHD": 8.2944,
};
const LOW_VRAM_TWO_STAGE_MIN_FIRST_MP = 0.20;
const LOW_VRAM_TWO_STAGE_SCALE = 1.5;
const LOW_VRAM_TWO_STAGE_MAX_FINAL_SCALE = 1.45;
const RESOLUTIONS = [
  "0.26 MP", "0.30 MP", "0.36 MP", "0.40 MP", "0.50 MP", "0.52 MP", "0.60 MP", "0.65 MP", "0.70 MP",
  "0.80 MP", "0.83 MP", "0.90 MP", "1.00 MP", "1.05 MP", "1.10 MP", "1.20 MP", "1.30 MP", "1.35 MP",
  "1.40 MP", "1.50 MP", "1.55 MP", "1.60 MP", "1.65 MP", "1.70 MP", "1.75 MP", "1.80 MP", "1.90 MP",
  "2.00 MP", "2.10 MP", "1080p FHD", "2K QHD", "4K UHD",
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
const POSTPROCESS_MODES = [
  ["native", "原生尺寸直出"],
  ["lanczos", "Lanczos 快速放大"],
  ["ai_upscale", "AI 自动超分"],
  ["rtx_vsr", "AI 细节重建（RTX VSR）"],
];
const POSTPROCESS_MODES_BY_PERFORMANCE = {
  "免费智能 1080p": [["ai_upscale", "AI X2 超分（智能锁定）"]],
  "质量优先二采样": [["rtx_vsr", "AI 细节重建（RTX VSR）"]],
  "低显存二采": [["ai_upscale", "AI X2 细节重建（低显存）"]],
};
const RTX_QUALITIES = [
  ["HIGH", "HIGH（质量）"],
  ["ULTRA", "ULTRA（更高质量）"],
  ["HIGHBITRATE_ULTRA", "HIGHBITRATE_ULTRA（原画源最高保真）"],
];
const MOTION_SMOOTHING = [
  ["off", "关闭（默认，保留原始帧）"],
  ["rife_x2", "RIFE 2x（48 FPS）"],
];
const AUDIO_LOUDNESS = [
  ["auto", "自动增强响度"],
  ["original", "保持原始响度"],
];
const VOICE_REFERENCE_LABELS = ["音色参考 1", "音色参考 2", "音色参考 3"];
const VOICE_REFERENCE_MODES = ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"];
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
    .h3p{box-sizing:border-box;width:100%;min-width:${DIRECTOR_MIN_CONTENT_WIDTH}px;max-width:none;height:${DIRECTOR_VIEWPORT_HEIGHT}px;max-height:none;padding:8px;background:#0f151b;color:#d9e4eb;font:12px system-ui,sans-serif;display:flex;flex-direction:column;gap:8px;overflow-y:auto;overflow-x:auto}
    .h3p *{box-sizing:border-box;letter-spacing:0}.h3p-head{display:flex;align-items:center;justify-content:space-between;gap:10px}.h3p-title{font-size:14px;font-weight:700;color:#fff}.h3p-badge{padding:2px 7px;border:1px solid #4d6372;border-radius:4px;color:#9fc5d8;background:#17232c}
    .h3p-workbench-bar{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;padding:10px 11px;border:1px solid #3f789e;border-radius:5px;background:#14232b;box-shadow:inset 0 1px 0 rgba(137,202,230,.08)}.h3p-workbench-copy{min-width:0}.h3p-workbench-kicker{color:#8fc6dc;font-size:10px;line-height:1.3;text-transform:uppercase}.h3p-workbench-name{margin-top:2px;color:#fff;font-size:16px;font-weight:750;line-height:1.25}.h3p-workbench-hint{margin-top:3px;color:#9fb2bf;font-size:11px;line-height:1.4}.h3p-workbench-badge{align-self:start;padding:4px 8px;border:1px solid #4c806b;border-radius:4px;background:#18372f;color:#a8e0bd;font-size:11px;white-space:nowrap}.h3p-status-strip{grid-column:1/-1;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin-top:2px}.h3p-status-item{min-width:0;padding:6px 7px;border:1px solid #344956;border-radius:4px;background:#101b21}.h3p-status-label{display:block;color:#8298a6;font-size:10px;line-height:1.2}.h3p-status-value{display:block;margin-top:2px;overflow:hidden;color:#e0e8ed;font-size:11px;font-weight:650;line-height:1.35;text-overflow:ellipsis;white-space:nowrap}
    .h3p-section{border-top:1px solid #2d3b46;padding-top:8px}.h3p-section-title{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;color:#fff;font-weight:650}.h3p-hint{color:#8fa5b4;font-weight:400;font-size:11px}
    .h3p-segments{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:4px}.h3p-segments.preset{grid-template-columns:repeat(5,minmax(0,1fr))}.h3p button,.h3p-segment{min-height:29px;padding:4px 6px;border:1px solid #3c4d59;border-radius:4px;background:#18232c;color:#aebfca;cursor:pointer;text-align:center;display:flex;align-items:center;justify-content:center}.h3p button:hover,.h3p-segment:hover{background:#22323e;color:#fff}.h3p-segment.active{border-color:#78bce0;background:#17384a;color:#fff;box-shadow:inset 0 0 0 1px rgba(120,188,224,.2)}.h3p-segment input{position:absolute;opacity:0;pointer-events:none;width:1px;height:1px}
    .h3p-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}.h3p-field{display:flex;flex-direction:column;gap:4px;min-width:0}.h3p label{color:#9fb2bf}.h3p select,.h3p input,.h3p textarea{width:100%;max-width:100%;min-width:0;box-sizing:border-box;border:1px solid #3c4d59;border-radius:4px;background:#0b1116;color:#e6edf2;padding:6px}.h3p textarea{min-height:104px;resize:vertical;line-height:1.5}.h3p-readonly{min-height:30px;padding:6px;border:1px solid #344956;border-radius:4px;background:#111d24;color:#8fd2f3}.h3p-spec-note{margin-top:5px;color:#8298a6;font-size:11px}
    .h3p-prompt-tools{margin-top:6px;padding:7px;border:1px solid #344956;border-radius:4px;background:#111d24}.h3p-prompt-tools-title{display:flex;align-items:center;justify-content:space-between;gap:8px;color:#fff;font-weight:650}.h3p-prompt-tools-hint{color:#8fa5b4;font-weight:400;font-size:11px}.h3p-prompt-actions{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}.h3p-prompt-actions button{min-height:26px;padding:3px 8px}.h3p-prompt-help{margin-top:6px;color:#9fb2bf;font-size:11px;line-height:1.45}
    .h3p-route{display:grid;grid-template-columns:auto 1fr;gap:5px 10px;padding:7px;border:1px solid #344956;border-radius:4px;background:#111d24}.h3p-route strong{color:#8fd2f3}.h3p-route .warning{grid-column:1/-1;color:#e9c176;line-height:1.45}.h3p-route .ok{grid-column:1/-1;color:#93cda8;line-height:1.45}
    .h3p-voice{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px}.h3p-audio-lane{margin-top:6px;padding:7px;border:1px dashed #4e7182;border-radius:4px;background:#101b21;color:#adc2ce;line-height:1.45}.h3p-audio-lane b{color:#dfeaf0}.h3p-audio-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin-top:7px}.h3p-voice-slot{display:flex;flex-direction:column;gap:5px;min-width:0}.h3p-fish{margin-top:6px;border:1px solid #3b4a55;border-radius:4px;background:#111920}.h3p-fish summary{padding:7px;cursor:pointer;color:#d5dee4;font-weight:600}.h3p-fish-body{padding:0 7px 7px;color:#99aebc;line-height:1.5}
    .h3p-materials{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}.h3p-material{min-height:52px;padding:7px;border:1px solid #344956;border-radius:4px;background:#111a20}.h3p-material b{display:block;color:#e0e8ed;margin-bottom:3px}.h3p-material span{color:#8298a6;font-size:11px;line-height:1.35}.h3p-upload-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin-top:6px}.h3p-upload{display:grid;grid-template-columns:minmax(0,1fr) auto auto;align-items:center;gap:6px;padding:7px;border:1px dashed #4e7182;border-radius:4px;background:#101b21;color:#adc2ce}.h3p-upload>span{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.h3p-upload input{display:none}.h3p-upload button{min-height:25px;padding:3px 8px}.h3p-upload.busy{opacity:.6;pointer-events:none}.h3p-upload.error{border-color:#d87979;color:#ffb1b1}.h3p-media-preview{grid-column:1/-1;min-width:0}.h3p-image-link{display:flex;width:100%;height:132px;border:1px solid #2f424e;border-radius:3px;overflow:hidden;background:#0b1116;align-items:center;justify-content:center}.h3p-image-preview{display:block;width:100%;height:100%;object-fit:contain}.h3p-audio-preview{display:block;width:100%;height:34px}.h3p-preview-status{display:block;color:#e9c176;font-size:11px;padding:4px 0}
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

function enforceDirectorSize(node) {
  const currentWidth = Number(node.size?.[0]) || 0;
  const currentHeight = Number(node.size?.[1]) || 0;
  const width = Math.max(currentWidth, DIRECTOR_UI_WIDTH);
  const height = Math.max(currentHeight, DIRECTOR_UI_HEIGHT);
  if (width !== currentWidth || height !== currentHeight) node.setSize?.([width, height]);
  return [width, height];
}

function protectDirectorComputeSize(node) {
  const originalComputeSize = node.computeSize?.bind(node);
  node.computeSize = (...args) => {
    const computed = originalComputeSize ? originalComputeSize(...args) : [0, 0];
    return [
      Math.max(Number(computed?.[0]) || 0, DIRECTOR_UI_WIDTH),
      Math.max(Number(computed?.[1]) || 0, DIRECTOR_UI_HEIGHT),
    ];
  };
}

function setWidget(node, name, value, notify = true) {
  const item = widget(node, name);
  if (!item || item.value === value) return;
  item.value = value;
  if (notify) item.callback?.(value);
  node.graph?.setDirtyCanvas(true, true);
}

function allowedPerformancePresets(mode, voiceMode) {
  let presets;
  if (voiceMode !== "none" || mode === "REF2VA") presets = PERFORMANCE_PRESETS_BY_ROUTE.reference;
  else if (mode === "T2VA") presets = PERFORMANCE_PRESETS_BY_ROUTE.t2va;
  else presets = PERFORMANCE_PRESETS_BY_ROUTE.endpoint;
  if (voiceMode === "fish_lock") return presets.filter((item) => !["质量优先二采样", "低显存二采"].includes(item));
  return presets;
}

function performancePresetHint(preset) {
  if (preset === SMART_PRESET) return "按后端、显存和时长自动选择路线；无需手动组合超分、模型和运动平滑";
  if (preset === "质量优先二采样") return "训练型 3D latent 二采：匹配 LoRA 首采 4 步 + 神经 latent 放大 + 匹配低 sigma 二采";
  if (preset === "低显存二采") return "8GB 专用真二采：4–6 秒，最高 1080p FHD；时长越长首采网格越小，阶段间自动释放显存";
  if (preset === "质量优先加速") return "20 步 + SageAttention，关闭 Turbo/EasyCache";
  if (preset === "高清快速（v4 8步）") return "v4 8步仅适用于 FL/T2V 后端：社区 v4 LoRA + simple/Euler，单采不做 latent 二采";
  if (preset === "参考高清（原生20步）") return "仅适用于 REF2VA/音色参考：原生 20 步 + SageAttention，不使用 Turbo/二采";
  if (preset === "参考极速（官方4步）") return "仅适用于 REF2VA/音色参考：官方 Ref2VA Turbo 4 步 + 原生 Euler";
  if (preset === "低显存") return "动态分层加载，适合显存受限设备";
  if (preset === "极速4步") return "官方 Turbo LoRA，速度优先";
  return "模式与加速预设";
}

function allowedPostprocessModes(preset) {
  return POSTPROCESS_MODES_BY_PERFORMANCE[preset] || POSTPROCESS_MODES;
}

function allowedRtxQualities(performancePreset) {
  if (performancePreset === "质量优先二采样") {
    return RTX_QUALITIES.filter(([value]) => value === "HIGHBITRATE_ULTRA");
  }
  if (performancePreset === "低显存二采") {
    return RTX_QUALITIES.filter(([value]) => value === "ULTRA");
  }
  return RTX_QUALITIES.filter(([value]) => value !== "HIGHBITRATE_ULTRA");
}

function allowedMotionSmoothing(preset, postprocessMode) {
  if (preset === SMART_PRESET) return [["off", "关闭（智能锁定）"]];
  if (preset === "质量优先二采样") return [["off", "关闭（二采固定，避免重影）"]];
  if (preset === "低显存二采") return [["off", "关闭（低显存二采固定）"]];
  if (preset === "低显存") return [["off", "关闭（低显存固定）"]];
  if (postprocessMode !== "rtx_vsr") {
    return MOTION_SMOOTHING.filter(([value]) => value !== "rife_x2");
  }
  return MOTION_SMOOTHING;
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
  if (!item) return;
  const normalized = value || "";
  const values = item.options?.values;
  if (normalized && Array.isArray(values) && !values.includes(normalized)) item.options.values.push(value);
  item.value = normalized;
  item.callback?.(normalized);
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

function smart1080pTarget(width, height) {
  const shortEdge = 1080;
  const longEdge = Math.max(
    shortEdge,
    Math.round(shortEdge * Math.max(width, height) / Math.max(1, Math.min(width, height))),
  );
  const even = (value) => Math.max(2, Math.round(value / 2) * 2);
  return width >= height ? [even(longEdge), shortEdge] : [shortEdge, even(longEdge)];
}

function calculatedResolution(node) {
  const preset = String(widget(node, "resolution_preset")?.value || "0.83 MP");
  const aspect = String(widget(node, "aspect_ratio")?.value || "16:9");
  const ratio = aspect === "CUSTOM"
    ? [Math.max(1, Number(widget(node, "custom_width")?.value) || 16), Math.max(1, Number(widget(node, "custom_height")?.value) || 9)]
    : (ASPECTS[aspect] || ASPECTS["16:9"]);
  const performancePreset = String(widget(node, "performance_preset")?.value || "");
  if (performancePreset === SMART_PRESET) return smart1080pTarget(...ratio);
  const exact = EXACT_OUTPUT_TARGETS[`${preset}|${aspect}`];
  if (exact) return exact;
  const megapixels = (RESOLUTION_MEGAPIXELS[preset] ?? Number.parseFloat(preset)) || 0.83;
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

function twoStageSizeHint(finalWidth, finalHeight, backend, firstStageMegapixels = 0.90, finalMethod = "RTX VSR") {
  const ratio = finalWidth / Math.max(1, finalHeight);
  const baseArea = firstStageMegapixels * 1000 * 1000;
  const snap = (value) => Math.max(32, Math.round(value / 32) * 32);
  const floorSnap = (value) => Math.max(32, Math.floor(value / 32) * 32);
  let firstWidth = snap(Math.sqrt(baseArea * ratio));
  let firstHeight = snap(Math.sqrt(baseArea / ratio));
  let secondWidth = snap(firstWidth * 1.5);
  let secondHeight = snap(firstHeight * 1.5);
  if (secondWidth > finalWidth || secondHeight > finalHeight) {
    firstWidth = floorSnap(finalWidth / 1.5);
    firstHeight = floorSnap(finalHeight / 1.5);
    secondWidth = snap(firstWidth * 1.5);
    secondHeight = snap(firstHeight * 1.5);
  }
  const scale = Math.max(finalWidth / secondWidth, finalHeight / secondHeight);
  const route = backend === "ref2va_model"
    ? "Reference 训练型 3D latent 二采（Ref LoRA + Sigma 尾段强化）"
    : "FL 训练型 3D latent 二采（8步 LoRA 首采 + 768p LoRA 二采）";
  const finalStage = scale <= 1.01 ? `神经二采已到目标尺寸，${finalMethod} 自动旁路` : `${finalMethod} 约 ${scale.toFixed(2)}x`;
  return `${route}；最终输出 ${finalWidth}×${finalHeight}；H3首采 ${firstWidth}×${firstHeight}；神经latent二采 ${secondWidth}×${secondHeight}；${finalStage}`;
}

function lowVramFirstStageMegapixels(finalWidth, finalHeight) {
  const qualityFloor = (finalWidth * finalHeight)
    / (1000 * 1000 * (LOW_VRAM_TWO_STAGE_SCALE * LOW_VRAM_TWO_STAGE_MAX_FINAL_SCALE) ** 2);
  return Math.max(LOW_VRAM_TWO_STAGE_MIN_FIRST_MP, qualityFloor);
}

function isX2UpscaleModel(value) {
  const name = String(value || "").replaceAll("\\", "/").split("/").pop();
  return /(^|[^0-9])(x2|2x)(?=[^0-9]|$)/i.test(name);
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

function statusItem(label, value) {
  const item = document.createElement("div");
  item.className = "h3p-status-item";
  const name = document.createElement("span");
  name.className = "h3p-status-label";
  name.textContent = label;
  const detail = document.createElement("span");
  detail.className = "h3p-status-value";
  detail.textContent = value;
  item.append(name, detail);
  return item;
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
  button.textContent = filename ? "更换" : "选择文件";
  const input = document.createElement("input");
  input.type = "file";
  input.accept = accept;
  input.dataset.h3pUpload = widgetName;
  input.dataset.h3pUploadLabel = label;
  input.dataset.h3pUploadFile = widgetName;
  input.dataset.h3pAccept = accept;
  button.dataset.h3pUploadButton = widgetName;
  wrapper.append(name, button);
  if (filename) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "移除";
    button.dataset.h3pRemoveFile = widgetName;
    button.setAttribute("data-h3p-remove-file", widgetName);
    button.title = `移除${label}`;
    wrapper.append(button);
  }
  wrapper.append(input);
  replaceMediaPreview(wrapper, filename, accept);
  return wrapper;
}

const VOICE_AUDIO_FIELDS = [
  "voice_reference_audio_file",
  "voice_reference_audio_2_file",
  "voice_reference_audio_3_file",
];

function removeUpload(node, widgetName, render) {
  const mode = String(widget(node, "mode")?.value || "");
  if (mode === "REF2VA") {
    const index = REF2VA_IMAGE_SLOTS.findIndex(([, field]) => field === widgetName);
    if (index >= 0) {
      const values = REF2VA_IMAGE_SLOTS.map(([, field]) => widget(node, field)?.value || "");
      const compacted = compactSlots(values, index);
      REF2VA_IMAGE_SLOTS.forEach(([, field], slotIndex) => syncUploadWidget(node, field, compacted[slotIndex]));
      node._h3pAssetNotice = "参考图已重新编号，请检查提示词中的 <Picture N> 引用。";
      render();
      return;
    }
  }
  const audioIndex = VOICE_AUDIO_FIELDS.indexOf(widgetName);
  if (audioIndex >= 0) {
    const files = VOICE_AUDIO_FIELDS.map((field) => widget(node, field)?.value || "");
    const names = [1, 2, 3].map((index) => widget(node, `voice_reference_name_${index}`)?.value || "");
    const compacted = compactBoundSlots(files, names, audioIndex);
    VOICE_AUDIO_FIELDS.forEach((field, index) => syncUploadWidget(node, field, compacted.files[index]));
    [1, 2, 3].forEach((index) => syncUploadWidget(node, `voice_reference_name_${index}`, compacted.names[index]));
    node._h3pAssetNotice = "音色参考已重新编号，请检查提示词中的 <Audio N> 与角色名。";
    render();
    return;
  }
  if (["first_image_file", "last_image_file"].includes(widgetName)) {
    clearSlot([widget(node, widgetName)?.value || ""], 0);
    syncUploadWidget(node, widgetName, "");
    node._h3pAssetNotice = "已移除素材；当前模式可能需要重新选择必需的首帧或尾帧图片。";
    render();
  }
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
    if (button.dataset.h3pRemoveFile) {
      removeUpload(node, button.dataset.h3pRemoveFile, render);
      return;
    }
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
    if (button.dataset.h3pRemoveFile) {
      event.preventDefault();
      event.stopPropagation();
      removeUpload(controller.node, button.dataset.h3pRemoveFile, controller.render);
      return;
    }
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
  protectDirectorComputeSize(node);

  const hidden = [
    "mode", "prompt", "duration", "width", "height", "aspect_ratio", "resolution_preset", "custom_width", "custom_height", "seed", "seed_mode", "voice_mode", "fish_model_path",
    "ref_image_size", "performance_preset", "postprocess_mode", "rtx_quality", "ai_upscale_model", "motion_smoothing", "audio_loudness", "timeline_data",
    "target_dialogue", "reference_transcript", "voice_reference_name_1", "voice_reference_name_2", "voice_reference_name_3",
    "first_image_file", "last_image_file", "voice_reference_audio_file", "voice_reference_audio_2_file", "voice_reference_audio_3_file",
    "reference_image_1_file", "reference_image_2_file", "reference_image_3_file", "reference_image_4_file", "reference_image_5_file",
    "reference_image_6_file", "reference_image_7_file", "reference_image_8_file", "reference_image_9_file",
  ];
  hidden.forEach((name) => hideWidget(widget(node, name)));

  const root = document.createElement("div");
  root.className = "h3p";
  root.style.minWidth = `${DIRECTOR_MIN_CONTENT_WIDTH}px`;
  root.style.width = `${DIRECTOR_UI_WIDTH - DIRECTOR_CONTENT_INSET}px`;
  const controllerId = `h3p-${++nextControllerId}`;
  root.dataset.h3pController = controllerId;
  let fishPanel;

  function render() {
    const mode = widget(node, "mode")?.value || "FL2VA";
    const voiceMode = widget(node, "voice_mode")?.value || "none";
    const performanceOptions = allowedPerformancePresets(mode, voiceMode);
    let preset = widget(node, "performance_preset")?.value || "稳定质量";
    if (!performanceOptions.includes(preset)) {
      preset = performanceOptions[1] || performanceOptions[0] || "稳定质量";
      setWidget(node, "performance_preset", preset, false);
    }
    if (preset === SMART_PRESET) {
      setWidget(node, "postprocess_mode", "ai_upscale", false);
      setWidget(node, "ai_upscale_model", SMART_UPSCALE_MODEL, false);
      setWidget(node, "motion_smoothing", "off", false);
    }
    if (preset === "低显存二采" && Number(widget(node, "duration")?.value) > 6) {
      setWidget(node, "duration", 6, false);
    }
    const postprocessOptions = allowedPostprocessModes(preset);
    let postprocessMode = widget(node, "postprocess_mode")?.value || "native";
    if (!postprocessOptions.some(([value]) => value === postprocessMode)) {
      postprocessMode = postprocessOptions[0][0];
      setWidget(node, "postprocess_mode", postprocessMode, false);
    }
    const rtxQualityOptions = allowedRtxQualities(preset);
    let rtxQuality = widget(node, "rtx_quality")?.value || "HIGH";
    if (!rtxQualityOptions.some(([value]) => value === rtxQuality)) {
      rtxQuality = preset === "质量优先二采样" ? "HIGHBITRATE_ULTRA" : "ULTRA";
      setWidget(node, "rtx_quality", rtxQuality, false);
    }
    const motionOptions = allowedMotionSmoothing(preset, postprocessMode);
    let motionSmoothing = widget(node, "motion_smoothing")?.value || "off";
    if (!motionOptions.some(([value]) => value === motionSmoothing)) {
      motionSmoothing = motionOptions[0][0];
      setWidget(node, "motion_smoothing", motionSmoothing, false);
    }
    const aspect = widget(node, "aspect_ratio")?.value || "16:9";
    let resolutionPreset = widget(node, "resolution_preset")?.value || "0.83 MP";
    if (preset === SMART_PRESET && resolutionPreset !== "1080p FHD") {
      resolutionPreset = "1080p FHD";
      setWidget(node, "resolution_preset", "1080p FHD", false);
    }
    let [resolvedWidth, resolvedHeight] = syncResolution(node);
    if (
      preset === "低显存二采"
      && resolvedWidth * resolvedHeight > 1920 * 1080 * 1.02
    ) {
      resolutionPreset = "1080p FHD";
      setWidget(node, "resolution_preset", "1080p FHD", false);
      [resolvedWidth, resolvedHeight] = syncResolution(node);
    }
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
    badge.textContent = "自动路由";
    head.append(title, badge);
    root.append(head);

    const workbench = document.createElement("div");
    workbench.className = "h3p-workbench-bar";
    workbench.dataset.h3pWorkbench = "true";
    workbench.setAttribute("data-h3p-workbench", "true");
    const workbenchCopy = document.createElement("div");
    workbenchCopy.className = "h3p-workbench-copy";
    const kicker = document.createElement("div");
    kicker.className = "h3p-workbench-kicker";
    kicker.textContent = "MINIMAX H3 · DIRECTOR PLUS";
    const workbenchName = document.createElement("div");
    workbenchName.className = "h3p-workbench-name";
    workbenchName.textContent = "导演工作台";
    const workbenchHint = document.createElement("div");
    workbenchHint.className = "h3p-workbench-hint";
    workbenchHint.textContent = "模式切换后自动匹配模型、加速路线与素材入口";
    workbenchCopy.append(kicker, workbenchName, workbenchHint);
    const workbenchBadge = document.createElement("div");
    workbenchBadge.className = "h3p-workbench-badge";
    workbenchBadge.textContent = "工作台状态";
    const statusStrip = document.createElement("div");
    statusStrip.className = "h3p-status-strip";
    statusStrip.append(
      statusItem("模式 / 后端", `${mode} / ${resolvedBackend}`),
      statusItem("规格 / 时长", `${resolvedWidth} × ${resolvedHeight} · ${widget(node, "duration")?.value || 5} 秒`),
      statusItem("素材 / 音色", `${mode === "T2VA" ? "纯提示词" : mode === "REF2VA" ? "最多 9 张参考图" : "首尾帧入口"} · ${VOICE_MODE_LABELS[voiceMode]}`),
    );
    workbench.append(workbenchCopy, workbenchBadge, statusStrip);
    root.append(workbench);

    const quick = document.createElement("section");
    quick.className = "h3p-section";
    quick.innerHTML = `<div class="h3p-section-title"><span>快速设置</span><span class="h3p-hint">${performancePresetHint(preset)}</span></div>`;
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
    const durationControl = valueControl("视频时长（秒）", "duration", [], widget(node, "duration")?.value || 5, "number");
    const durationInput = durationControl.querySelector('input[data-h3p-value-widget="duration"]');
    if (durationInput) durationInput.max = preset === "低显存二采" ? "6" : "15";
    const resolutionControl = valueControl(
      "分辨率档位",
      "resolution_preset",
      preset === SMART_PRESET ? [["1080p FHD", "1080p FHD（智能锁定）"]] : RESOLUTIONS,
      resolutionPreset,
    );
    if (preset === SMART_PRESET) resolutionControl.querySelector("select")?.setAttribute("disabled", "disabled");
    specGrid.append(
      durationControl,
      valueControl("画面比例", "aspect_ratio", [...Object.keys(ASPECTS), "CUSTOM"], aspect),
      resolutionControl,
    );
    const finalSize = document.createElement("label");
    finalSize.className = "h3p-field";
    finalSize.innerHTML = `<span>最终尺寸</span><div class="h3p-readonly">${resolvedWidth} × ${resolvedHeight}</div>`;
    specGrid.append(
      finalSize,
      valueControl("噪音种子", "seed", [], widget(node, "seed")?.value ?? 0, "number"),
      valueControl("种子模式", "seed_mode", SEED_MODES, widget(node, "seed_mode")?.value || "randomize"),
    );
    const postprocessGrid = document.createElement("div");
    postprocessGrid.className = "h3p-grid";
    let aiUpscaleModel = widget(node, "ai_upscale_model")?.value || "auto";
    if (preset === SMART_PRESET && aiUpscaleModel !== SMART_UPSCALE_MODEL) {
      aiUpscaleModel = SMART_UPSCALE_MODEL;
      setWidget(node, "ai_upscale_model", SMART_UPSCALE_MODEL, false);
    }
    if (preset === "低显存二采" && aiUpscaleModel !== "auto" && !isX2UpscaleModel(aiUpscaleModel)) {
      aiUpscaleModel = "auto";
      setWidget(node, "ai_upscale_model", "auto", false);
    }
    const postprocessControl = valueControl("最终输出", "postprocess_mode", postprocessOptions, postprocessMode);
    const aiModelControlOptions = preset === SMART_PRESET
      ? [[SMART_UPSCALE_MODEL, "RealESRGAN X2（智能锁定）"]]
      : [["auto", "自动选择"]];
    const aiModelControl = postprocessMode === "ai_upscale"
      ? valueControl("AI 超分模型", "ai_upscale_model", aiModelControlOptions, aiUpscaleModel)
      : postprocessMode === "rtx_vsr"
        ? valueControl("RTX VSR 质量", "rtx_quality", rtxQualityOptions, rtxQuality)
        : document.createElement("span");
    const motionControl = valueControl("运动平滑", "motion_smoothing", motionOptions, motionSmoothing);
    postprocessGrid.append(
      postprocessControl,
      aiModelControl,
      motionControl,
      valueControl("最终音频", "audio_loudness", AUDIO_LOUDNESS, widget(node, "audio_loudness")?.value || "auto"),
    );
    if (preset === SMART_PRESET) {
      [postprocessControl, aiModelControl, motionControl].forEach((field) => {
        field.querySelector?.("select")?.setAttribute("disabled", "disabled");
      });
    }
    if (postprocessMode === "ai_upscale") {
      const modelField = postprocessGrid.children[1];
      const modelSelect = modelField?.querySelector("select");
      const modelWidget = widget(node, "ai_upscale_model");
      const aiModelOptions = modelWidget?.options?.values || [];
      if (modelSelect && (aiModelOptions.length || preset === SMART_PRESET)) {
        modelSelect.replaceChildren();
        const modelOptions = preset === SMART_PRESET
          ? [[SMART_UPSCALE_MODEL, "RealESRGAN X2（智能锁定）"]]
          : [["auto", "自动选择"], ...aiModelOptions.filter((value) => value && value !== "auto" && (preset !== "低显存二采" || isX2UpscaleModel(value))).map((value) => [value, value])];
        modelOptions.forEach(([value, display]) => {
          const option = document.createElement("option");
          option.value = value;
          option.textContent = display;
          option.selected = value === String(modelWidget?.value || "auto");
          modelSelect.append(option);
        });
      }
    }
    specification.append(specGrid);
    specification.append(postprocessGrid);
    const postprocessNote = document.createElement("div");
    postprocessNote.className = "h3p-spec-note";
    const postprocessNotes = {
      native: "原生尺寸直出：保留 H3 实际生成尺寸，不进行放大。选择 2K/4K 时不会自动变成 2K/4K。",
      lanczos: "Lanczos 快速放大：使用 CPU 分块缩放，兼容性最好、速度快，但只重采样不重建 AI 细节。",
      ai_upscale: "AI 自动超分：使用已安装的通用超分模型逐帧重建细节；默认自动选择合适倍率模型，模型不存在会在生成前提示。",
      rtx_vsr: "RTX VSR：目标尺寸大于 H3 原生尺寸时逐帧使用 NVIDIA RTX VSR；首次使用前请安装 nvidia-vfx、NVIDIA Broadcast SDK 与匹配驱动，导演节点会在生成前检查；同尺寸自动旁路。",
    };
    postprocessNote.textContent = postprocessNotes[postprocessMode] || postprocessNotes.native;
    if (preset === "质量优先二采样") {
      postprocessNote.textContent = `质量优先二采样已锁定 RTX VSR：单次 RTX VSR 使用 HIGHBITRATE_ULTRA 高码率档（同尺寸自动旁路）：${twoStageSizeHint(resolvedWidth, resolvedHeight, resolvedBackend)}；不启用不稳定的 DEBLUR_LOW 双效果链，RIFE 固定关闭以避免重影。实际尺寸仍以生成前显存检查为准。`;
    } else if (preset === "低显存二采") {
      postprocessNote.textContent = `低显存二采已锁定 AI X2 细节重建：${twoStageSizeHint(resolvedWidth, resolvedHeight, resolvedBackend, lowVramFirstStageMegapixels(resolvedWidth, resolvedHeight), "AI X2")}；1080p 4 秒保留约 1MP 神经二采基准，5–6 秒会按时长降低首采网格以控制显存，再逐帧 RealESRGAN X2 重建到 FHD；最长 6 秒，开始前检查至少 6GB 空闲显存。`;
    } else if (preset === SMART_PRESET) {
      postprocessNote.textContent = "免费智能 1080p：普通显存使用 20 步 SageAttention 保留细节，再执行一次 RealESRGAN X2 免费超分；低显存设备自动切换安全策略，可能限制为最长 6 秒，生成前会提示原因。";
    }
    specification.append(postprocessNote);
    if (aspect === "CUSTOM") {
      specGrid.append(
        valueControl("自定义宽比", "custom_width", [], widget(node, "custom_width")?.value || 16, "number"),
        valueControl("自定义高比", "custom_height", [], widget(node, "custom_height")?.value || 9, "number"),
      );
    }
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
      material("音色参考", VOICE_REFERENCE_MODES.includes(mode) ? "启用音色后上传样本" : "当前模式不需要"),
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
    if (node._h3pAssetNotice) {
      const notice = document.createElement("div");
      notice.className = "h3p-spec-note";
      notice.setAttribute("role", "status");
      notice.textContent = node._h3pAssetNotice;
      director.append(notice);
    }
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
      : "先写主体、动作、镜头和时间点；I2VA/FL2VA 可用 <Picture 1> 指定首帧，音色参考模式用 <Audio 1> 到 <Audio 3>。";
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
    if (voiceMode !== "none" && VOICE_REFERENCE_MODES.includes(mode)) {
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

    enforceDirectorSize(node);
    node.graph?.setDirtyCanvas(true, true);
    requestAnimationFrame(bindRenderedControllers);
  }

  const domWidget = node.addDOMWidget?.("minimax_h3_director_plus_ui", "custom", root, {
    serialize: false,
    hideOnZoom: false,
    getHeight: () => DIRECTOR_VIEWPORT_HEIGHT,
  });
  if (domWidget) domWidget.computeSize = () => [DIRECTOR_UI_WIDTH - DIRECTOR_CONTENT_INSET, DIRECTOR_VIEWPORT_HEIGHT];
  if (domWidget?.element) {
    domWidget.element.style.minWidth = `${DIRECTOR_MIN_CONTENT_WIDTH}px`;
    domWidget.element.style.width = `${DIRECTOR_UI_WIDTH - DIRECTOR_CONTENT_INSET}px`;
  }
  enforceDirectorSize(node);
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

  ["mode", "voice_mode", "performance_preset", "postprocess_mode", "rtx_quality", "motion_smoothing", "audio_loudness", "aspect_ratio", "resolution_preset", "seed_mode"].forEach((name) => {
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
