import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function element(tag, text) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  return item;
}

function attachProgress(node) {
  const root = element("div");
  root.style.cssText = "height:100%;padding:10px;box-sizing:border-box;color:var(--input-text);background:var(--comfy-input-bg);font:13px sans-serif;overflow:auto;";
  const status = element("p", "长视频默认关闭。开启后沿原 U11 质量链路逐窗生成；每窗原生画面确认后再增强和续拍。");
  const actions = element("div");
  actions.style.cssText = "display:flex;gap:8px;flex-wrap:wrap";
  const content = element("div");
  root.append(status, actions, content);
  const progressWidget = node.addDOMWidget("long_video_progress", "div", root, {
    serialize: false, hideOnZoom: false, getHeight: () => 360,
  });
  progressWidget.computeSize = () => [500, 360];
  node.setSize([Math.max(node.size[0], 520), Math.max(node.size[1], node.computeSize()[1])]);
  let timer;
  let removed = false;
  const project = () => node.widgets?.find((w) => w.name === "project_name")?.value || "h3-native-av";
  const endpoint = () => "/h3-director-plus/long-video/" + encodeURIComponent(project());

  async function request(action, data = {}) {
    try {
      const response = await api.fetchApi(endpoint() + "/control", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, ...data }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || response.statusText);
      render(result);
    } catch (error) {
      status.textContent = String(error.message || error);
    }
  }

  function button(parent, label, callback) {
    const item = element("button", label);
    item.type = "button";
    item.addEventListener("click", callback);
    parent.append(item);
  }

  async function refresh() {
    clearTimeout(timer);
    if (removed) return;
    try {
      const response = await api.fetchApi(endpoint());
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      render(data);
    } catch (error) {
      status.textContent = String(error.message || error);
    }
  }

  function render(data) {
    clearTimeout(timer);
    const labels = { new: "尚未开始", preparing: "核对项目和素材中", queued: "等待本窗开始", generating: "原生生成中", decoding: "保存原生预览中",
      awaiting_review: "待确认本窗", enhancing: "沿原 U11 增强中", assembling: "装配完整视频中",
      complete: "已完成", awaiting_extension: "已到当前剧情结尾，可扩展剧情后继续", stopped: "已停止续拍",
      stopping: "本窗保存后停止", failed: "运行失败，已保留检查点", rejected: "本窗需要重生成", ready: "等待继续" };
    const completed = Number(data.completed_seconds || 0).toFixed(2);
    const total = Number(data.actual_seconds || 0).toFixed(2);
    status.textContent = `${labels[data.status] || data.status} · 已确认 ${completed} / ${total} 秒` + (data.error ? `\n${data.error}` : "");
    content.replaceChildren();
    const window = data.windows?.find((w) => w.status === "awaiting_review" || w.status === "rejected" || w.can_retry) || data.windows?.at(-1);
    if (window?.preview_url) {
      content.append(element("p", `第 ${window.index + 1} 窗 · 交付 ${window.start.toFixed(2)}–${window.end.toFixed(2)} 秒；预览含保护历史和生成余量。`));
      const video = element("video");
      video.controls = true;
      video.preload = "metadata";
      video.src = api.apiURL(window.preview_url);
      video.style.width = "100%";
      content.append(video);
    }
    if (window?.status === "awaiting_review") {
      content.append(element("p", "检查原生动作、声音、口型、手和道具，以及首个自由帧与上一窗的接点。填写实测尾态后继续。"));
      const outgoing = element("textarea");
      outgoing.placeholder = "实际尾态：人物位置、手持物、动作阶段、机位和声音状态";
      const scope = element("textarea");
      scope.placeholder = "已检查的范围与发现（含自由边界、对白和收尾）";
      for (const field of [outgoing, scope]) {
        field.style.cssText = "width:100%;box-sizing:border-box;min-height:70px;margin:4px 0";
        content.append(field);
      }
      button(content, "接受本窗，增强并继续", () => {
        if (!outgoing.value.trim() || !scope.value.trim()) {
          status.textContent = "请记录实际尾态和检查范围。";
          return;
        }
        request("review", { take_id: window.take_id, accepted: true,
          observed_outgoing_state: outgoing.value, scope: scope.value });
      });
      button(content, "退回本窗", () => request("review", { take_id: window.take_id, accepted: false,
        observed_outgoing_state: outgoing.value, scope: scope.value || "已退回，需重新检查原生内容" }));
    }
    if (window?.can_retry) {
      button(content, "放弃本次局部返修，重新生成本窗", () => request("retry", { index: window.index }));
    }
    if (window?.status === "rejected") {
      button(content, "重新生成本窗", () => request("retry", { index: window.index }));
      const detail = element("details");
      detail.append(element("summary", "局部返修（锁定音频）"));
      const ranges = element("input");
      ranges.placeholder = "视频 latent 时间步范围，例如 [[20,30]]";
      ranges.style.width = "100%";
      detail.append(ranges);
      button(detail, "按范围返修", () => {
        try {
          request("repair", { index: window.index, video_ranges: JSON.parse(ranges.value), lock_audio: true });
        } catch {
          status.textContent = "范围应为整数区间数组，例如 [[20,30]]。";
        }
      });
      content.append(detail);
    }
    if (data.output_url) {
      const link = element("a", "查看完整成片");
      link.href = api.apiURL(data.output_url);
      link.target = "_blank";
      content.append(element("p"), link);
    }
    node.setDirtyCanvas(true, true);
    if (["preparing", "generating", "queued", "decoding", "enhancing", "assembling", "stopping", "ready"].includes(data.status)) {
      timer = setTimeout(refresh, 3000);
    }
  }
  button(actions, "刷新进度", refresh);
  button(actions, "继续", () => request("resume"));
  button(actions, "停止续拍", () => request("stop"));
  const original = node.onRemoved;
  node.onRemoved = function () {
    removed = true;
    clearTimeout(timer);
    original?.apply(this, arguments);
  };
  node._h3LongRefresh = refresh;
}

app.registerExtension({
  name: "MiniMaxH3.NativeAVLongVideo",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "MiniMaxH3LongVideoConfig") {
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        created?.apply(this, arguments);
        attachProgress(this);
      };
    }
    if (nodeData.name === "MiniMaxH3U11Output") {
      const executed = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        executed?.apply(this, arguments);
        if (message.h3_long_video) {
          for (const node of app.graph?._nodes || []) node._h3LongRefresh?.();
        }
      };
    }
  },
});
