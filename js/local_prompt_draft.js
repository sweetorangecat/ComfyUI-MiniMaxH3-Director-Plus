import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "MiniMaxH3.LocalPromptDraft",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MiniMaxH3LocalPromptDraft") return;
    const original = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      original?.apply(this, arguments);
      if (!this._h3DraftPreview) {
        const text = document.createElement("textarea");
        text.readOnly = true;
        text.style.cssText = "width:100%;height:100%;box-sizing:border-box;resize:none;white-space:pre-wrap;";
        this.addDOMWidget("draft_preview", "text", text, { serialize: false });
        this._h3DraftPreview = text;
      }
      this._h3DraftPreview.value = (message.text || []).join("\n\n");
      this.setDirtyCanvas(true, true);
    };
  },
});
