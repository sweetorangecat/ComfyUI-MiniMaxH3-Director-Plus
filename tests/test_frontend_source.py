from pathlib import Path


SOURCE_PATH = Path("js/minimax_h3_director_plus_v9.js")


def source():
    return SOURCE_PATH.read_text(encoding="utf-8")


def test_upload_slots_restore_persistent_image_and_audio_previews():
    text = source()
    assert "function mediaViewUrl(filename)" in text
    assert "new URLSearchParams" in text
    assert 'type: "input"' in text
    assert 'query.set("subfolder", subfolder)' in text
    assert "function mediaPreview(filename, accept)" in text
    assert 'link.target = "_blank"' in text
    assert 'image.className = "h3p-image-preview"' in text
    assert "audio.controls = true" in text
    assert 'audio.preload = "metadata"' in text
    assert 'status.textContent = "图片预览不可用"' in text
    assert 'status.textContent = "音频预览不可用"' in text
    assert 'const wrapper = document.createElement("div");' in text
    assert "replaceMediaPreview(wrapper, result.asset, input.dataset.h3pAccept)" in text


def test_upload_success_registers_new_file_in_combo_widget_before_queueing():
    text = source()
    assert "function syncUploadWidget(node, widgetName, value)" in text
    assert "item.options?.values" in text
    assert "item.options.values.push(value)" in text
    assert "syncUploadWidget(node, input.dataset.h3pUploadFile, result.asset)" in text


def test_ui_has_chinese_primary_sections():
    text = source()
    for label in ("快速设置", "生成规格", "导演与素材", "音色参考", "实际后端", "高级音色锁定"):
        assert label in text


def test_ui_exposes_duration_aspect_and_resolution_controls():
    text = source()
    for label in ("视频时长", "画面比例", "分辨率档位", "最终尺寸"):
        assert label in text
    assert 'data-h3p-value-widget' in text
    assert 'setWidget(node, "width"' in text
    assert 'setWidget(node, "height"' in text
    for aspect in ("3:2", "4:3", "8:5", "16:9", "21:9"):
        assert aspect in text
    assert '"2K QHD", "4K UHD"' in text
    assert '"2K QHD|16:9": [2560, 1440]' in text
    assert '"4K UHD|16:9": [3840, 2160]' in text


def test_ui_exposes_one_final_output_postprocess_surface():
    text = source()
    for label in ("最终输出", "原生尺寸直出", "Lanczos 快速放大", "AI 自动超分", "AI 细节重建（RTX VSR）", "RTX VSR 质量", "AI 超分模型"):
        assert label in text
    assert 'valueControl("最终输出", "postprocess_mode"' in text
    assert 'valueControl("RTX VSR 质量", "rtx_quality"' in text
    assert 'valueControl("AI 超分模型", "ai_upscale_model"' in text
    assert 'postprocessMode === "ai_upscale"' in text
    assert 'const aiModelOptions' in text
    assert '"postprocess_mode", "rtx_quality", "ai_upscale_model", "timeline_data"' in text
    assert "同尺寸自动旁路" in text


def test_frontend_filters_performance_presets_by_mode_and_voice():
    text = source()
    assert "allowedPerformancePresets" in text
    assert 'voiceMode !== "none"' in text
    assert 'setWidget(node, "performance_preset", preset, false)' in text


def test_ui_keeps_audio_lane_for_i2va_and_fl2va():
    text = source()
    assert "I2VA" in text and "FL2VA" in text
    assert "voice_reference_audio" in text
    assert "音色参考 1" in text
    assert "音色参考 2" in text
    assert "音色参考 3" in text
    assert '"参考图 9", "reference_image_7_file"' in text
    assert "<Audio 1> / <Audio 2> / <Audio 3>" in text


def test_ref2va_ui_maps_nine_picture_slots_without_exceeding_native_limit():
    text = source()
    assert "REF2VA_IMAGE_SLOTS" in text
    assert "genericPictureLabels" in text
    assert "REF2VA 参考图均为普通参考图" in text
    assert '"参考图 1（起始构图建议）", "first_image_file"' in text
    assert '"参考图 2（结束构图建议）", "last_image_file"' in text
    assert '"参考图 9", "reference_image_7_file"' in text


def test_ui_uses_embedded_upload_widgets_instead_of_connection_instructions():
    text = source()
    assert "first_image_file" in text
    assert "last_image_file" in text
    assert "reference_image_1_file" in text
    assert "voice_reference_audio_file" in text
    assert "连接 first_image" not in text
    assert "连接 last_image" not in text
    assert "连接 voice_reference_audio" not in text


def test_each_mode_only_exposes_the_media_uploads_it_needs():
    text = source()
    assert '["I2VA", "FL2VA", "REF2VA"].includes(mode)' in text
    assert '["FL2VA", "L2VA", "REF2VA"].includes(mode)' in text
    assert '["I2VA", "FL2VA", "L2VA", "REF2VA"].includes(mode)' not in text


def test_dom_controls_stop_litegraph_pointer_events():
    text = source()
    assert 'element.addEventListener("click"' in text
    assert 'element.addEventListener("change"' in text
    assert "domWidget?.element" in text
    assert "MutationObserver" in text
    assert "dataset.h3pControl" in text
    assert "event.stopPropagation()" in text
    assert "bindDocumentControls" in text
    assert "controllerForTarget" in text
    assert 'valueControl("生成模式", "mode"' in text
    assert 'valueControl("音色模式", "voice_mode"' in text


def test_ui_explains_native_reference_routing():
    text = source()
    assert "H3 原生音色参考" in text
    assert "ref2va_model" in text
    assert "fl2va_model" in text


def test_ui_does_not_offer_copy_semantics():
    text = source()
    assert "fully_copy" not in text
    assert "partially_copy" not in text
    assert "audio reuse" not in text


def test_fish_controls_are_collapsed_by_default():
    text = source()
    assert "fishPanel.open = false" in text


def test_director_ui_uses_stable_size_when_sidebar_changes_available_width():
    text = source()
    assert "DIRECTOR_UI_WIDTH" in text
    assert "DIRECTOR_UI_HEIGHT" in text
    assert "node.setSize?.([DIRECTOR_UI_WIDTH, DIRECTOR_UI_HEIGHT])" in text
    assert "node.size?.[0] || 500" not in text
    assert "width:${DIRECTOR_UI_WIDTH}px" in text
    assert "min-width:${DIRECTOR_UI_WIDTH}px" in text
    assert "flex:0 0 ${DIRECTOR_UI_WIDTH}px" in text
    assert ".h3p select,.h3p input,.h3p textarea" in text
    assert "max-width:100%;min-width:0" in text


def test_fish_mode_exposes_model_dialogue_and_sample_transcript_controls():
    text = source()
    assert 'voiceMode === "fish_lock"' in text
    assert 'valueControl("Fish 模型", "fish_model_path"' in text
    assert 'valueControl("新的目标对白", "target_dialogue"' in text
    assert 'valueControl("音色样本文本（建议填写）", "reference_transcript"' in text
    assert 'type === "textarea"' in text
    assert 'input.value = current ?? ""' in text


def test_prompt_assistant_is_visible_and_writes_back_to_prompt_widget():
    text = source()
    assert "if (!window.PromptAssistant_Version)" in text
    assert "提示词小助手" in text
    assert "promptTemplate" in text
    assert "插入 H3 结构" in text
    assert "插入 <Picture 1>" in text
    assert "插入 <Audio 1>" in text
    assert 'setWidget(node, "prompt", prompt.value)' in text
    assert "h3p-prompt-help" in text
    assert 'const promptWidget = widget(node, "prompt")' in text
    assert "promptWidget.inputEl = prompt" in text
    assert "promptWidget.element = prompt" in text
    assert "const mountPromptAssistant = () =>" in text
    assert "window.promptAssistant || app.promptAssistant" in text
    assert "assistant.checkAndSetupNode(node)" in text
    assert "assistant.cleanup?.(node.id, true)" in text
    assert 'promptAnchor.className = "h3p-prompt-anchor dom-widget"' in text
    assert "keepPromptAssistantReadable(promptAnchor)" in text
    assert 'assistant.style.setProperty("--assistant-scale", String(1 / canvasScale))' in text
    assert "setTimeout(mountPromptAssistant, 250)" in text
    assert "const mountedPrompt = root.querySelector" in text


def test_ui_adds_seed_without_reordering_existing_sections():
    text = source()
    for label in ("噪音种子", "种子模式", "固定", "每次递增", "每次递减", "每次随机"):
        assert label in text
    sections = [
        "<span>快速设置</span>",
        "<span>生成规格</span>",
        "<span>导演与素材</span>",
        "<span>音色参考</span>",
    ]
    assert [text.index(label) for label in sections] == sorted(text.index(label) for label in sections)
    assert 'valueControl("噪音种子", "seed"' in text
    assert 'valueControl("种子模式", "seed_mode"' in text
