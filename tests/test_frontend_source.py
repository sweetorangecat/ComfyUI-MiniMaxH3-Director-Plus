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


def test_upload_controls_support_non_destructive_remove_and_slot_compaction():
    text = source()
    assert 'data-h3p-remove-file' in text
    assert 'button.textContent = "移除"' in text
    assert 'const normalized = value || ""' in text
    assert 'compactSlots' in text
    assert 'compactBoundSlots' in text
    assert '\u53c2\u8003\u56fe\u5df2\u91cd\u65b0\u7f16\u53f7\uff0c\u8bf7\u68c0\u67e5\u63d0\u793a\u8bcd\u4e2d\u7684 <Picture N> \u5f15\u7528\u3002' in text
    assert '\u97f3\u8272\u53c2\u8003\u5df2\u91cd\u65b0\u7f16\u53f7\uff0c\u8bf7\u68c0\u67e5\u63d0\u793a\u8bcd\u4e2d\u7684 <Audio N> \u4e0e\u89d2\u8272\u540d\u3002' in text
    assert 'setAttribute("role", "status")' in text
    assert 'method: "DELETE"' not in text
    assert 'unlink' not in text.lower()


def test_upload_controls_expose_replace_and_non_destructive_remove_contract():
    text = source()
    assert 'button.textContent = filename ? "\u66f4\u6362" : "\u9009\u62e9\u6587\u4ef6"' in text
    assert 'data-h3p-remove-file' in text
    assert 'const normalized = value || ""' in text
    assert 'callback?.(normalized)' in text
    assert 'setDirtyCanvas(true, true)' in text
    assert 'method: "DELETE"' not in text
    assert "unlink" not in text


def test_upload_removal_compacts_ref_images_and_bound_audio_names():
    text = source()
    assert "compactSlots" in text
    assert "compactBoundSlots" in text
    assert "REF2VA_IMAGE_SLOTS" in text
    assert "voice_reference_name_1" in text
    assert "\u53c2\u8003\u56fe\u5df2\u91cd\u65b0\u7f16\u53f7" in text
    assert "\u97f3\u8272\u53c2\u8003\u5df2\u91cd\u65b0\u7f16\u53f7" in text


def test_endpoint_removal_is_independent_from_ref_compaction():
    text = source()
    assert "clearSlot" in text
    assert "mode === \"REF2VA\"" in text
    assert "\u9009\u62e9\u6587\u4ef6" in text


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


def test_ui_has_visible_workbench_status_layer_inside_director_dom():
    text = source()
    for marker in (
        "h3p-workbench-bar",
        "h3p-status-strip",
        "h3p-status-item",
        "工作台状态",
        "模式 / 后端",
        "规格 / 时长",
        "素材 / 音色",
        'setAttribute("data-h3p-workbench", "true")',
    ):
        assert marker in text


def test_ui_exposes_separate_low_vram_two_stage_without_changing_layout():
    text = source()

    assert '"低显存", "低显存二采"' in text
    assert '"1080p FHD"' in text
    assert "低显存二采" in text
    assert 'valueControl("最终输出", "postprocess_mode"' in text
    assert 'valueControl("RTX VSR 质量", "rtx_quality"' in text
    assert 'valueControl("AI 超分模型", "ai_upscale_model"' in text
    assert 'postprocessMode === "ai_upscale"' in text
    assert 'const aiModelOptions' in text
    assert '"postprocess_mode", "rtx_quality", "ai_upscale_model", "motion_smoothing", "audio_loudness", "timeline_data"' in text
    assert "同尺寸自动旁路" in text


def test_ui_exposes_route_isolated_quality_presets():
    text = source()
    assert '"免费智能 1080p"' in text
    assert 't2va: ["免费智能 1080p"' in text
    assert 'reference: ["免费智能 1080p"' in text
    assert "高清快速（v4 8步）" in text
    assert "参考高清（原生20步）" in text
    assert "参考极速（官方4步）" in text
    assert "fl_quality_fast_v4" in text
    assert "ref_quality_native" in text
    assert "ref_fast_4step" in text
    assert "v4 8步仅适用于 FL/T2V 后端" in text


def test_ui_locks_quality_two_stage_to_rtx_vsr():
    text = source()
    assert "POSTPROCESS_MODES_BY_PERFORMANCE" in text
    assert '"质量优先二采样": [["rtx_vsr"' in text
    assert "allowedPostprocessModes(preset)" in text
    assert "质量优先二采样已锁定 RTX VSR" in text
    assert "单次 RTX VSR" in text


def test_ui_locks_low_vram_two_stage_to_six_second_fhd_ai_x2_reconstruction():
    text = source()

    assert '"低显存二采": [["ai_upscale"' in text
    assert 'preset === "低显存二采"' in text
    assert 'setWidget(node, "duration", 4, false)' not in text
    assert 'Number(widget(node, "duration")?.value) > 6' in text
    assert 'setWidget(node, "duration", 6, false)' in text
    assert 'setWidget(node, "resolution_preset", "1080p FHD", false)' in text
    assert "resolvedWidth * resolvedHeight > 1920 * 1080 * 1.02" in text
    assert "function lowVramFirstStageMegapixels" in text
    assert "LOW_VRAM_TWO_STAGE_MAX_FINAL_SCALE" in text
    assert "lowVramFirstStageMegapixels(resolvedWidth, resolvedHeight)" in text
    assert "低显存二采已锁定 AI X2 细节重建" in text
    assert "最长 6 秒" in text
    assert "postprocessGrid.children[1]" in text
    assert '"AI X2"' in text
    assert "isX2UpscaleModel" in text
    assert 'preset !== "低显存二采" || isX2UpscaleModel(value)' in text


def test_ui_makes_smart_free_1080p_fully_automatic():
    text = source()

    assert '"免费智能 1080p": [["ai_upscale"' in text
    assert 'preset === SMART_PRESET' in text
    assert 'setWidget(node, "ai_upscale_model", SMART_UPSCALE_MODEL, false)' in text
    assert 'const SMART_UPSCALE_MODEL = "RealESRGAN_x2plus.pth"' in text
    assert '关闭（智能锁定）' in text
    assert "按后端、显存和时长自动选择路线" in text
    assert "无需手动组合超分、模型和运动平滑" in text


def test_named_resolution_presets_use_explicit_megapixels_for_nonstandard_aspects():
    text = source()

    assert '"1080p FHD": 1920 * 1080 / (1024 * 1024)' in text
    assert '"2K QHD": 3.6864' in text
    assert '"4K UHD": 8.2944' in text
    assert "RESOLUTION_MEGAPIXELS[preset] ?? Number.parseFloat(preset)" in text


def test_ui_isolates_high_bitrate_vsr_in_the_existing_quality_control():
    text = source()
    assert '["HIGHBITRATE_ULTRA", "HIGHBITRATE_ULTRA（原画源最高保真）"]' in text
    assert "function allowedRtxQualities(performancePreset)" in text
    assert 'performancePreset === "质量优先二采样"' in text
    assert "allowedRtxQualities(preset)" in text
    assert 'setWidget(node, "rtx_quality", rtxQuality, false)' in text
    assert text.count('valueControl("RTX VSR 质量", "rtx_quality"') == 1


def test_two_stage_ui_names_trained_route_and_real_size_stages():
    text = source()
    assert "训练型 3D latent 二采" in text
    assert "H3首采" in text
    assert "神经latent二采" in text
    assert "最终输出" in text
    assert "双线性放大" not in text
    assert "前 6 步" not in text
    assert "最后 2 步" not in text
    assert "twoStageSizeHint" in text


def test_ui_exposes_compatible_chinese_motion_smoothing_control():
    text = source()
    assert 'valueControl("运动平滑", "motion_smoothing"' in text
    assert "allowedMotionSmoothing" in text
    assert '["off", "关闭（默认，保留原始帧）"]' in text
    assert '["rife_x2", "RIFE 2x（48 FPS）"]' in text
    assert 'preset === "质量优先二采样"' in text
    assert '关闭（二采固定，避免重影）' in text
    assert 'preset === "低显存"' in text
    assert 'postprocessMode !== "rtx_vsr"' in text
    assert '"motion_smoothing", "audio_loudness", "timeline_data"' in text


def test_ui_exposes_final_audio_loudness_control():
    text = source()
    assert 'valueControl("最终音频", "audio_loudness"' in text
    assert '["auto", "自动增强响度"]' in text
    assert '["original", "保持原始响度"]' in text


def test_frontend_filters_performance_presets_by_mode_and_voice():
    text = source()
    assert '"质量优先二采样"' in text
    assert "allowedPerformancePresets" in text
    assert 'voiceMode !== "none"' in text
    assert 'voiceMode === "fish_lock"' in text
    assert 'setWidget(node, "performance_preset", preset, false)' in text


def test_frontend_exposes_only_two_best_performance_presets_per_route():
    text = source()
    assert 't2va: ["免费智能 1080p", "稳定质量"]' in text
    assert 'endpoint: ["免费智能 1080p", "稳定质量"]' in text
    assert 'reference: ["免费智能 1080p", "参考高清（原生20步）"]' in text


def test_ui_keeps_audio_lane_for_all_reference_compatible_modes():
    text = source()
    assert "I2VA" in text and "FL2VA" in text
    assert "voice_reference_audio" in text
    assert "音色参考 1" in text
    assert "音色参考 2" in text
    assert "音色参考 3" in text
    assert '"参考图 9", "reference_image_7_file"' in text
    assert "<Audio 1> / <Audio 2> / <Audio 3>" in text
    assert 'const VOICE_REFERENCE_MODES = ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"]' in text


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
    assert '["FL2VA", "L2VA", "REF2VA"].includes(mode)' in text
    assert 'VOICE_REFERENCE_MODES.includes(mode)' in text


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
    assert ".h3p{box-sizing:border-box;width:100%;min-width:0;max-width:100%;" in text
    assert "overflow-y:auto" in text
    assert "overflow-x:hidden" in text
    assert "DIRECTOR_CONTENT_INSET" in text
    assert "DIRECTOR_UI_WIDTH - DIRECTOR_CONTENT_INSET" in text
    assert "max-width:none" not in text
    assert "flex:0 0 ${DIRECTOR_UI_WIDTH}px" not in text
    assert ".h3p select,.h3p input,.h3p textarea" in text
    assert "max-width:100%;min-width:0" in text
    assert "const DIRECTOR_UI_WIDTH = 1350;" in text
    assert "const DIRECTOR_UI_HEIGHT = 1510;" in text
    assert "const DIRECTOR_DOM_HEIGHT = 1050;" in text


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
