"""The measured short-clip geometry, scoped to internal editorial requests."""
import math

PROFILE_ID = 'short_1088_selflift_v1'
META_KEY = 'h3_short_quality_reference'
LIFTER_SHA256 = '6e0f35692784ba2376704a48a90fb07f90aef76ef21200d5525d6627a54fe909'


def eligible(settings):
    try:
        aspect = settings.get('aspect_ratio', '16:9')
        sides = ((settings.get('custom_width',16), settings.get('custom_height',9))
                 if aspect == 'CUSTOM' else str(aspect).split(':'))
        ratio = float(sides[0]) / float(sides[1])
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        return False
    return (9/16 <= ratio <= 16/9 and settings.get('mode') == 'REF2VA'
        and settings.get('sampling_method') == 'selflift'
        and settings.get('performance_preset') in {'quality_two_stage', '质量优先二采样'}
        and settings.get('community_lora_profile') == 'continuity'
        and settings.get('resolution_preset') == '1080p FHD'
        and settings.get('postprocess_mode') == 'vosr2'
        and settings.get('voice_mode', 'none') in {'none', 'h3_reference'})


def requested(prompt_graph, node_id):
    if not isinstance(prompt_graph, dict) or node_id is None:
        return False
    meta = prompt_graph.get(str(node_id), {}).get('_meta', {})
    return (meta.get('h3_long_video_entry') == 'independent_shots'
            and meta.get(META_KEY) == PROFILE_ID)


def plan(width, height, duration, total, free):
    """Keep smaller machines on their old plan; never shrink a qualified one."""
    width, height = int(width), int(height)
    total, free = float(total), float(free)
    if not all(math.isfinite(value) for value in (total, free)):
        raise ValueError('短片画质基准无法读取有效显存')
    if total < 28 or not 4 <= duration <= 15 or not 1080 <= min(width, height) <= 1440:
        return None
    high_w, high_h = (math.ceil(side / 32) * 32 for side in (width, height))
    # Allow the existing FHD pixel budget's32px rounding (including8:5).
    if high_w * high_h > 1920 * 1088 * 1.02:
        raise ValueError('短片画质基准不支持该画布预算；不会自动回退低分辨率')
    if free < 24:
        raise ValueError('短片1088p画质基准需要至少24GB空闲显存；不会自动缩小采样网格')
    low_w, low_h = (max(2, round(side / 16 * .7 / 2) * 2) * 16 for side in (high_w, high_h))
    return {'allowed':True, 'reason':'采用用户认可的短片原生画质网格；仍需逐段审看片段',
        'first_stage_width':low_w, 'first_stage_height':low_h,
        'second_stage_width':high_w, 'second_stage_height':high_h,
        'second_stage_megapixels':high_w * high_h / 1_000_000,
        'two_stage_tiling_required':False,
        'final_scale_x':1.0, 'final_scale_y':1.0, 'final_scale':1.0,
        'max_final_vsr_scale':1.0, 'vram_safety_tier':'short_reference_32gb',
        'quality_basis':'已认可短片网格 / SelfLift6+4 / 原生FHD直出',
        'short_quality_reference':PROFILE_ID, 'selflift_lowres_scale_requested':.7}


def validate_output(guide, images):
    """Check actual producer evidence; never turn it into visual approval."""
    if guide.get('short_quality_reference') != PROFILE_ID:
        return None
    width, height = (int(guide.get(key, 0)) for key in ('second_stage_width', 'second_stage_height'))
    from .execution_report import report_from
    report = report_from(images) or {}
    producer = report.get('sampling_settings') or {}
    producer_plan = report.get('current_plan') or report.get('plan') or {}
    stages = [stage for stage in report.get('stages',[]) if stage.get('stage') == 'sampling']
    expected = {'selflift_lowres_scale':.7, 'selflift_transition_step':6,
                'selflift_cfg':1.0, 'selflift_rho':0.0,
                'selflift_highres_tiling_mode':'global_stream',
                'sol_attn_applied':True, 'selflift_lifter':'trained_2d_tier3',
                'sampling_actual_method':'selflift','steps':10,'sigma_count':11,
                'sampler':'euler','scheduler':'beta', 'video_shift':12.0,'audio_shift':3.0,
                'two_stage_first_executed_steps':6,'two_stage_second_executed_steps':4,
                'width':width,'height':height,'short_quality_reference':PROFILE_ID}
    lifter = producer.get('selflift_lifter_evidence') or {}
    first = [max(2,round(side / 16 * .7 / 2) * 2) * 16 for side in (width,height)]
    if (len(images.shape) != 4 or tuple(images.shape[1:3]) != (height,width)
            or min(width,height) < 1088
            or any(producer.get(key) != value for key,value in expected.items())
            or producer_plan.get('short_quality_reference') != PROFILE_ID
            or [producer_plan.get('width'),producer_plan.get('height')] != [width,height]
            or [guide.get('first_stage_width'),guide.get('first_stage_height')] != first
            or not stages or stages[-1].get('status') != 'completed'
            or stages[-1].get('steps') != 10 or stages[-1].get('path') != 'SelfLiftAvatarH3Sampler'
            or lifter.get('tier') != 3 or lifter.get('checkpoint_sha256') != LIFTER_SHA256):
        raise ValueError('实际执行未保持已认可短片画质基准，已阻止作为基准长片发布；请检查采样报告')
    return {'profile':PROFILE_ID, 'canvas':[width,height],
            'first_canvas':first,
            'lowres_scale':.7, 'steps':[6,4], 'lifter_sha256':LIFTER_SHA256,
            'keyframe_policy':'original_workflow_only', 'content_review':'pending'}
