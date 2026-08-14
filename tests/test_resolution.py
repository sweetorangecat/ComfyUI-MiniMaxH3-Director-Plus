import pytest

from nodes.resolution import ASPECTS, MEGAPIXELS, calculate_resolution


def test_resolution_menu_has_fine_grained_megapixel_steps():
    values = sorted(MEGAPIXELS.values())

    assert len(values) >= 20
    assert 0.30 in values
    assert 0.40 in values
    assert 0.90 in values
    assert 1.10 in values
    assert 2.00 in values


@pytest.mark.parametrize(
    "aspect",
    ["1:1", "3:2", "2:3", "4:3", "3:4", "8:5", "5:8", "16:9", "9:16", "21:9", "9:21"],
)
def test_named_aspects_are_divisible_by_32(aspect):
    width, height = calculate_resolution("0.83 MP", aspect)

    assert width % 32 == 0
    assert height % 32 == 0


def test_all_named_aspects_are_explicitly_exposed():
    assert tuple(ASPECTS) == (
        "1:1", "3:2", "2:3", "4:3", "3:4", "8:5", "5:8", "16:9", "9:16", "21:9", "9:21"
    )


def test_landscape_and_portrait_are_not_hidden_behind_a_swap_toggle():
    landscape = calculate_resolution("0.83 MP", "16:9")
    portrait = calculate_resolution("0.83 MP", "9:16")

    assert landscape[0] > landscape[1]
    assert portrait[0] < portrait[1]


def test_custom_aspect_rejects_zero_component():
    with pytest.raises(ValueError, match="自定义比例"):
        calculate_resolution("0.83 MP", "CUSTOM", custom_width=0, custom_height=9)


def test_unknown_resolution_preset_is_rejected():
    with pytest.raises(ValueError, match="分辨率档位"):
        calculate_resolution("unknown", "16:9")
