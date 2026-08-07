import pytest

from phase.common.slice_utils import descriptor_frame_to_source_frame


def test_descriptor_frame_maps_through_stride_slice():
    assert descriptor_frame_to_source_frame(0, "::5", 30) == 0
    assert descriptor_frame_to_source_frame(3, "::5", 30) == 15


def test_descriptor_frame_maps_through_bounded_slice():
    assert descriptor_frame_to_source_frame(0, "2:12:3", 20) == 2
    assert descriptor_frame_to_source_frame(3, "2:12:3", 20) == 11


def test_descriptor_frame_rejects_out_of_range_index():
    with pytest.raises(ValueError, match="outside the sliced trajectory"):
        descriptor_frame_to_source_frame(4, "2:12:3", 20)
