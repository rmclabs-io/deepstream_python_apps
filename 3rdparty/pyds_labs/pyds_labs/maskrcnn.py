from typing import Tuple

import numpy as np
import pyds


def _gen_ranges(
    original_height,
    original_width,
    target_height,
    target_width,
):
    ratio_h = float(original_height / target_height)
    ratio_w = float(original_width / target_width)

    h = np.arange(0, target_height, ratio_h)[
        :target_height
    ]  # its y0 : TODO why is the arange so large!?
    w = np.arange(0, target_width, ratio_w)[
        :target_width
    ]  # its x0 : TODO why is the arange so large!?
    return h, w


def _clip(value, low, up):
    return np.clip(value, 0.0, up)


def _gen_clips(
    w,
    original_width,
    h,
    original_height,
):

    w_left = np.clip(np.floor(w), 0.0, original_width - 1)  # its left
    w_right = np.clip(np.ceil(w), 0.0, original_width - 1)  # its right
    h_top = np.clip(np.floor(h), 0.0, original_height - 1)  # its top
    h_bottom = np.clip(np.ceil(h), 0.0, original_height - 1)  # its bottom
    return w_left, w_right, h_top, h_bottom


def _gen_idxs(
    original_height,
    original_width,
    w_left,
    w_right,
    h_top,
    h_bottom,
):

    left_top_idx = np.add.outer(h_top * original_width, w_left).astype(int)
    right_top_idx = np.add.outer(h_top * original_width, w_right).astype(int)
    left_bottom_idx = np.add.outer(h_bottom * original_width, w_left).astype(int)
    right_bottom_idx = np.add.outer(h_bottom * original_width, w_right).astype(int)

    return left_top_idx, right_top_idx, left_bottom_idx, right_bottom_idx


def _take_vals(
    src,
    *idxmats,
):
    return tuple(src.take(idxmat) for idxmat in idxmats)


def _interpolate(
    w,
    w_left,
    h,
    h_top,
    left_top_val,
    right_top_val,
    left_bottom_val,
    right_bottom_val,
):
    delta_w = w - w_left
    top_lerp = left_top_val + (right_top_val - left_top_val) * delta_w
    bottom_lerp = left_bottom_val + (right_bottom_val - left_bottom_val) * delta_w
    return top_lerp + ((bottom_lerp - top_lerp).T * (h - h_top)).T


def resize_mask_vec(
    src: np.ndarray,
    src_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    threshold: float,
) -> np.ndarray:
    """Resize mask from original deepstream object into numpy array."""

    original_height, original_width = src_shape
    target_height, target_width = target_shape

    h, w = _gen_ranges(
        original_height,
        original_width,
        target_height,
        target_width,
    )

    w_left, w_right, h_top, h_bottom = _gen_clips(
        w,
        original_width,
        h,
        original_height,
    )

    left_top_idx, right_top_idx, left_bottom_idx, right_bottom_idx = _gen_idxs(
        original_height,
        original_width,
        w_left,
        w_right,
        h_top,
        h_bottom,
    )

    left_top_val, right_top_val, left_bottom_val, right_bottom_val = _take_vals(
        src,
        left_top_idx,
        right_top_idx,
        left_bottom_idx,
        right_bottom_idx,
    )

    lerp = _interpolate(
        w,
        w_left,
        h,
        h_top,
        left_top_val,
        right_top_val,
        left_bottom_val,
        right_bottom_val,
    )

    ret = np.zeros_like(lerp, dtype=np.uint8)
    ret[lerp >= threshold] = 255
    return ret


def extract_maskrcnn_mask(obj_meta: pyds.NvDsObjectMeta) -> np.ndarray:
    rect_height = int(np.ceil(obj_meta.rect_params.height))
    rect_width = int(np.ceil(obj_meta.rect_params.width))
    return resize_mask_vec(
        obj_meta.mask_params.data,
        (obj_meta.mask_params.height, obj_meta.mask_params.width),
        (rect_height, rect_width),
        obj_meta.mask_params.threshold,
    )
