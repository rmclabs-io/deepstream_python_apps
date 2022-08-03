from typing import Generator, Tuple

import numpy as np
import pyds
from pyds_labs.maskrcnn import extract_maskrcnn_mask


def glist_iter(container_list, cast):
    while container_list:
        try:
            meta = cast(container_list.data)
        except StopIteration:
            break
        yield meta
        try:
            container_list = container_list.next
        except StopIteration:
            break


def glist_iter_conditioned(
    container_list,
    cast,
    condition,
):
    for meta in glist_iter(container_list, cast):
        if condition(meta):
            yield meta
        else:
            print(f"{meta} does not fulfill {condition}")


def glist_iter_internal(
    container_list,
    cast,
    condition,
    second_cast,
):
    for meta in glist_iter_conditioned(container_list, cast, condition):
        try:
            yield second_cast(meta.user_meta_data)
        except StopIteration:
            break


def frames_per_batch(
    batch_meta: pyds.NvDsBatchMeta,
) -> Generator[pyds.NvDsFrameMeta, None, None]:
    for frame_meta in glist_iter(batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast):
        yield frame_meta
    return batch_meta


def objects_per_frame(
    frame_meta: pyds.NvDsFrameMeta,
) -> Generator[pyds.NvDsObjectMeta, None, None]:
    yield from glist_iter(frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast)


def is_analytics_meta(user_meta: pyds.NvDsUserMeta) -> bool:
    return user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type(
        "NVIDIA.DSANALYTICSOBJ.USER_META"
    )


def is_frameanalytics_meta(user_meta: pyds.NvDsUserMeta) -> bool:
    return user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type(
        "NVIDIA.DSANALYTICSFRAME.USER_META"
    )


def is_segmentation_meta(user_meta: pyds.NvDsUserMeta) -> bool:
    return user_meta.base_meta.meta_type == pyds.NVDSINFER_SEGMENTATION_META


def analytics_per_obj(
    obj_meta: pyds.NvDsObjectMeta,
) -> Generator[pyds.NvDsAnalyticsObjInfo, None, None]:
    yield from glist_iter_internal(
        obj_meta.obj_user_meta_list,
        pyds.NvDsUserMeta.cast,
        condition=is_analytics_meta,
        second_cast=pyds.NvDsAnalyticsObjInfo.cast,
    )


def classification_per_obj(
    obj_meta: pyds.NvDsObjectMeta,
) -> Generator[pyds.NvDsClassifierMeta, None, None]:
    yield from glist_iter(
        obj_meta.classifier_meta_list,
        pyds.NvDsClassifierMeta.cast,
    )


def labels_per_classification(
    classifier_meta: pyds.NvDsClassifierMeta,
) -> Generator[pyds.NvDsLabelInfo, None, None]:
    yield from glist_iter(
        classifier_meta.label_info_list,
        pyds.NvDsLabelInfo.cast,
    )


def labels_per_obj(
    obj_meta: pyds.NvDsObjectMeta,
) -> Generator[pyds.NvDsLabelInfo, None, None]:
    for classifier_meta in classification_per_obj(obj_meta):
        yield from labels_per_classification(classifier_meta)


def analytics_per_frame(
    frame_meta: pyds.NvDsFrameMeta,
) -> Generator[pyds.NvDsAnalyticsFrameMeta, None, None]:
    yield from glist_iter_internal(
        frame_meta.frame_user_meta_list,
        pyds.NvDsUserMeta.cast,
        condition=is_frameanalytics_meta,
        second_cast=pyds.NvDsAnalyticsFrameMeta.cast,
    )


def semantic_masks_per_frame(
    frame_meta: pyds.NvDsFrameMeta,
) -> Generator[np.ndarray, None, None]:
    for segmeta in glist_iter_internal(
        frame_meta.frame_user_meta_list,
        pyds.NvDsUserMeta.cast,
        condition=is_segmentation_meta,
        second_cast=pyds.NvDsInferSegmentationMeta.cast,
    ):
        masks_ = pyds.get_segmentation_masks(segmeta)
        masks = np.array(masks_, copy=True, order="C")
        yield masks


def objects_per_batch(batch_meta):
    for frame_meta in frames_per_batch(batch_meta):
        for obj_meta in objects_per_frame(frame_meta):
            yield frame_meta, obj_meta


def instance_masks_per_batch(
    batch_meta: pyds.NvDsBatchMeta,
) -> Generator[Tuple[pyds.NvDsFrameMeta, pyds.NvDsObjectMeta, np.ndarray], None, None]:
    for frame_meta, obj_meta in objects_per_batch(batch_meta):
        yield frame_meta, obj_meta, extract_maskrcnn_mask(obj_meta)
