import ctypes
from logging import getLogger
from typing import Any, Callable, Dict, Optional, Tuple

import pyds
from pyds_labs import GLib, Gst
from pyds_labs.types import Gst_PadProbeCallback, Loop

logger = getLogger(__name__)


def build_pipeline(
    pipeline_template,
    **kw,
) -> Tuple[Gst.Bin, str]:
    pipeline_str = "\n".join(
        line
        for line in pipeline_template.splitlines()
        if not line.strip().startswith("#")
    ).format(**kw)
    logger.info(f"Constructing pipeline:\n```{pipeline_str}```")
    return Gst.parse_launch(pipeline_str), pipeline_str


def bus_call(bus, message, loop: Loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        logger.info("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logger.warning("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True


def long_to_uint64(long):
    value = ctypes.c_uint64(long & 0xFFFFFFFFFFFFFFFF).value
    return value


def get_element(pipeline, name):
    element = pipeline.get_by_name(name)
    if element:
        return element
    raise NameError(name)


def get_srcpad(pipeline, element_name):
    pad = get_element(pipeline, element_name).get_static_pad("src")
    if pad:
        return pad
    raise NameError(f"{element_name} has no srcpad")


def get_sinkpad(pipeline, element_name):
    pad = get_element(pipeline, element_name).get_static_pad("sink")
    if pad:
        return pad
    raise NameError(f"{element_name} has no sinkpad")


def inject_external_classification(
    batch_meta: pyds.NvDsBatchMeta,
    obj_meta: pyds.NvDsObjectMeta,
    **data: Dict[str, Any],
):

    classifier_meta = pyds.nvds_acquire_classifier_meta_from_pool(batch_meta)
    label_info = pyds.nvds_acquire_label_info_meta_from_pool(batch_meta)

    for name, value in data.items():
        setattr(label_info, name, value)

    pyds.nvds_add_label_info_meta_to_classifier(classifier_meta, label_info)
    pyds.nvds_add_classifier_meta_to_object(obj_meta, classifier_meta)

    if "label" in data:
        label = data["label"]
        txt_params = obj_meta.text_params
        original = pyds.get_string(txt_params.display_text)
        obj_meta.text_params.display_text = f"{original} {label}"


class GstPyApp:
    def __init__(
        self,
        pipeline_str: str,
        **pipeline_kw: Any,
    ):
        Gst.init(None)
        self.pipeline_kw = pipeline_kw
        self.pipeline, self.pipeline_str = build_pipeline(
            pipeline_str, **self.pipeline_kw
        )
        self._probes: Dict[str, int] = {}

    @property
    def probes(self):
        return self._probes

    def add_probe(
        self,
        element_name: str,
        pad_getter_fcn: Callable[[Gst.Pipeline, str], Gst.Pad],
        callback: Gst_PadProbeCallback,
        kind: Gst.PadProbeType = Gst.PadProbeType.BUFFER,
        *probe_args: Any,
    ):
        probe_id = pad_getter_fcn(self.pipeline, element_name).add_probe(
            kind, callback, *probe_args
        )
        self._probes[element_name] = probe_id

    def __call__(
        self,
        on_bus_message: Optional[Callable] = None,
        loop: Optional[Loop] = None,
    ):
        loop = loop or GLib.MainLoop()
        if on_bus_message:
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", bus_call, loop)
        logger.info("Starting pipeline \n")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            loop.run()
        finally:
            self.pipeline.set_state(Gst.State.NULL)


def get_batch_meta(info: Gst.PadProbeInfo) -> pyds.NvDsBatchMeta:
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logger.warning("Unable to get GstBuffer")
        return
    return pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
