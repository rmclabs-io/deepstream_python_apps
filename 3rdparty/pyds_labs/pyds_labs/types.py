from typing import Any, Protocol

from pyds_labs import Gst


class Gst_PadProbeCallback(Protocol):
    def __call__(
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        *args: Any,
    ) -> Gst.PadProbeReturn:
        ...


class Loop:
    def run(self):
        ...

    def quit(self):
        ...
