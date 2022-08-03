ARG BASE_IMG=nvcr.io/nvidia/deepstream-l4t:6.1-samples

FROM ${BASE_IMG}
ARG DEBIAN_FRONTEND=noninteractive


ADD reqs/requirements.apt.list /tmp/requirements.apt.list

WORKDIR /opt/rmclabs/deepstream_python_apps

COPY . .
RUN scripts/install


# RUN gst-inspect-1.0 nvinfer

# RUN pip install --upgrade pip setuptools
# ADD requirements.txt /tmp/requirements.txt
# RUN pip install -r /tmp/requirements.txt

# ENV DS=/opt/nvidia/deepstream/deepstream
# ENV DS_STREAMS=${DS}/samples/streams

# WORKDIR /opt/rmclabs/dsmask

# # TODO: move these to multistage
# # TODO: generate engine
# RUN wget \
#   --content-disposition \
#   https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesegnet/versions/deployable_v2.0.2/zip \
#   -O peoplesegnet_deployable_v2.0.2.zip \
#   && unzip peoplesegnet_deployable_v2.0.2.zip \
#   && mkdir -p models/peoplesegnet \
#   && mv model.step* models/peoplesegnet \
#   && rm -rf model.step* peoplesegnet_deployable_v2.0.2.zip
# # ENDTODO: move these to multistage

# ARG precompiled_jetson_8_4_0_11=https://nvidia.box.com/shared/static/gcp6ylk1ku0zfobhj0sv8vpraz6yzaf9
# RUN wget ${precompiled_jetson_8_4_0_11} -O /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.4.0 \
#   && ldconfig

# ADD patch /patch

# ENV DPA=/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps
# RUN pip uninstall -y pyds \
#   && cd ${DPA}/bindings \
#   && mv /patch/bindnvdsmeta.cpp ${DPA}/bindings/src/bindnvdsmeta.cpp \
#   && mv /patch/bindnvosd.cpp ${DPA}/bindings/src/bindnvosd.cpp \
#   && mv /patch/pydocumentation.h ${DPA}/bindings/docstrings/pydocumentation.h \
#   && rm -rf build || true ; mkdir build \
#   && cd build \ 
#   && cmake \
#     .. \
#     -DPYTHON_MAJOR_VERSION=3 \
#     -DPYTHON_MINOR_VERSION=8 \
#     -DPIP_PLATFORM=linux_aarch64 \
#     -DDS_PATH=/opt/nvidia/deepstream/deepstream/ \
#   && make \
#   && pip install ${DPA}/bindings/build/pyds-1.1.3-py3-none-linux_aarch64.whl
# ADD entrypoint.py /entrypoint.py

# ENTRYPOINT ["/entrypoint.py"]
