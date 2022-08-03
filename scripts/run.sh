#!/usr/bin/env bash

set -euxo pipefail

cd "$(dirname "$0")"/..
echo $PWD

tag=rmclabs.io/deepstream-python-apps_aarch64_py38:latest
docker build \
  -f scripts/jetson.Dockerfile \
  -t $tag \
  .

docker run \
  --rm \
  -it \
  --entrypoint bash \
  $tag
