#!/bin/bash

source vars.sh

$CONTAINER_CMD run --gpus all -it --rm \
  --env HOME=$WORKDIR \
  --env SHELL="/bin/bash" \
  --publish $PORT:$PORT \
  --workdir $WORKDIR \
  --volume $WORKDIR:$WORKDIR \
  --ipc=host \
  $IMAGE python -m jupyterlab --allow-root --ServerApp.port $PORT --no-browser \
    --ServerApp.ip='0.0.0.0' --certfile=$CERTFILE --keyfile=$KEYFILE
