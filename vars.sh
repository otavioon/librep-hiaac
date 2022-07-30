CONTAINER_CMD=docker
WORKDIR=$(realpath $(pwd))
IMAGE=hiaac-cuda-11.5
PORT=10008
CERT_DIR=.ssl_cert
CERTFILE=$CERT_DIR/hiaac.crt
KEYFILE=$CERT_DIR/hiaac.key
