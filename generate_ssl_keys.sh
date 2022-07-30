#!/bin/bash

source vars.sh
mkdir -p $CERT_DIR

openssl req -newkey rsa:4096 \
            -x509 \
            -sha256 \
            -days 3650 \
            -nodes \
            -out $CERTFILE \
            -keyout $KEYFILE \
            -subj "/C=BR/ST=Sao Paulo/L=Campinas/O=UNICAMP/OU=H.IACC/CN=hiaac.unicamp.br"
