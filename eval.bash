#!/bin/bash

DATA_PATH="$1"
rm -rf "${DATA_PATH}/renders"
ns-eval --load-config "${DATA_PATH}/config.yml" \
        --render-output-path "${DATA_PATH}/renders" \
        --output-path "${DATA_PATH}/results.json"
