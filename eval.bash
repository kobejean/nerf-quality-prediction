#!/bin/bash

DATA_PATH="$1"

ns-eval --load-config "${DATA_PATH}/config.yml" \
        --render-output-path "${DATA_PATH}/renders" \
        --output-path "${DATA_PATH}/results.json"
