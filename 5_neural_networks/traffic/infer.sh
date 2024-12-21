#!/bin/bash

selected_model=$(find . -type f -name "*.keras" | sort | \
    fzf --prompt="Select model: " \
        --preview-window=hidden)

if [ ! -z "$selected_model" ]; then
    selected_image=$(find ./gtsrb -type f -name "*.ppm" | sort | \
        fzf --preview 'viu {}' \
            --preview-window=right:50% \
            --prompt="Select image: ")

    if [ ! -z "$selected_image" ]; then
        python ./infer.py "$selected_model" "$selected_image"
    fi
fi
