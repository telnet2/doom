#!/bin/bash

# https://github.com/openai/whisper
pip install -U openai-whisper
pip install --upgrade transformers 'datasets[audio]' accelerate

brew install ffmpeg
