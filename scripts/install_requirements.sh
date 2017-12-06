#!/bin/bash

pip install -r requirements.txt
# NLTK and Spacy models (copied from AllenNLP)
python -m nltk.downloader -u https://pastebin.com/raw/D3TBY4Mj punkt
python -m spacy.en.download all
