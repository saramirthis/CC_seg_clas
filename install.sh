#! /bin/bash

echo "Creating virtual environment with requirements"
python3 -m venv virtualenv
source virtualenv/bin/activate
pip3 install -r requirements.txt
deactivate
