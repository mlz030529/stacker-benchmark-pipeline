#!/bin/sh

# DOCKER CONFIGURATION SCRIPT
#
# A shell script to run in your Docker container. This will set up
# the container's environment and install all dependencies, ensuring
# your python distribution is ready to use.
#
# If you follow the procedure in setup/environment-setup-docker.md,
# this shell script will be executed automatically and should not
# need to be called manually. Feel free to call this script at your
# own discretion.

apt clean
apt update
apt install -y openexr=2.3.0-6ubuntu0.5 ffmpeg=7:4.2.7-0ubuntu0.1 libsm6=2:1.2.3-1 libxext6=2:1.3.4-0ubuntu1
pip install pip==23.3.1
pip install -r docker-requirements.txt