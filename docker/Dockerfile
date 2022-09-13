FROM ubuntu:xenial

RUN apt-get update -y && \
    apt-get install -y python3.5 \
    python-distutils-extra \
    python3-pip python3-apt
RUN apt-get install python3-numpy \
    python3-scipy \
    python3-matplotlib -y
RUN pip3 install tabulate \
    funcsigs \
    nibabel==2.2.1 \
    pandas==0.13.0 \
    scikit-learn==0.19.0 \
    joblib==0.11

RUN useradd -ms /bin/bash cc_seg
USER cc_seg
WORKDIR /home/cc_seg
