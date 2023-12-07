FROM nvcr.io/nvidia/pytorch:23.01-py3
RUN pip install tensorboardX==2.5
RUN pip install pyyaml==6.0
RUN pip install yacs==0.1.8
RUN pip install termcolor==1.1.0
RUN pip install opencv-python==4.4.0.46
RUN pip install timm==0.6.12
RUN pip install lmdb
WORKDIR /app
COPY . /app
