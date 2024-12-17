FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
  
ARG DEBIAN_FRONTEND=noninteractive  
ARG TARGETARCH  
  
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 vim ffmpeg zip unzip htop screen tree build-essential gcc g++ make libfreeimage-dev && apt-get clean && rm -rf /var/lib/apt/lists  
  
RUN pip install --upgrade pip  
  
# 將 requirements.txt 複製到 Docker 映像中  
COPY requirements.txt . 
COPY deb/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb /tmp/  
COPY whl/whisperx-3.1.1-py3-none-any.whl .

# 安裝 python packages  
RUN pip3 install -r requirements.txt  

# 安裝 cudnn 9.1
RUN dpkg -i /tmp/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb  
RUN cp /var/cudnn-local-repo-ubuntu2204-9.1.0/cudnn-local-52C3CBCA-keyring.gpg /usr/share/keyrings/  
RUN apt-get update  
RUN apt-get -y install cudnn-cuda-11

# 設置工作目錄  
WORKDIR /app  
  
# 复制 app 资料夹到 Docker 映像中的 /app 目录  
COPY . /app  
  
# 设置环境变量  
ENV LC_ALL=C.UTF-8  
ENV LANG=C.UTF-8  
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility  
ENV NVIDIA_VISIBLE_DEVICES=all  
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64  
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH  
