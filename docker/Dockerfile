FROM  763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.3.1-gpu-py37-cu110-ubuntu18.04

RUN pip install jupyterlab

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y python3-opencv libopenblas-base \
        libomp-dev build-essential

RUN curl -sL https://deb.nodesource.com/setup_15.x | bash
RUN apt-get install -y nodejs
#ENV PATH="/usr/local/nvm/versions/node/v15.0.1/bin:${PATH}"

RUN pip install --upgrade tensorflow_addons

RUN pip install ipywidgets imgaug \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN pip install jupyterlab-nvdashboard tqdm tensorflow_datasets \
    && jupyter labextension install jupyterlab-nvdashboard

RUN pip install matplotlib scikit-learn scikit-image seaborn \
        cython numba tqdm tensorflow_datasets

RUN pip install pybind11
RUN git clone https://github.com/johnbensnyder/cocoapi && \
	cd cocoapi/PythonAPI && \
	pip install -v --no-cache-dir -e .
RUN pip install mpi4py
RUN pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip
RUN pip install opencv-python==3.4.11.45

RUN apt-get update
RUN apt-get install -y python3-opencv

# Dali 0.27.0-0.29.0 has seg fault in coco loader, install from weekly until 0.30.0
# RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
# RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-tf-plugin-cuda110
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly nvidia-dali-weekly-cuda110 nvidia-dali-tf-plugin-weekly-cuda110
RUN pip install nvgpu

RUN apt-get install -y zsh
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
RUN cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
RUN apt-get install bc
RUN pip install git+https://github.com/NVIDIA/dllogger.git
RUN mkdir -p /workspace
WORKDIR /workspace

ENV SHELL="/bin/zsh"
CMD SHELL="/bin/zsh" nohup jupyter lab --allow-root --ip=0.0.0.0 --no-browser > notebook.log

# docker build -t johnbensnyder/tensorflow:mrcnn .
# docker run --rm -it -d --gpus all --name aws_tensorflow --net=host --uts=host --ipc=host --ulimit stack=67108864 --ulimit memlock=-1 --security-opt seccomp=unconfined -v ~/data/:/data/ -v ~/DeepLearningExamples/:/DeepLearningExamples/ johnbensnyder/tensorflow:mrcnn 
