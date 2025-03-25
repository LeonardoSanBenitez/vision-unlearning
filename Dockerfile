FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install --no-install-recommends -y -q \
        software-properties-common \
        apt-utils \
        build-essential \
        pkg-config \
        gnupg2 \
        ca-certificates \
        curl \
        git \
        wget \
        libx11-6 \
        libxext6 \
        ffmpeg \
        libsm6 

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10 python3.10-dev python3.10-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip
    
RUN pip3 install poetry~=1.2 && poetry config virtualenvs.create false
RUN git config --global --add safe.directory /src

########################################
# Application dependencies
WORKDIR /src/
COPY pyproject.toml /src/
RUN poetry install --with dev --no-root --no-interaction --no-ansi

########################################
# Add Jupyter notebook
RUN poetry add jupyter pickleshare
ADD https://github.com/krallin/tini/releases/download/v0.6.0/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]
