FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install OpenGL libraries and other necessary tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    wget \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV MINICONDA_VERSION 4.10.3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_${MINICONDA_VERSION}-Linux-x86_64.sh -O /miniconda.sh && \
    /bin/bash /miniconda.sh -b -p /miniconda && \
    rm /miniconda.sh
ENV PATH=/miniconda/bin:$PATH

# Copy environment file
COPY environment.yml .

# Create the conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
RUN echo "conda activate cpl_pipeline" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# The entrypoint script so that conda activate works
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
