FROM continuumio/miniconda3

WORKDIR /app

# Install OpenGL libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
RUN echo "conda activate spk2extract" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# The so that conda activate works:
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
