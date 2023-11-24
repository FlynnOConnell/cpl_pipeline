FROM continuumio/miniconda3

# Create a non-root user, standard practice for security reasons
ARG username=spk_user
ARG uid=1000
ARG gid=100

ENV HOME /home/$username

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $uid \
    --gid $gid \
    --home $HOME \
    $username

COPY ../environment.yml /tmp/
RUN chown $uid:$gid /tmp/environment.yml

# Create a project directory inside user home
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# Copy project files
COPY . $PROJECT_DIR

# Build the conda environment
ENV ENV_PREFIX $PWD/env
RUN conda update --name base --channel defaults conda && \
    conda env create --prefix $ENV_PREFIX --file /tmp/environment.yml --force && \
    conda clean --all --yes

# Set up a script to activate the environment when the container starts
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate $ENV_PREFIX" >> ~/.bashrc

USER $username
CMD [ "/bin/bash" ]

