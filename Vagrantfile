Vagrant.configure("2") do |config|
  config.vm.hostname = "ubuntu"

  config.vm.provider :docker do |docker, override|
    override.vm.box = nil
    docker.image = "rofrano/vagrant-provider:ubuntu"
    docker.remains_running = true
    docker.has_ssh = true
    docker.privileged = true
    docker.volumes = ["/sys/fs/cgroup:/sys/fs/cgroup:rw"]
    docker.create_args = ["--cgroupns=host"]
  end

  # Provision Miniconda for ARM architecture
  config.vm.provision "shell", inline: <<-SHELL
    if [ ! -d "$HOME/miniconda3" ]; then
        echo "Downloading Miniconda for ARM64..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh
        echo "Installing Miniconda..."
        bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    fi
    echo "Initializing Miniconda..."
    $HOME/miniconda3/bin/conda init
    echo "Installing Miniconda packages..."
    $HOME/miniconda3/bin/conda env create -f $HOME/repos/cpl_pipeline/environment.yml
    echo "Installing Neovim and other apt installs"
    sudo apt-get update
    sudo apt-get upgrade -y
    sudo apt install zsh -y
    ch sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
    sudo apt-get install -y apt-utils neovim git curl wget zsh tmux htop tree python3-pip python3-dev python3-venv python3-setuptools python3-wheel build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev

    mkdir -p $HOME/repos
    cd $HOME/repos
    git clone https://github.com/FlynnOConnell/cpl_pipeline.git
    cd cpl_pipeline
    $HOME/miniconda3/


    git clone https://github.com/FlynnOConnell/.dotfiles.git
    ln -s $HOME/.config $HOME/repos/.dotfiles/dots/.config
    ln -s $HOME/.zshrc $HOME/repos/.dotfiles/dots/.zshrc
    ln -s $HOME/tmux/.tmux.conf $HOME/repos/.dotfiles/dots/tmux/.tmux.conf
    ln -s $HOME/.local/bin $HOME/repos/.dotfiles/dots/.local/bin

    echo "Setting up SSH..."
    if [ ! -f "$HOME/.ssh/id_rsa" ]; then
        ssh-keygen -t rsa -b 4096 -f $HOME/.ssh/id_rsa -q -N ""
        echo "Don't forget to add the public SSH key to your GitHub account."
        cat $HOME/.ssh/id_rsa.pub
    fi
    SHELL
  end


