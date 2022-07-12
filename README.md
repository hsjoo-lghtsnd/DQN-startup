# DQN-startup
This project provides a mini DQN startup environment.

For your information, you can <i>copy and paste</i> this line (when your directory is <i>~/DQN-startup</i>) into your CLI terminal.

```shell
git clone https://github.com/hsjoo-lghtsnd/DQN-startup ~/DQN-startup
```

## Requirements
### miniconda
This project assumes <i>miniconda</i> for the python <i>(pytorch)</i> environment maintainer. You can use the following executive lines for your <i>linux</i> machine to install miniconda.

```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

You may reopen your terminal. You will see (base) in your terminal line. You can turn off the auto conda startup by typing:

```shell
conda config --set auto_activate_base false
```

Let's create a conda environment by using:
```shell
conda create --name DQN
```

Now, you can activate your (DQN) environment by typing simply:
```shell
conda activate DQN
```
You will see a similar screen like this:

![conda-activate](https://user-images.githubusercontent.com/46191084/178430890-7f6caeba-50be-40a7-9bd3-557cdb089ae8.png)

You can deactivate (unnecessarily) your conda environment by:

```shell
conda deactivate
```

### DQN environment setup

