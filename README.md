# DQN-startup
This project provides a mini DQN startup environment.

For your information, you can <i>copy and paste</i> this line (when your directory is <i>~/DQN-startup</i>) into your CLI terminal to download this project into your machine.

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

You may reopen your terminal. You will see (base) in your terminal line. You can turn off the auto conda startup on terminal by typing:

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

(optional FYI) You can deactivate (unnecessarily) your conda environment by:

```shell
conda deactivate
```

You can check your conda environment by:
```python
import torch
```
like this:

![conda-activate-reason](https://user-images.githubusercontent.com/46191084/178488583-8b5569cf-2f8f-470f-b7c2-d582b96cf4ef.png)


### DQN environment setup
You need to install a <b>GPU-supported pytorch</b> by running below. You would also need the <b>scikit-learn</b> package for the data analysis. This would take a couple of minutes.

```shell
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install scikit-learn
```

#### NOTE
Please kindly note that cudatoolkit version may be different.

## Code Usage
You may run <i>main.py</i> or <i>run.ipynb</i> to see what's going on.

Additional changes may take place.
