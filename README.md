# Programmatic Reinforcement Learning Without Oracles

This code is gratefully adapted from the authors of Programmatic Reinforcement Learning without Oracles, in ICLR 2022. Credit to Wenjie Qiu and He Zhu.

## Installation

We tested our code on Ubuntu 18.04 LTS x86_64 platform. Please make sure to read below instructions before using requirements.txt.

- Required Dependencies

    ```
    $ sudo apt install libopenmpi-dev libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev patchelf
    ```

- Install MuJoCo

    Download zip:

    ```
    $ wget https://roboti.us/download/mujoco200_linux.zip
    ```

    Create mujoco directory on your home folder:

    ```
    $ mkdir ~/.mujoco
    ```

    Unzip and move:

    ```
    $ unzip ./mujoco200_linux.zip 
    $ mv ./mucjoco200_linux ~/.mujoco/

    $ cd ~/.mujoco 
    $ mv ./mujoco200_linux mujoco200
    ```

    Download the key:
    ```
    $ wget https://roboti.us/file/mjkey.txt
    ```

    Check if the MuJoCo library and key are ready:
    ```
    $ ls ~/.mujoco
    mjkey.txt mujoco200
    ```

    Export to library path:

    ```
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
    ```

    Upgrade pip (required):

    ```
    python3 -m pip --upgrade
    ```

    **Notes: at this point, you may use requirements.txt to install the reamaining python packages.**

    Install mujoco-py:

    ```
    pip3 install mujoco-py==2.0.2.13
    ```

    Now you should be able to import `mujoco_py`.

- Install Python Packages

    Tested on Python 3.6.9 (default version on Ubuntu 18.04).

    Install python packages:

    ```
    pip3 install numpy torch matplotlib z3-solver pillow tqdm tabulate joblib gym pathos==0.2.8 mpi4py==3.1.1
    ```


## Architecture Search for Programmatic Policies (π-PRL)

Solving **Ant Cross Maze** (0), **Ant Random Goal** (1), **HalfCheetah Hurdle** (2) and **Pusher** (3) environments

```
python3 pi_PRL.py -e [environment number] -s [random seed] -d [directory to save]
```


## Programmatic High-level Planning (π-HPRL)

Solving **Ant Maze** (0), **Ant Push** (1) and **Ant Fall** (2) environments.

```
python3 pi_HPRL.py -e [environment number] -s [random seed] -d [directory to save]
```
