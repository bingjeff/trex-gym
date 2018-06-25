# trex-gym
OpenAI Gym environment using pybullet for a Tyrannosaur.

## Installation
This work depends on the installation of other resources.

The biggest three dependencies are:

* [pyBullet](https://github.com/bulletphysics/bullet3) - Used for physics and rendering.
* [OpenAI Gym](https://github.com/openai/gym) - Provides the basis for the model "environment".
* [OpenAI Baselines](https://github.com/openai/baselines) - Used for the RL agents.
* [Tensorflow](https://github.com/tensorflow/tensorflow) - Used as a dependency for the ML infrastructure.

The above packages should be installed and working. It may be possible to skim by on a PIP install, but as of this
writing at least the OpenAI Baselines required some additional love to get working.

Furthermore, while the environments will work with most flavors of Python, the agents do not. Therefore it is suggested
that you use Python >=3.5, and to install the various components into a virtualenv.

### Fast install

It is possible that the following may just work:

`pip install absl-py baselines gym numpy pybullet tensorflow`

### Slow install

More than likely you will need the more recent packages than exist from the pip standard repository. These are the
steps that I use to setup my virtualenv.

Create a directory and clone the repos that need to be built from source:

```
mkdir trex_training
cd trex_training
git clone https://github.com/bingjeff/trex-gym.git
git clone https://github.com/openai/baselines.git
```

Install prerequisites for building the various libraries from scratch. For example:

`sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev`

Create the virtualenv that will be used, and activate it:

```
virtualenv venv_trex_training --python=python3
source venv_trex_training/bin/activate
```

Install the packages that don't need to build from source:

`pip install absl-py gym numpy pybullet tensorflow`

Start installing the packages that need to be built from source.

`pip install -e baselines/`

