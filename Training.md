# Self-driving-car for Udacity simulator
Training a deep-neural network to drive

## Overview

We're going to use Udacity's [self driving car simulator](https://github.com/udacity/self-driving-car-sim) as a testbed for training an autonomous car. 

## Dependencies

If you have [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) installed on your machine, you can create and install all dependencies on a dedicated virtual environment, by running one of the following commands



```python
# Use TensorFlow without GPU
conda env create -f environments.yml 

# Use TensorFlow with GPU
conda env create -f environment-gpu.yml
```

Alternatively you can manually install the required libraries (see the contents of the environemnt*.yml files) using ```pip```.


## Usage


### Run the pretrained model

Start up [the Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim), choose a scene, and press the Autonomous Mode button. Then, run the model as follows:

```python
python drive.py models/model.h5
```

### To train the model

You'll need the data folder which contains the training images.

```python
python model.py
```

This will generate a file `model-<epoch>.h5` whenever the performance in the epoch is better than the previous best.  For example, the first epoch will generate a file called `model-000.h5`.

## Credits

The credits for this code go to his original creator [naokishibuya](https://github.com/naokishibuya), [llSourcell](https://github.com/llSourcell/How_to_simulate_a_self_driving_car/commits?author=llSourcell) who created a nice wrapper to get people involved, and Siraj Raval who explained its functioning on a nice [video](https://youtu.be/EaY5QiZwSP4) on Youtube.