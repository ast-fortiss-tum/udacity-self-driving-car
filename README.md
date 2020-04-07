# Self-driving-car for Udacity simulator
Project that simulates a self-driving car for the Udacity simulator

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

Alternatively you can manually install the required libraries using ```pip```.

```python
virtualenv udacity-self-driving-car

source udacity-self-driving-car/bin/activate 

pip install -r requirements.txt
```

## Usage

### Run a pretrained model in the original Udacity simulator

Start up [the original Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim), choose a scene, and press the Autonomous Mode button. Then, run the prea pretrained as follows:

```python
python drive-original-simulator.py  # runs the model within models/model.h5
```

### Run a pretrained model in the improved Udacity simulator

Alternatively, you may want to use the improved Udacity self-driving simulator developed in the work "Misbehaviour Prediction for Autonomous Driving Systems" by Stocco et al. and published at ICSE 2020.

First, download [the improved Udacity self-driving simulator](https://drive.google.com/open?id=1gS_dGgpasywJZzhy5eUQoqNaYZH9X-5V). Second, download [this autoencoder](https://drive.google.com/open?id=1m5teCThr_VcG0EPcCO-LUiecxzw9GTj1), and place it in the ```sao``` folder. Then choose a scene, and press the Autonomous Mode button. Then, run the model as follows:

```python
python drive.py  # improved Udacity simulator with CTE and effects
```

If you want to record the data of the simulation in a CSV file run as follows:
```python
python drive.py -t="folder-name"  # records the simulation in the specified directory under the simulations folder
```

### To train the model

You'll need the data folder which contains the training images.

```python
python model.py
```

This will generate a file `model-<epoch>.h5` whenever the performance in the epoch is better than the previous best.  For example, the first epoch will generate a file called `model-000.h5`.

## Credits

The credits for this code go to his original creator [naokishibuya](https://github.com/naokishibuya), [llSourcell](https://github.com/llSourcell/How_to_simulate_a_self_driving_car/commits?author=llSourcell) who created a nice wrapper to get people involved.
