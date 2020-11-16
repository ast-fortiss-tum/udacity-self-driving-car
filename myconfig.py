# project settings
TRAINING_DATA_DIR = "datasets"  # root folder for all driving training sets
SIMULATION_DATA_DIR = "dataset5"  # the driving training set to use
TEST_SIZE = 0.05  # split of training data used for the validation set (keep it low)

TRACK = ["track1"]  # ["track1"|"track2"|"track3"|"track1","track2","track3"] the race track to use
TRACK1_DRIVING_STYLES = ["normal", "recovery", "reverse"]
# TRACK1_DRIVING_STYLES = ["normal"]
TRACK2_DRIVING_STYLES = ["normal", "recovery", "recovery2", "recovery3", "reverse", "sport_normal", "sport_reverse"]
TRACK3_DRIVING_STYLES = ["normal", "recovery", "recovery2", "reverse", "sport_normal"]

# self-driving car model settings
NUM_EPOCHS_SDC_MODEL = 50  # training epochs for the self-driving car model
SAMPLES_PER_EPOCH = 100  # number of samples to process before going to the next epoch
BATCH_SIZE = 32  # number of samples per gradient update
SAVE_BEST_ONLY = True  # only saves when the model is considered the "best" according to the quantity monitored
LEARNING_RATE = 1.0e-4  # amount that the weights are updated during training

# autoencoder-based self-assessement oracle settings
NUM_EPOCHS_SAO_MODEL = 5  # training epochs for the autoencoder-based self-assessment oracle
USE_ONLY_CENTER_IMG = False  # train the autoencoder-based self-assessment oracle only using front-facing camera images
USE_CROP = True  # crop the images the same way as the car
LOSS_SAO_MODEL = "MSE"  # "VAE"|"MSE" objective function for the autoencoder-based self-assessment oracle
