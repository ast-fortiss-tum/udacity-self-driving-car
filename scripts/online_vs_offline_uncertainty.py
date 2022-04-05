# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
from pathlib import Path

import tensorflow
from scipy.stats import stats

import utils
from config import Config
from utils import *

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    print("Script to compare offline vs online (within Udacity's) uncertainty values")

    # load the online uncertainty from csv
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)
    online_uncertainty = data_df["uncertainty"]
    print("loaded %d uncertainty values" % len(online_uncertainty))

    # compute the steering angle from the images stored on the fs
    data = data_df["center"]
    print("read %d images from file" % len(data))

    dave2 = tensorflow.keras.models.load_model(Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME)))

    offline_uncertainty = []
    all_errors = []
    image = None
    total_time = 0
    max = -1
    min = 100
    min_idx = -1
    max_idx = -1

    batch_size = 128

    for i, x in enumerate(data):
        image = mpimg.imread(x)
        image = utils.preprocess(image)  # apply the pre-processing
        image = np.array([image])  # the model expects 4D array

        start = time.time()

        x = np.array([image for idx in range(batch_size)])

        # save predictions from a sample pass
        outputs = dave2.predict_on_batch(x)

        # average over all passes if the final steering angle
        steering_angle = outputs.mean(axis=0)
        # evaluate against labels
        uncertainty = outputs.var(axis=0)

        duration = round(time.time() - start, 4)
        # print("Prediction completed in %s." % str(duration))

        total_time += duration

        error = abs(online_uncertainty[i] - uncertainty)

        if error > max:
            max = error
            max_idx = i

        if error < min:
            min = error
            min_idx = i

        all_errors.append(error)
        offline_uncertainty.append(uncertainty)

    print("All predictions completed in %s (%s/s)." % (
        str(round(total_time, 0)), str(round(total_time / len(online_uncertainty), 2))))
    print("Mean error %s." % (str(np.sum(all_errors) / len(all_errors))))

    # compute and plot the rec errors
    x_losses = np.arange(len(online_uncertainty))
    plt.plot(x_losses, online_uncertainty, color='blue', alpha=0.2, label='online uncertainty')
    plt.plot(x_losses, offline_uncertainty, color='red', alpha=0.2, label='offline uncertainty')

    plt.ylabel('uncertainty')
    plt.xlabel('Frames')
    plt.title("offline vs online (within Udacity's) uncertainty values (batch_size=" + str(batch_size) + ")")
    plt.legend()
    plt.savefig("plots/online-vs-offline-uncertainty.png")
    plt.show()

    print(stats.mannwhitneyu(online_uncertainty, pd.Series(offline_uncertainty)))

    plt.figure(figsize=(80, 16))
    # display original
    ax = plt.subplot(1, 2, 1)
    plt.imshow(mpimg.imread(data[min_idx]).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("min diff (offline=%s, online=%s)" % (round(offline_uncertainty[min_idx][0], 8),
                                                    round(online_uncertainty[min_idx], 8)),
              fontsize=50)

    # display reconstruction
    ax = plt.subplot(1, 2, 2)
    plt.imshow(mpimg.imread(data[max_idx]).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("max diff (offline=%s, online=%s)" % (round(offline_uncertainty[max_idx][0], 8),
                                                    round(online_uncertainty[max_idx], 8)),
              fontsize=50)

    plt.savefig("plots/uncertainty-diff.png")
    plt.show()
    plt.close()
