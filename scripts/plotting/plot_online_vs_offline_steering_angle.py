# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
from pathlib import Path

from scipy.stats import stats

import utils
from utils import *

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts/plotting', ''))

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    print("Script to compare offline vs online (within Udacity's) steering_angle values")

    # load the online uncertainty from csv
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)
    online_steering_angles = data_df["steering_angle"]
    print("loaded %d steering_angle values" % len(online_steering_angles))

    # compute the steering angle from the images stored on the fs
    data = data_df["center"]
    print("read %d images from file" % len(data))

    dave2 = tensorflow.keras.models.load_model(Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME)))

    offline_steering_angles = []
    all_errors = []
    image = None
    total_time = 0
    max = -1
    min = 100
    min_idx = -1
    max_idx = -1

    for i, x in enumerate(data):
        image = mpimg.imread(x)
        image = utils.preprocess(image)  # apply the pre-processing
        image = np.array([image])  # the model expects 4D array

        start = time.time()
        y = float(dave2.predict(image, batch_size=1))

        total_time += round(time.time() - start, 4)

        error = abs(online_steering_angles[i] - y)

        if error > max:
            max = error
            max_idx = i

        if error < min:
            min = error
            min_idx = i

        all_errors.append(error)
        offline_steering_angles.append(y)

    np.save("offline_steering_angles", offline_steering_angles)
    np.save("all_errors", all_errors)

    print("All predictions completed in %s (%s/s)." % (
        str(total_time), str(total_time / len(online_steering_angles))))
    print("Mean error %s." % (str(np.sum(all_errors) / len(all_errors))))

    # compute and plot the rec errors
    x_losses = np.arange(len(online_steering_angles))
    plt.plot(x_losses, online_steering_angles, color='blue', alpha=0.2, label='online steering angles')
    plt.plot(x_losses, offline_steering_angles, color='red', alpha=0.2, label='offline steering angles')
    # plt.plot(x_losses, all_errors, color='red', label='errors')

    plt.ylabel('steering angles')
    plt.xlabel('Frames')
    plt.title("offline vs online (within Udacity's) steering_angle values")
    plt.legend()
    plt.savefig("plots/online-vs-offline-steering-angle.png")
    plt.show()

    print(stats.mannwhitneyu(online_steering_angles, offline_steering_angles))

    image = mpimg.imread(x)

    plt.figure(figsize=(80, 16))
    # display original
    ax = plt.subplot(1, 2, 1)
    plt.imshow(mpimg.imread(data[min_idx]).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("min diff (offline=%s, online=%s)" % (round(offline_steering_angles[min_idx], 4),
                                                    round(online_steering_angles[min_idx], 4)),
              fontsize=60)

    # display reconstruction
    ax = plt.subplot(1, 2, 2)
    plt.imshow(mpimg.imread(data[max_idx]).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("max diff (offline=%s, online=%s)" % (round(offline_steering_angles[max_idx], 4),
                                                    round(online_steering_angles[max_idx], 4)),
              fontsize=60)

    plt.savefig("plots/steering-angle-diff.png")
    plt.show()
    plt.close()
