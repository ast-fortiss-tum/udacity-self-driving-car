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

    cfg.SIMULATION_NAME = 'gauss-journal-track1-nominal'

    print("Script to compare offline vs online (within Udacity's) uncertainty values")

    # load the online uncertainty from csv
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)

    data_df = data_df[data_df["crashed"] == 0]

    online_uncertainty = data_df["cte"]
    online_steering_angles = data_df["steering_angle"]
    print("loaded %d steering_angle values" % len(online_steering_angles))
    print("loaded %d CTE values" % len(online_uncertainty))

    min_idx_unc = np.argmin(online_uncertainty)
    max_idx_unc = np.argmax(online_uncertainty)

    # compute the steering angle from the images stored on the fs
    center_images = data_df["center"]
    print("read %d images from file" % len(center_images))

    plt.figure(figsize=(80, 16))
    # display original
    ax = plt.subplot(1, 2, 1)
    plt.imshow(mpimg.imread(center_images[min_idx_unc]).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("min uncertainty (steering angle=%s, CTE=%s)" % (round(online_steering_angles[min_idx_unc], 5),
                                                               round(online_uncertainty[min_idx_unc], 2)),
              fontsize=50)

    # display reconstruction
    ax = plt.subplot(1, 2, 2)
    plt.imshow(mpimg.imread(center_images[max_idx_unc]).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("max uncertainty (steering angle=%s, CTE=%s)" % (round(online_steering_angles[max_idx_unc], 5),
                                                               round(online_uncertainty[max_idx_unc], 2)),
              fontsize=50)

    plt.savefig("plots/steering-cte.png")
    plt.show()
    plt.close()
