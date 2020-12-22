from pathlib import Path

import tensorflow
from scipy.stats import stats

import utils
from config import Config
from utils import *

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    print("Script to compare offline vs online (within Udacity's) steering_angle values")

    # load the online uncertainty from csv
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        'gauss-journal-track1-dave2-nominal-latent2',
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
        duration = round(time.time() - start, 4)
        print("Prediction completed in %s." % str(duration))
        total_time += duration

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

    plt.ylabel('steering angles')
    plt.xlabel('Frames')
    plt.title("offline vs online (within Udacity's) steering_angle values")
    plt.legend()
    plt.show()

    print(stats.mannwhitneyu(online_steering_angles, offline_steering_angles))

    image = mpimg.imread(x)

    plt.figure(figsize=(80, 16))
    # display original
    ax = plt.subplot(1, 2, 1)
    plt.imshow(mpimg.imread(data[min_idx]).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("min diff (steering angle=%s, predicted=%s" % (round(offline_steering_angles[min_idx], 4),
                                                             round(online_steering_angles[min_idx], 4)),
              fontsize=60)

    # display reconstruction
    ax = plt.subplot(1, 2, 2)
    plt.imshow(mpimg.imread(data[max_idx]).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("max diff (steering angle=%s, predicted=%s" % (round(offline_steering_angles[max_idx], 4),
                                                             round(online_steering_angles[max_idx], 4)),
              fontsize=60)

    plt.savefig("plots/steering-angle-diff.png")
    plt.show()
    plt.close()