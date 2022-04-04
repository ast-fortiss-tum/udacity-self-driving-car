# Copyright 2021 Testing Automated @ Università della Svizzera italiana (USI)
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
from config import Config
from utils import *

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    interval = np.arange(10, 101, step=10)

    plt.figure(figsize=(30, 8))

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)

    # read steering angle values
    steering_angles = data_df["steering_angle"]

    # read uncertainty values
    uncertainty = data_df["uncertainty"]

    # read CTE values
    crashes = data_df[data_df["crashed"] == 1]
    is_crash = (crashes.crashed - 1) + cfg.CTE_TOLERANCE_LEVEL
    is_crash_2 = (crashes.crashed - 1) - cfg.CTE_TOLERANCE_LEVEL

    x_losses = np.arange(len(steering_angles))

    # changes the frequency of the ticks on the X-axis to simulation's seconds
    plt.xticks(
        np.arange(0, len(steering_angles) + 1, cfg.FPS),
        labels=range(0, len(steering_angles) // cfg.FPS + 1))

    # visualize crashes
    crashes = data_df[data_df["crashed"] == 1]
    crashes = crashes
    is_crash = (crashes.crashed - 1) + np.mean(steering_angles)
    plt.plot(is_crash, 'x:r', markersize=4)

    # visualize steering angle
    plt.plot(x_losses, steering_angles, '--', color='black', label="steering angle")

    # visualize uncertainty as continuous error bands
    plt.fill_between(x_losses, (steering_angles - uncertainty), (steering_angles + uncertainty),
                     color='red',
                     alpha=0.3)

    plt.legend()
    plt.ylabel('Steering Angle')
    plt.xlabel('Frames')
    plt.title("Steering angles values for "
              + cfg.SIMULATION_NAME,
              fontsize=20)

    plt.savefig('plots/steering-angle.png')

    plt.show()
