import argparse
import csv
import glob
import os

import numpy as np
from scipy.stats import entropy

import utils


def get_metric_speed(metric_array, frame_time):
    """
    get metric values over a time window
    :param metric_array: the metric
    :param frame_time: the time window expressed in number of frames
    :return: metric value over the specified time window
    """
    metric_array_copy = metric_array[1:len(metric_array)]
    metric_array_copy = np.append(metric_array_copy, 0)
    metric_speed_array = metric_array_copy - metric_array
    metric_speed_array = metric_speed_array[0:len(metric_speed_array) - 1] / frame_time
    return metric_speed_array


def get_metrics_from_csv_files(args):
    files = glob.glob(args.data_dir + "/*.csv")  # all CSV files
    listoflists = []
    list = ['metric',
            'Distance', 'Time', 'Crashes', 'OBE',
            'Mean(SA)', 'SD(SA)', 'Var(SA)', 'Entropy(SA)',
            'Mean(AD)',
            'Min(LP)', 'Max(LP)', 'Mean(LP)', 'SD(LP)', 'Var(LP)',
            'Mean(LS)', 'SD(LS)',
            'Mean(LA)', 'SD(LA)',
            'Min(Speed)', 'Max(Speed)', 'Mean(Speed)', 'SD(Speed)', 'Var(Speed)', 'MAD(Speed)', 'Skew(Speed)',
            'Min(Acc)', 'Max(Acc)', 'Mean(Acc)', 'SD(Acc)', 'SoS(Acc)',
            'Mean(Throttle)', 'SD(Throttle)',
            'Mean(Brake)', 'SD(Brake)',
            ]

    listoflists.append(list)

    for csvFile in np.sort(files):
        print('csvFile: ' + str(csvFile))

        # read CSV file
        data_df = utils.load_simulation_data(csvFile)

        # whole simulation metrics
        tot_laps = data_df['lap'].max()
        waypoints_per_lap = data_df['waypoint'].max() + 1  # add one cause it starts from zero
        tot_frames = data_df['frameId'].max()

        tot_obes = data_df['tot_OBEs'].max()
        tot_crashes = data_df['tot_crashes'].max()

        # TODO: add calculation of frame time from sikuli script data
        frame_time = 3

        # Steering Angle => Mean, SD, Var, SAS?, Entropy
        mean_sa = data_df['steering_angle'].mean()
        std_sa = data_df['steering_angle'].std()
        var_sa = data_df['steering_angle'].var()
        steering_angle_speed = get_metric_speed(data_df['steering_angle'], frame_time)
        entropy_sa = entropy(np.absolute(data_df['steering_angle'].values), base=2)

        # TODO: Angular Deviation => Mean. Add other operators?
        mean_ad = data_df['ang_diff'].mean()

        # Lateral Position => Min, Max, Mean, SD, Var, SER, DevT
        min_lp = data_df['cte'].min()
        max_lp = data_df['cte'].max()
        mean_lp = data_df['cte'].mean()
        std_lp = data_df['cte'].std()
        var_lp = data_df['cte'].var()
        ser_lp = None  # same as std_lp?
        devt_lp = None  # we do not have the target

        # Lateral Speed => Mean, SD
        lateral_speed = get_metric_speed(data_df['cte'], frame_time)
        mean_ls = np.min(lateral_speed)
        std_ls = np.std(lateral_speed)

        # Lateral Acceleration => Mean, SD
        lateral_acceleration = get_metric_speed(lateral_speed, frame_time)
        mean_la = np.min(lateral_acceleration)
        std_la = np.std(lateral_acceleration)

        # Driven distance
        driven_distance = data_df['distance'].max()

        # Driven time (in seconds)
        driven_time = data_df['time'].max()

        # Speed => Min, Max, Mean, SD, Var, AbsDev, Skew,
        # TODO: MeanDev, DevT, CountC, TimeC
        min_speed = data_df['speed'].min()
        max_speed = data_df['speed'].max()
        mean_speed = data_df['speed'].mean()
        std_speed = data_df['speed'].std()
        var_speed = data_df['speed'].var()
        mad_speed = data_df['speed'].mad()
        skew_speed = data_df['speed'].skew()

        # Acceleration => Mean, Min, Max, SD, SoS
        speed_array = data_df['speed']
        acceleration = get_metric_speed(speed_array, frame_time)
        mean_acc = np.mean(acceleration)
        min_acc = np.min(acceleration)
        max_acc = np.max(acceleration)
        std_acc = np.std(acceleration)
        sos_acc = np.sum(acceleration ** 2)

        # Throttle => Mean, Std
        # It defines Acceleration, or gas pedal
        mean_throttle = data_df['throttle'].mean()
        std_throttle = data_df['throttle'].std()

        # Brake => Mean, Std
        mean_brake = data_df['brake'].mean()
        std_brake = data_df['brake'].std()

        short_filename = csvFile.replace(args.data_dir, '')
        short_filename = short_filename.replace('_1.h5.csv', '')
        short_filename = short_filename.replace('/', '')

        list = [str(short_filename), str(driven_distance), str(driven_time), str(tot_crashes), str(tot_obes),
                str(mean_sa), str(std_sa), str(var_sa), str(entropy_sa),
                str(mean_ad),
                str(min_lp), str(max_lp), str(mean_lp), str(std_lp), str(var_lp),
                str(mean_ls), str(std_ls),
                str(mean_la), str(std_la),
                str(min_speed), str(max_speed), str(mean_speed), str(std_speed), str(var_speed), str(mad_speed),
                str(skew_speed),
                str(min_acc), str(max_acc), str(mean_acc), str(std_acc), str(sos_acc),
                str(mean_throttle), str(std_throttle),
                str(mean_brake), str(std_brake)
                ]

        listoflists.append(list)

    final_data = zip(*listoflists)
    return final_data


def write_data_to_csv(csv_file, final_data):
    if os.path.exists(csv_file):
        os.remove(csv_file)

    with open(csv_file, 'a') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        for list in final_data:
            writer.writerow(list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driving Quality Metrics')
    parser.add_argument('-d', help='data dir', dest='data_dir', type=str, default='test_simulation')
    parser.add_argument('-n', help='simulation name', dest='sim_name', type=str, default='')
    args = parser.parse_args()

    data = get_metrics_from_csv_files(args)
    write_data_to_csv('driving_metrics_values.csv', data)
