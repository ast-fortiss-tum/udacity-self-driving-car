import os

import numpy as np
import pandas as pd
from scipy.stats import entropy, pointbiserialr, pearsonr


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


def load_simulation_data(csvFile) -> object:
    data_df = None
    path = None
    try:
        path = os.path.join(csvFile)
        data_df = pd.read_csv(path, keep_default_na=False)
    except FileNotFoundError:
        print("Unable to read file %s" % path)
        exit()

    return data_df


def get_sim_level_metrics(data_df, args, csvFile, listoflists):
    """
    whole simulation metrics
    :param data_df:
    :return:
    """

    tot_obes = data_df['tot_OBEs'].max()
    tot_crashes = data_df['tot_crashes'].max()

    # TODO: add calculation of frame time from sikuli script data

    # Driven time (in seconds)
    driven_time = data_df['time'].max()
    # TODO: replace this with actual value
    frame_time = 3

    # Steering Angle
    min_sa = data_df['steering_angle'].min()
    max_sa = data_df['steering_angle'].max()
    mean_sa = data_df['steering_angle'].mean()
    std_sa = data_df['steering_angle'].std()

    # Lateral Position => Min, Max, Mean, SD, Var, SER, DevT
    min_lp = data_df['cte'].min()
    max_lp = data_df['cte'].max()
    mean_lp = data_df['cte'].mean()
    std_lp = data_df['cte'].std()

    lateral_speed = get_metric_speed(data_df['cte'], frame_time)
    lateral_acceleration = get_metric_speed(lateral_speed, frame_time)

    # Driven time (in seconds)
    driven_time = data_df['time'].max()

    # Speed
    min_speed = data_df['speed'].min()
    max_speed = data_df['speed'].max()
    mean_speed = data_df['speed'].mean()
    std_speed = data_df['speed'].std()

    # Acceleration
    speed_array = data_df['speed']
    acceleration = get_metric_speed(speed_array, frame_time)
    mean_acc = np.mean(acceleration)
    min_acc = np.min(acceleration)
    max_acc = np.max(acceleration)
    std_acc = np.std(acceleration)

    # Brake
    min_brake = data_df['brake'].min()
    max_brake = data_df['brake'].max()
    mean_brake = data_df['brake'].mean()
    std_brake = data_df['brake'].std()

    short_filename = csvFile.replace(args.data_dir, '')
    short_filename = short_filename.replace('_1.h5.csv', '')
    short_filename = short_filename.replace('/', '')

    metric_info = [str(short_filename), str(driven_time),
                   min_sa, max_sa, mean_sa, std_sa,
                   min_lp, max_lp, mean_lp, std_lp,
                   lateral_speed, lateral_acceleration,
                   min_speed, max_speed, mean_speed, std_speed,
                   mean_acc, min_acc, max_acc, std_acc,
                   min_brake, max_brake, mean_brake, std_brake,
                   str(tot_crashes), str(tot_obes),
                   ]

    return metric_info

    # calculate_correlation()


def calculate_correlation(data_df, window_size, reaction_time):
    # first we need to relabel sectors
    pass


def get_sector_level_metrics(data_df, data_dir, csvFile):
    tot_laps = data_df['lap'].max()
    waypoints_per_lap = data_df['waypoint'].max() + 1  # add one cause it starts from zero

    track_completed = 0.0
    sector_completed = 0.0

    # TODO: get actual value
    frame_time = 3
    metric_waypoint_list = []

    header_list = ['Filename', 'Lap', 'WP', 'Time', 'Min(SA)', 'Max(SA)', 'Mean(SA)', 'Std(SA)', 'Min(LP)', 'Max(LP)',
                   'Mean(LP)',
                   'Std(LP)', 'Min(LS)', 'Max(LS)', 'Mean(LS)', 'Std(LS)',
                   'Min(LA)', 'Max(LA)', 'Mean(LA)', 'Std(LA)', 'Min(Speed)', 'Max(Speed)', 'Mean(Speed)', 'Std(Speed)',
                   'Min(Acc)', 'Max(Acc)', 'Mean(Acc)', 'Std(Acc)',
                   'Min(Brake)', 'Max(Brake)', 'Mean(Brake)', 'Std(Brake)',
                   'Crashes', 'OBEs']

    metric_waypoint_list.append(header_list)

    # per-sector metric: for all sectors (pair of consecutive waypoints), calculate the metrics in that sector
    for lap in range(1, tot_laps + 1):
        for wp in range(waypoints_per_lap):
            # rows = data_df[(data_df['waypoint'] == wp) & (data_df['lap'] == lap)]
            # rows = rows[
            #     ['frameId', 'steering_angle', 'loss', 'threshold', 'lap', 'waypoint', 'throttle', 'speed', 'brake',
            #      'cte', 'crashed', 'tot_OBEs']]

            # percentage of sector completed
            # completed = (1 - (rows['crashed'] > 0).mean()) * 100
            # assert completed <= 100

            # print("lap %d sector %d completed: %.2f" % (lap, wp, completed))

            rows = data_df

            driven_time = np.absolute(rows['steering_angle'].values).max() - np.absolute(
                rows['steering_angle'].values).min() + 1

            # Steering Angle => Mean, SD, Var, SAS?, Entropy
            min_sa = np.absolute(rows['steering_angle'].values).min()
            max_sa = np.absolute(rows['steering_angle'].values).max()
            mean_sa = np.absolute(rows['steering_angle'].values).mean()
            std_sa = np.absolute(rows['steering_angle'].values).std()

            # Lateral Position => Min, Max, Mean, SD, Var, SER, DevT
            min_lp = np.absolute(rows['cte'].values).min()
            max_lp = np.absolute(rows['cte'].values).max()
            mean_lp = np.absolute(rows['cte'].values).mean()
            std_lp = np.absolute(rows['cte'].values).std()

            lateral_speed = get_metric_speed(np.absolute(rows['cte'].values), frame_time)
            lateral_acceleration = get_metric_speed(lateral_speed, frame_time)

            min_ls = np.min(lateral_speed)
            max_ls = np.max(lateral_speed)
            mean_ls = np.mean(lateral_speed)
            std_ls = np.std(lateral_speed)

            min_la = np.min(lateral_acceleration)
            max_la = np.max(lateral_acceleration)
            mean_la = np.mean(lateral_acceleration)
            std_la = np.std(lateral_acceleration)

            min_speed = np.absolute(rows['speed'].values).min()
            max_speed = np.absolute(rows['speed'].values).max()
            mean_speed = np.absolute(rows['speed'].values).mean()
            std_speed = np.absolute(rows['speed'].values).std()

            acceleration = get_metric_speed(np.absolute(rows['speed'].values), frame_time)
            mean_acc = np.mean(acceleration)
            min_acc = np.min(acceleration)
            max_acc = np.max(acceleration)
            std_acc = np.std(acceleration)

            min_brake = np.absolute(rows['brake'].values).min()
            max_brake = np.absolute(rows['brake'].values).max()
            mean_brake = np.absolute(rows['brake'].values).mean()
            std_brake = np.absolute(rows['brake'].values).std()

            tot_obes = np.absolute(rows['tot_OBEs'].values).max()
            tot_crashes = np.absolute(rows['crashed'].values).max()

            scores = np.asarray(rows['score'].values)
            # cte = pd.DataFrame(rows['cte'])
            #
            # print(cte.corr(scores))

            corr, pv = pearsonr(np.array(rows['cte'].values), scores)
            print('Spearman\'s correlation for std_lp: %.3f - %.2f' % (corr, pv))

            short_filename = csvFile.replace(data_dir, '')
            short_filename = short_filename.replace('_1.h5.csv', '')
            short_filename = short_filename.replace('/', '')

            metric_list = [str(short_filename), str(lap), str(wp),
                           str(driven_time),
                           min_sa, max_sa, mean_sa, std_sa,
                           min_lp, max_lp, mean_lp, std_lp,
                           min_ls, max_ls, mean_ls, std_ls,
                           min_la, max_la, mean_la, std_la,
                           min_speed, max_speed, mean_speed, std_speed,
                           mean_acc, min_acc, max_acc, std_acc,
                           min_brake, max_brake, mean_brake, std_brake,
                           str(tot_crashes),
                           str(tot_obes)]

            # sector_completed += completed
            # track_completed += completed

            metric_waypoint_list.append(metric_list)

            # print("lap %d sector %d ::= avg CTE %.4f - avg loss %.4f - mean speed %f - completed: %.2f" % (
            #    lap, wp, avg_cte_sector, avg_loss_sector, avg_speed_sector, completed))
    return metric_waypoint_list
