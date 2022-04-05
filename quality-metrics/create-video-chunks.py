import argparse
import csv
import os
import numpy as np

import utils
import itertools
from itertools import combinations, permutations

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


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


def get_all_possible_video_chunks(args, csv, video, window_size):

    l = np.arange(27)
    contiguous = [l[x:y] for x in range(len(l) + 1) for y in range(x + 1, len(l) + 1)]
    for i in list(contiguous):
        if len(i) == window_size:

            print(i)

            # read CSV file
            df = utils.load_simulation_data(csv)

            # df = df[['lap', 'waypoint', 'time']]

            csv_chunk = df[(df['lap'] == 1) & (df['waypoint'] >= i[0]) & (df['waypoint'] <= i[-1])]

            start_time = csv_chunk['time'].iloc[0]

            end_time = csv_chunk['time'].iloc[-1]

            chunk_name = os.path.join(str(args.data_dir), str(args.sim_name), 'chunk-' + str(i[0]) + '-' + str(i[-1]) + '.mp4')
            chunk_name_csv = os.path.join(str(args.data_dir), str(args.sim_name), 'chunk-' + str(i[0]) + '-' + str(i[-1]) + '.csv')

            csv_chunk.to_csv(chunk_name_csv, index=False, header=True)

            ffmpeg_extract_subclip(video, start_time, end_time, targetname=chunk_name)


def get_video_chunk(args, csv, video, start, end):
    # read CSV file
    df = utils.load_simulation_data(csv)

    df = df[['lap', 'waypoint', 'time']]

    csv_chunk = df[(df['lap'] == 1) & (df['waypoint'] >= start) & (df['waypoint'] <= end)]

    start_time = csv_chunk['time'].iloc[0]

    end_time = csv_chunk['time'].iloc[-1]

    chunk_name = os.path.join(str(args.data_dir), str(args.sim_name), 'chunk-' + str(start) + '-' + str(end) + '.mp4')

    ffmpeg_extract_subclip(video, start_time, end_time, targetname=chunk_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driving Quality Metrics')
    parser.add_argument('-d', help='data dir', dest='data_dir', type=str, default='simulations')
    parser.add_argument('-n', help='simulation name', dest='sim_name', type=str, default='video-track2')
    args = parser.parse_args()

    csv = os.path.join(str(args.data_dir), str(args.sim_name), 'driving_log.csv')
    video = os.path.join(str(args.data_dir), str(args.sim_name), 'video.mov')

    # start = 0
    # end = 3
    # get_video_chunk(args, csv, video, start, end)

    chunk_size = 3

    get_all_possible_video_chunks(args, csv, video, chunk_size)
