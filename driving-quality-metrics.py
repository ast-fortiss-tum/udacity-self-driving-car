import argparse
import csv
import glob
import os
from scipy.stats import pearsonr

import numpy as np

from utils_driving_quality_metrics import load_simulation_data, get_sim_level_metrics, get_sector_level_metrics


def get_metrics_from_csv_files(args):
    files = glob.glob(args.data_dir + "/*")  # all CSV files
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

    for file in np.sort(files):
        # csvFile = file + "/driving_log.csv"
        # print('csvFile: ' + str(csvFile))
        csvFile = file
        print('csvFile: ' + str(csvFile))
        # names = ['frameId', 'model', 'anomaly_detector', 'threshold', 'sim_name', 'lap', 'waypoint', 'loss', 'cte',  'steering_angle',
        #          'throttle', 'speed', 'brake', 'crashed', 'distance', 'time', 'ang_diff', 'center', 'tot_OBEs', 'tot_crashes']
        if os.path.exists(csvFile):
            # read CSV file
            data_df = load_simulation_data(csvFile)

            if args.granularity == "simulation":
                sim_level_metrics = get_sim_level_metrics(data_df, args, csvFile, listoflists)
            elif args.granularity == "sector":
                sector_level_metrics = get_sector_level_metrics(data_df, args.data_dir, csvFile)
                output_file = csvFile.replace('.csv', '_output.csv')
                write_data_to_csv(output_file, sector_level_metrics)
            elif args.granularity == "frame":
                # frame_level_metrics = get_frame_level_metrics(data_df)
                print('Frame level metrics not yet implemented.')
                exit(0)

    final_data = zip(*listoflists)
    return final_data


def write_data_to_csv(csv_file, final_data):
    if os.path.exists(csv_file):
        os.remove(csv_file)  # remove the file if present, it regenerates the results all the time

    with open(csv_file, 'a') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        for list in final_data:
            writer.writerow(list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driving Quality Metrics')
    parser.add_argument('-d', help='data dir', dest='data_dir', type=str,
                        default='icst21_rq1')
    parser.add_argument('-n', help='simulation name', dest='sim_name', type=str, default='')
    parser.add_argument('-g', help='granularity', dest='granularity', type=str, default='sector',
                        choices=['simulation', 'sector', 'frame'])
    args = parser.parse_args()

    data = get_metrics_from_csv_files(args)
    write_data_to_csv('driving_metrics_values.csv', data)
