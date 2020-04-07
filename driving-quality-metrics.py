import argparse

import numpy as np

import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driving Quality Metrics')
    parser.add_argument('-d', help='data save directory', dest='data_dir', type=str,
                        default='simulations')
    parser.add_argument('-n', help='simulation name', dest='sim_name', type=str, default='run1-foggy')
    parser.add_argument('-g', help='granularity', dest='granularity', type=str, default='per-sector',
                        choices=['per-sector', 'per-frame'])

    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    # read CSV file
    data_df = utils.load_driving_data(args)

    # check that the desired files/values are present
    # TODO

    # whole simulation metrics
    tot_laps = data_df['lap'].max()
    waypoints_per_lap = data_df['waypoint'].max() + 1  # add one cause it starts from zero

    tot_obes = data_df['tot_OBEs'].max()
    tot_crashes = data_df['tot_crashes'].max()

    mean_speed = data_df['speed'].mean()
    std_speed = data_df['speed'].std()

    mean_cte = data_df['cte'].mean()
    std_cte = data_df['cte'].std()

    track_completed = 0.0
    sector_completed = 0.0

    # per-sector metric: for all sectors (pair of consecutive waypoints), calculate the metrics in that sector
    for lap in range(tot_laps):
        for wp in range(waypoints_per_lap):
            rows = data_df[data_df['waypoint'] == wp]
            rows = rows[['frameId', 'lap', 'waypoint', 'speed', 'cte', 'crashed']]

            # percentage of sector completed
            completed = (1 - (rows['crashed'] > 0).mean()) * 100
            assert completed <= 100

            sector_completed += completed

            print("sector %d: avg CTE %.4f - mean speed %f - completed: %.2f" % (
                wp, np.absolute(rows['cte'].values).mean(),
                np.absolute(rows['speed'].values).mean(),
                completed,
            ))

            # per-frame metric: for all frames, calculate the metrics in that frame
            if args.granularity == "per-frame":
                for ind in rows.index:
                    print("\tFrameId %d: CTE %.4f - speed %.2f - crash: %d" % (
                        rows['frameId'][ind],
                        rows['cte'][ind],
                        rows['speed'][ind],
                        rows['crashed'][ind],
                    ))

        print('-' * 30)
        print("Percentage track completed: %.2f %%" % float(np.true_divide(sector_completed, waypoints_per_lap)))
        print("Simulation's mean speed: %.2f mph (std %.2f)" % (mean_speed, std_speed))
        print("Simulation's mean CTE: %.2f (std %.2f)" % (mean_cte, std_cte))
        print("Crashes: %d - OBEs: %d" % (tot_crashes, tot_obes))
