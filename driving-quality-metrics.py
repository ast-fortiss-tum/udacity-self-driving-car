import argparse

import numpy as np

import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driving Quality Metrics')
    parser.add_argument('-d', help='data save directory', dest='data_dir', type=str, default='simulations')
    parser.add_argument('-n', help='simulation name', dest='sim_name', type=str, default='track1-sunny')
    parser.add_argument('-g', help='granularity', dest='granularity', type=str, default='sector',
                        choices=['sector', 'frame'])

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
    tot_frames = data_df['frameId'].max()

    tot_obes = data_df['tot_OBEs'].max()
    tot_crashes = data_df['tot_crashes'].max()

    mean_speed = data_df['speed'].mean()
    std_speed = data_df['speed'].std()

    mean_throttle = data_df['throttle'].mean()
    std_throttle = data_df['throttle'].std()

    mean_brake = data_df['brake'].mean()
    std_brake = data_df['brake'].std()

    mean_cte = data_df['cte'].mean()
    std_cte = data_df['cte'].std()

    track_completed = 0.0
    sector_completed = 0.0
    tn = list()
    tn_usable = list()

    # per-sector metric: for all sectors (pair of consecutive waypoints), calculate the metrics in that sector
    for lap in range(1, tot_laps + 1):
        for wp in range(waypoints_per_lap):

            rows = data_df[(data_df['waypoint'] == wp) & (data_df['lap'] == lap)]

            rows = rows[
                ['frameId', 'loss', 'threshold', 'lap', 'waypoint', 'throttle', 'speed', 'brake', 'intensity', 'cte',
                 'crashed']]

            # percentage of sector completed
            completed = (1 - (rows['crashed'] > 0).mean()) * 100
            assert completed <= 100

            # per-sector metrics
            avg_cte_sector = np.absolute(rows['cte'].values).mean()
            std_cte_sector = np.absolute(rows['cte'].values).std()

            avg_loss_sector = np.absolute(rows['loss'].values).mean()
            std_loss_sector = np.absolute(rows['loss'].values).std()

            avg_speed_sector = np.absolute(rows['speed'].values).mean()
            std_speed_sector = np.absolute(rows['speed'].values).std()

            avg_throttle_sector = np.absolute(rows['throttle'].values).mean()
            std_throttle_sector = np.absolute(rows['throttle'].values).std()

            avg_brake_sector = np.absolute(rows['brake'].values).mean()
            std_brake_sector = np.absolute(rows['brake'].values).std()

            sector_completed += completed
            track_completed += completed

            print("lap %d sector %d ::= avg CTE %.4f - avg loss %.4f - mean speed %f - completed: %.2f" % (
                lap, wp, avg_cte_sector, avg_loss_sector, avg_speed_sector, completed))

            # per-frame metric: for all frames, calculate the metrics in that frame
            if args.granularity == "frame":
                for ind in rows.index:
                    print("\tFrameId %d: CTE %.4f - loss %.4f - speed %.2f - intensity: %.2f - crash: %d" % (
                        rows['frameId'][ind],
                        rows['cte'][ind],
                        rows['loss'][ind],
                        rows['speed'][ind],
                        rows['intensity'][ind] * 100,
                        rows['crashed'][ind],
                    ))

                    if rows['loss'][ind] < rows['threshold'][ind]:
                        tn.append(rows['frameId'][ind])

                        if abs(rows['cte'][ind]) < avg_cte_sector and rows['crashed'][ind] == 0:
                            tn_usable.append(rows['frameId'][ind])

        print('-' * 30)
        print("Percentage lap %d completed: %.2f %%" % (lap, float(
            np.true_divide(sector_completed, waypoints_per_lap))))
        print("Lap %d's mean speed: %.2f mph (std %.2f)" % (lap, avg_speed_sector, std_speed_sector))
        print("Lap %d's mean throttle: %.2f (std %.2f)" % (lap, avg_throttle_sector, std_throttle_sector))
        print("Lap %d's mean brake: %.2f (std %.2f)" % (lap, avg_brake_sector, std_brake_sector))
        print("Lap %d's mean CTE: %.2f (std %.2f)" % (lap, avg_cte_sector, std_cte_sector))
        print('-' * 30)

        # reset counter
        sector_completed = 0.0

print("Percentage simulation completed: %.2f %%" % float(
    np.true_divide(track_completed, waypoints_per_lap * tot_laps)))
print("Simulation's mean speed: %.2f mph (std %.2f)" % (mean_speed, std_speed))
print("Simulation's mean throttle: %.2f (std %.2f)" % (mean_throttle, std_throttle))
print("Simulation's mean brake: %.2f (std %.2f)" % (mean_brake, std_brake))
print("Simulation's mean CTE: %.2f (std %.2f)" % (mean_cte, std_cte))
print("Crashes: %d - OBEs: %d" % (tot_crashes, tot_obes))

if args.granularity == "frame":
    print("TN: %d (%.2f %%) - Usable: %d (%.2f %%)" % (
        len(tn), len(tn) / int(tot_frames) * 100, len(tn_usable), len(tn_usable) / int(tot_frames) * 100))
