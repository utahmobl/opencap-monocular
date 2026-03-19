# function which loads a csv file at specified path and computes the min, avg, max, mean and std of the data in the file across columns
import pandas as pd
import os


def summary_hp_results(path, filename):
    df = pd.read_csv(os.path.join(path, filename))
    # write a new csv file with the min, avg, max, mean and std of the data in the file across columns
    df_summary = df.describe()
    df_summary.to_csv(os.path.join(path, filename))

    # compute the stats per subject, per session, per camera, per movement
    df_summary = df.groupby(["subject"]).describe()
    df_summary.to_csv(os.path.join(path, "summary_hp_results_grouped_subject.csv"))

    df_summary = df.groupby(["session"]).describe()
    df_summary.to_csv(os.path.join(path, "summary_hp_results_grouped_session.csv"))

    df_summary = df.groupby(["movement"]).describe()
    df_summary.to_csv(os.path.join(path, "summary_hp_results_grouped_movement.csv"))

    df_summary = df.groupby(["cam"]).describe()
    df_summary.to_csv(os.path.join(path, "summary_hp_results_grouped_cam.csv"))

    return


csv_path = "/home/selim/opencap-mono/output/reprojection-20_contact_velocity-5_contact_position-75_smoothness_diff-10_flat_floor-10_stability-10_walking-6_squats-4_STS-4_DJ-6"
filename = "results.csv"
summary_hp_results(path=csv_path, filename=filename)
