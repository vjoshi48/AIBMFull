import glob
import argparse
import pandas as pd
from freesurfer_stats import CorticalParcellationStats
from sklearn.model_selection import train_test_split


def get_gm_volume(path):
    """get gm volume from aparc file """
    return CorticalParcellationStats.read(path).structural_measurements[['gray_matter_volume_mm^3']]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="getting train/ val/ test splits for brain morphometry estimation")
    parser.add_argument(
        "--image_dir",
        metavar="DIR",
        default="/data/users2/vjoshi6/MRIDataConverted/",
        help="Base directory of normalizaed HCP brains",
    )
    parser.add_argument(
        "--stats_dir",
        metavar="DIR",
        default="/data/users2/vjoshi6/aibm_labels/",
        help="Base directory of stats",
    )

    args = parser.parse_args()
    mri_paths = glob.glob(args.image_dir + '*')
    stats_dirs = glob.glob(args.stats_dir + '*')

    rh_gm_volume_stats = []
    lh_gm_volume_stats = []
    volume_paths = []
    for stats_dir in stats_dirs:
        subject = stats_dir.split('/')[-1]
        lh_gm_volume_stats.append(
            get_gm_volume(
                stats_dir + "/lh.aparc.stats")
        )
        rh_gm_volume_stats.append(
            get_gm_volume(
                stats_dir + "/rh.aparc.stats")
        )
        volume_paths.extend(
            [mri_path for mri_path in mri_paths if subject in mri_path]
        )

    lh_df = pd.concat([stat.T for stat in lh_gm_volume_stats]).reset_index(drop=True)
    lh_df.columns = ['lh_gm_vol_' + str(col) for col in lh_df.columns]

    rh_df = pd.concat([stat.T for stat in rh_gm_volume_stats]).reset_index(drop=True)
    rh_df.columns = ['rh_gm_vol_' + str(col) for col in rh_df.columns]
    volume_regression_df = pd.concat([lh_df, rh_df], axis=1)
    volume_regression_df['volume_paths'] = volume_paths

    train_df = volume_regression_df.sample(frac=0.7, random_state=200)
    test = volume_regression_df.drop(train_df.index).sample(frac=1)
    val_df = test.sample(frac=.66, random_state=200)
    test_df = test.drop(val_df.index)

    train_df.to_csv('mri_volume_estimation_train1.csv', index=False)
    val_df.to_csv('mri_volume_estimation_val1.csv', index=False)
    test_df.to_csv('mri_volume_estimation_test1.csv', index=False)