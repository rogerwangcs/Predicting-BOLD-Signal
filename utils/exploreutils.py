from scipy import stats
from utils.loadutils import loadFMRI

def explore_features_data(feature_models):
    for feature_model in encoded_feature_types:
        for run in range(8):
            print('Model {} run {}: '.format(encoded_feature_type, run+1), end='')
            [anat_data, mask_data] = loadFMRI(sub, run)

            features = None
            if encoded_feature_type == 'Pixels':
                features = loadPixelFeatures(Paths.movie_run_snapshots_path(run), run, responseLen=anat_data.shape[3])
            else:
                features = loadFeatures(encoded_feature_type, run, responseLen=anat_data.shape[3], save_name=features_save_name)
            features_conv = convolveFeatures(features, 2)

            s = stats.describe(features_conv, axis=None)
            print('Min: {:.2f} Max: {:.2f} Mean: {:.2f} Std: {:.2f} Var: {:.2f}'.format(s[1][0], s[1][1], s[2], s[3], s[4]))


def explore_fmri_data():
    for run in range(8):
        for sub in range(1, 5+1):
            print('Sub {} run {}: '.format(sub, run+1), end='')
            [func_values, mask_data] = loadFMRI(sub, run)
            func_reshaped = np.reshape(func_values,[func_values.shape[0]*func_values.shape[1]*func_values.shape[2],func_values.shape[3]])
            mask_reshaped = np.reshape(mask_data, -1).astype(bool)
            func_data_roi = func_reshaped[mask_reshaped, :]
            s = stats.describe(func_data_roi, axis=None)
            print('Min: {:6.2f} Max: {:6.2f} Mean: {:6.2f} Var: {:.2f} Skew: {:.2f}'.format(s[1][0], s[1][1], s[2], s[3], s[4]))
