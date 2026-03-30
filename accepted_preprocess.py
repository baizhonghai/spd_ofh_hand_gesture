from util.feature_util import read_from_feature_file, feature_combine
import os
from feature_extract_object.OpticalFlow import OpticalFlow

dataset_feature_paths = {
    'North': {
        'spd': './data/north_1_spd_X_y_verify_original.pt',
        'ofh': './data/north_hof_X_y.pt_key_frames_new_16_bin8'
    },
    'Cambridge': {
        'spd': './data/cambridge_1_spd_X_y_verify.pt',
        'ofh': './data/cambridge_hof_X_y_key_frames_06_bin8'
    }
}





directory_paths = {
    'North': '/Users/xxx/TP/HandGestureRecognition/datasets/Northwestern_Hand_Gesture/',
    'Cambridge': '/Users/xxx/TP/HandGestureRecognition/datasets/Cambridge_Hand_Gesture'
}


def test_not_exist_create(feature_type, feature_paths, dataset):
    feature_path = feature_paths.get(feature_type)
    if feature_path is not None:
        if not os.path.isfile(feature_path):
            generate_feature(feature_type, feature_paths, dataset)
    else:
        for key, value in feature_paths.items():
            if not os.path.isfile(value):
                generate_feature(key, feature_paths, dataset)


def generate_feature(feature_type, feature_paths, dataset):
    if feature_type == 'spd':
        raise Exception("it's matlab code for spd, check the path")
    else:
        optical_flow = OpticalFlow(12, directory_paths.get(dataset), feature_paths.get(feature_type))
        optical_flow.generate()


def generate_feature_label_info(feature_type, dataset):
    if dataset not in ['North', 'Cambridge']:
        raise Exception("dataset except 'north','cambridge',but get" + str(dataset))
    if feature_type not in ['spd', 'ofh', 'spd_ofh']:
        raise Exception("feature type expect 'spd','ofh', or 'spd_ofh', but received " + str(feature_type))
    feature_paths = dataset_feature_paths.get(dataset)
    test_not_exist_create(feature_type, feature_paths, dataset)
    # 就这样不写成for了，凑合用
    if feature_type in ['spd', 'ofh']:
        feature_X, label_y, instance_name_list = read_from_feature_file(feature_paths.get(feature_type))
    else:
        feature_X_spd, label_y, instance_name_list = read_from_feature_file(feature_paths.get('spd'))
        feature_X_ofh, label_y, instance_name_list = read_from_feature_file(feature_paths.get('ofh'))
        feature_X = feature_combine(feature_X_spd, feature_X_ofh)

    return feature_X, label_y, instance_name_list
