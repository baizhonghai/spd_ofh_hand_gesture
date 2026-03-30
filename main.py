from util.param_load import config
from util.random_util import read_random_list
from util.analyze_util import print_mean_std, show_confusion_matrix_colorful
from util.train_util import generate_standard_train_test_data, train_predict
from accepted_preprocess import generate_feature_label_info

if __name__ == '__main__':

    # step 1 数据库选择从配置中读取，根据选择的数据库从另一个函数中返回特征
    dataset = config['common_settings']['data_set']
    feature_type = config['common_settings']['feature_type']
    test_size = float(config['common_settings']['test_size'])
    iteration = int(config['common_settings']['iteration'])
    feature_X, label_y, instance_name_list = generate_feature_label_info(feature_type, dataset)
    # step 2.拿到特征进行特征处理，而后进行分类
    random_state_list = [43] if iteration == 1 else read_random_list()
    accuracy_list = []

    for rand_num in random_state_list:
        X_train, X_test, Y_train, Y_test, verify_train, verify_test = generate_standard_train_test_data(feature_X,
                                                                                                        label_y,
                                                                                                        instance_name_list,
                                                                                                        rand_num,
                                                                                                        test_size)
        accuracy, y_pred = train_predict(X_train, Y_train, X_test, Y_test, show_report=True)
        print(f'Accuracy:{accuracy}, rand:{rand_num}')
        accuracy_list.append(accuracy)
        show_confusion_matrix_colorful(Y_test, y_pred, dataset) if len(random_state_list) == 1 else None
    # step 3 打印结果
    print_mean_std(accuracy_list)
