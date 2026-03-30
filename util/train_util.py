from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def generate_standard_train_test_data(X, Y, instance_name_list, random_num, test_size):

    X_train, X_test, Y_train, Y_test, verify_train, verify_test = train_test_split(X, Y, instance_name_list,
                                                                                   test_size=test_size,
                                                                                   random_state=random_num)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test, verify_train, verify_test


def train_predict(X_train, Y_train, X_test, Y_test, show_report=False):
    svm = SVC(C=10)

    svm.fit(X_train, Y_train)
    y_pred = svm.predict(X_test)
    # Accuracy
    accuracy = svm.score(X_test, Y_test)
    class_report = classification_report(Y_test, y_pred, digits=3)
    if show_report:
        print("Classification Report:")
        print(class_report)
    return accuracy, y_pred
