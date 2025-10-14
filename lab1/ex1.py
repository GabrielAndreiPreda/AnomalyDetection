import pyod as od
from matplotlib import pyplot

def main():
    # generate data
    X_train, y_train, X_test, y_test = od.utils.generate_data(
        n_train=400, n_test=100, n_features=2, contamination=0.1, random_state=123, behaviour='old'
    )
    pyplot.scatter(X_train[:, 0], X_train[:, 1], c='blue', label='Train data')
    pyplot.scatter(X_test[:, 0], X_test[:, 1], c='red', label='Test data')
    pyplot.legend()
    pyplot.show()

if __name__ == '__main__':
    main()
    