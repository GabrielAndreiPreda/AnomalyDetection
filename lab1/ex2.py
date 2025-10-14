import pyod as od
from matplotlib import pyplot
from pyod.models.knn import KNN  # kNN detector
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


def main():
    # generate data
    X_train, X_test, y_train, y_test = od.utils.generate_data(
        n_train=400, n_test=100, n_features=2, contamination=0.4, random_state=123
    )
    pyplot.scatter(X_train[:, 0], X_train[:, 1], c='blue', label='Train data')
    pyplot.scatter(X_test[:, 0], X_test[:, 1], c='red', label='Test data')
    pyplot.legend()
    pyplot.show()

    # pyod model
    # train kNN detector
    clf_name = 'KNN'
    clf = KNN()
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # it is possible to get the prediction confidence as well
    y_test_pred, y_test_pred_confidence = clf.predict(X_test,
                                                      return_confidence=True)  # outlier labels (0 or 1) and confidence in the range of [0,1]

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)

    visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=False)

    # confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # roc curve
    roc = roc_curve(y_test, y_test_scores)
    pyplot.plot(roc[0], roc[1], label='ROC curve (area = %0.2f)' % auc(roc[0], roc[1]))
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    main()
