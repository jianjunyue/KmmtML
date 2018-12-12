from __future__ import division
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0 - epsilon)
    return - np.mean(actual * np.log(attempt) +
                     (1.0 - actual) * np.log(1.0 - attempt))


if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False

    path = "train.csv"
    data = pd.read_csv(path)
    path_test = "test.csv"
    data_test = pd.read_csv(path_test)
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "NameLength", "Title"]
    train = data[predictors]
    y = data["Survived"]
    test = data_test[predictors]

    X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=111)

    # X, y, X_submission = load_data.load()
    X=X_train.values
    y=y_train.values
    X_submission=X_test.values

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print("Creating train and test sets for blending.")

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        # print(j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            # print("Fold", i)
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            # y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    # y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
    y_submission = clf.predict(dataset_blend_test)

    print("Linear stretch of predictions to [0,1]")
    # y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    y_pred=y_submission
    accuracy_score = metrics.accuracy_score(y_pred, y_test)
    print("Blend : ", accuracy_score)
    # print
    # "Saving Results."
    # tmp = np.vstack([range(1, len(y_submission) + 1), y_submission]).T
    # np.savetxt(fname='submission.csv', X=tmp, fmt='%d,%0.9f',
