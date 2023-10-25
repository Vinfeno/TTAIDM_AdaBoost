import numpy as np


# Define the Decision Stump classifier
class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None


# AdaBoost with Decision Stumps
class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        m, n = X.shape

        # Initialize weights
        w = np.full(m, 1 / m)

        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()

            # Set initial minimum error to infinity
            min_error = float("inf")
            # train one classifier for each feature and each possible threshold with both polarities
            for feature in range(n):
                feature_values = np.sort(np.unique(X[:, feature]))
                thresholds = (feature_values[:-1] + feature_values[1:]) / 2
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        # predict
                        y_pred = np.ones(m)
                        y_pred[polarity * X[:, feature] < polarity * threshold] = -1
                        # calcculate error
                        error = w[(y_pred != y)].sum()

                        # Save the best stump
                        if error < min_error:
                            clf.polarity = polarity
                            clf.threshold = threshold
                            clf.feature_index = feature
                            min_error = error

            # Calculate alpha (clf weight)
            eps = 1e-10  # used to prevent zero division
            clf.alpha = 0.5 * np.log((1.0 - min_error + eps) / (min_error + eps))

            # Update sample weights
            y_pred = np.ones(m)
            y_pred[
                clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold
            ] = -1
            w *= np.exp(-clf.alpha * y * y_pred)
            w /= np.sum(w)

            # Save the classifier
            self.clfs.append(clf)

    def predict(self, X):
        m, _ = X.shape
        y_pred = np.zeros(m)
        for clf in self.clfs:
            pred = np.ones(m)
            pred[
                clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold
            ] = -1
            y_pred += clf.alpha * pred
        return np.sign(y_pred)


# Example usage
if __name__ == "__main__":
    X = np.array([[0, 1], [1, 0], [1, 1], [2, 2]])
    y = np.array([1, 1, -1, -1])

    print(f"X:\n{X}")
    print(f"shape of X: {X.shape}")
    clf = AdaBoost(n_clf=3)
    clf.fit(X, y)
    predictions = clf.predict(X)
    print(f"True labels: {y}")
    print(f"Predictions: {predictions}")
