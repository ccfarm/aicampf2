import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib



class Classifier:
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        self.clf = LogisticRegression()
        self.clf.fit(X_train, y_train)
        self.score = self.clf.score(X_test, y_test)

    def get_score(self):
        return self.score

    def save_clf(self, name):

        joblib.dump(self.clf, "model/" + name + ".m")

        return None


    # def predict(self, csv):
    #     X = pd.read_csv()
    #     y = self.clf.predict(X)
    #     data = pd.concat(y, X, axis=1)
    #     data.to_csv()


