import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import mysql.connector
import const



class Classifier:
    def __init__(self, csv_path, param_dict):
        data = pd.read_csv(csv_path)
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        self.clf = LogisticRegression(**param_dict)
        self.clf.fit(X_train, y_train)
        self.score = self.clf.score(X_test, y_test)

    def test_classifier(self, csv_path):
        data = pd.read_csv(csv_path)
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0]
        yy = self.clf.predict(X)
        y = list(y)
        yy = list(yy)
        size = len(y)
        count = 0
        for i in xrange(size):
            if y[i] == yy[i]:
                count += 1
        p = float(count) / size
        return {'p' : p}

    def get_score(self):
        return self.score


    def save_clf(self, name):
        model_path = "model/" + name + ".m"
        joblib.dump(self.clf, model_path)
        conn = mysql.connector.connect(user=const.db_user_name, password=const.db_password, database=const.db_database
                                       ,auth_plugin='mysql_native_password')
        cursor = conn.cursor()
        cursor.execute('insert into model_manage (model_name, model_zhonglei, model_zuoyong, model_address) '
                       'values (%s, %s, %s, %s)', [name, 'LR', 'classification', model_path])
        conn.commit()
        cursor.close()
        conn.close()
        return None


    # def predict(self, csv):
    #     X = pd.read_csv()
    #     y = self.clf.predict(X)
    #     data = pd.concat(y, X, axis=1)
    #     data.to_csv()

def load_classifier(id):
    conn = mysql.connector.connect(user=const.db_user_name, password=const.db_password, database=const.db_database
                                   , auth_plugin='mysql_native_password')
    cursor = conn.cursor()
    cursor.execute('select * from model_manage where id = %s', [id])
    model = cursor.fetchone()
    cursor.close()
    conn.close()
    path = model[4]
    return joblib.load(path)


