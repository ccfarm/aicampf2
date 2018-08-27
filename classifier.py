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
        self.clf = LogisticRegression(**param_dict)
        self.clf.fit(X, y)

    def test_classifier(self, csv_path):
        data = pd.read_csv(csv_path)
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0]
        yy = self.clf.predict(X)
        y = list(y)
        yy = list(yy)
        print y
        print yy

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
