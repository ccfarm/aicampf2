import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import mysql.connector
import const
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score#R square

class Regression:
    def __init__(self, csv_path,regression_config):
        data = pd.read_csv(csv_path)
        self.param = str(regression_config)
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # if regression_type == "LinearRegression":
        #     self.clf = LinearRegression()
        # elif regression_type == "SVR":
        #     self.clf = SVR()
        # elif regression_type == "RandomForestRegressor":
        #     self.clf = RandomForestRegressor()

        self.clf = RandomForestRegressor()
        self.clf.set_params(**regression_config)
        self.clf.fit(X_train, y_train)
        self.score = self.clf.score(X_test, y_test)


    def get_score(self):
        return self.score

    def save_clf(self, name):
        model_path = "model/" + name + ".m"
        joblib.dump(self.clf, model_path)
        conn = mysql.connector.connect(user=const.db_user_name, password=const.db_password, database=const.db_database
                                       ,auth_plugin='mysql_native_password')
        cursor = conn.cursor()
        cursor.execute('insert into model_manage (model_name, model_zhonglei, model_zuoyong, model_address, model_config) '
                       'values (%s, %s, %s, %s, %s)', [name, 'RandomForestRegressor', 'regression', model_path, self.param])
        conn.commit()
        cursor.close()
        return None

    def test_regression(self, csv_path):
        data = pd.read_csv(csv_path)
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0]
        yy = self.clf.predict(X)
        y = list(y)
        yy = list(yy)
        mse = mean_squared_error(y,yy)
        mae = mean_absolute_error(y,yy)

        return {'p' : "mse : "+str(mse)+" , mae : "+str(mae)}

def load_regression(id):
    conn = mysql.connector.connect(user=const.db_user_name, password=const.db_password, database=const.db_database
                                   , auth_plugin='mysql_native_password')
    cursor = conn.cursor()
    cursor.execute('select * from model_manage where id = %s', [id])
    model = cursor.fetchone()
    cursor.close()
    conn.close()
    path = model[4]
    return joblib.load(path)