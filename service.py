import mysql.connector
import const

def read_model_list():
    conn = mysql.connector.connect(user=const.db_user_name, password=const.db_password, database=const.db_database
                                   , auth_plugin='mysql_native_password')
    cursor = conn.cursor()
    cursor.execute('select * from model_manage')
    model_list = cursor.fetchall()
    cursor.close()
    conn.close()
    return model_list