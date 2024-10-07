# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:12:54 2023

@author: Solido
"""

import datetime
import pandas as pd
import numpy as np
import psycopg2 as psycopg


def load_dataset_sql(startDate = datetime,
                     endDate = datetime,
                     server = '127.0.0.1', 
                     database = 'CTraderBot', 
                     uname = "postgres", 
                     pword = "Langley2552",
                     port = 5432
                     ):

    
    conn = psycopg.connect(host=server, 
                           dbname= database,
                           user= uname,
                           password = pword,
                           port = port
                           )
    

    
    #fecha1 = __dataConvert(startDate)
    #fecha2 = __dataConvert(endDate)
    fecha1 = startDate
    fecha2 = endDate
    
    try:

        input_query = 'SELECT * FROM "TrendBars" where "Time" between '"'{}'"' and '"'{}'"' order by "Time" asc'.format(fecha1, fecha2)
        
        sqlTable = conn.execute(input_query).fetchall()
        sqlTable = pd.DataFrame(sqlTable, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

        return sqlTable
        #return sqlTable.set_index('Time')

    
    except:
        print("Error in conexion to SQL")
    finally:
        conn.close()


def state_3d_creator(df, lag):

    # Creamos la variable X_train
    X_train = []
    #Bucle x cada día
    for variable in range(0, df.shape[1]):
        X = []
        #numero de días q voy hacia atras
        for i in range(lag, df.shape[0]+1):
            X.append(df[i-lag:i, variable])
        X_train.append(X)
    X_train, np.array(X_train)
    X_train = np.swapaxes(np.swapaxes(X_train, 0, 1), 1, 2)

    return X_train
        
"""
def __dataConvert(data):
    day, Hour  = str(data).split(" ")
    dateArraySQL = day.split("-")
    return dateArraySQL[0]+dateArraySQL[1]+dateArraySQL[2] + " " + Hour
"""
"""

start_date = '2023-01-01 00:00:00'
end_date = '2023-05-01 00:00:00'

df = load_dataset_sql(startDate=start_date, endDate=end_date)

df2 = df.drop(["Time"], axis=1)
array = df2.to_numpy()

print(array[1][3])
dato = df.iloc[5]
convert = dato["Time"]

convert = pd.DataFrame(df, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
convert = convert.set_index('Time')


"""








