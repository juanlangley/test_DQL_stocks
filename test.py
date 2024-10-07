# Inteligencia artificial aplicada a negocios y empresas
# Fase de testing


import numpy as np
import sys
from environment import Environment
from order_buffer import OrderBuffer
from dataSetLoader import load_dataset_sql
from features import feature_engineering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from brain import Brain
import torch


start_date = '2022-05-21 00:00:00'
end_date = '2022-06-05 00:00:00'

df = load_dataset_sql(startDate=start_date, endDate=end_date)


features_list =  feature_engineering(df).dropna()

prices_list = np.array(df.iloc[features_list.index[0]:])

features_new_list = features_list.reset_index(drop=True)
features_list = np.array(features_new_list)

features_full = features_list
features_list = np.delete(features_list, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 14, 17, 18, 19,22, 23], axis=1)

""" ------------------ StandardScaler ----------------------"""
"""
sc = StandardScaler()
sc_load = False
try:
    sc = joblib.load('standar_scaler/scaler.pkl')
    sc_load = True
except FileNotFoundError:
    print("Error al cargar el archivo scaler. Procure que un archivo para cargar")
    sys.exit()

features_list = sc.transform(features_list)
"""




""" ------------------ MinMaxScaler ----------------------"""

scalers = []
# Cargar todos los escaladores guardados
minmax_load = False
try:
    for i in range(features_list.shape[1]):
        scaler = joblib.load(f'minmax_scalers/scaler_columna_{i}.pkl')
        scalers.append(scaler)
        minmax_load = True
except FileNotFoundError:
    print("Error al cargar el archivo scaler. Procure que un archivo para cargar")
    sys.exit()

# Se aplica el scalado
features_list = np.hstack([scaler.transform(features_list[:, i].reshape(-1, 1)) for i, scaler in enumerate(scalers)])





env = Environment(observationShape= features_list.shape[1] ,price_array = prices_list, features_array= features_list, features_full=features_full)
buffer = OrderBuffer()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q = Brain(env.observation_shape, env.action_shape).to(device)
#Q = Brain(17, 3).to(device)


try:
    file_name = "models/DQN_model.ptm"
    agent_state = torch.load(file_name, map_location = lambda storage, loc: storage)
    Q.load_state_dict(agent_state["Q"])
    Q.to(device)
    print("Cargado del modelo Q desde", file_name)
except FileNotFoundError:
    print("ERROR: No existe ningún modelo entrenado para este entorno.")
    sys.exit()


def detectar_cambios(transiciones):
    cambios = []
    inicio = 0
    valor_anterior = transiciones[0]
    for i in range(0, len(transiciones)):
        if transiciones[i] != valor_anterior:
            #cambios.append((valor_anterior, inicio, i - 1))
            cambios.append((valor_anterior, inicio, i))
            inicio = i
            valor_anterior = transiciones[i]
    cambios.append((valor_anterior, inicio, len(transiciones) - 1))
    return cambios


obs = env.reset(initialIndex=0)
done = False
total_reward = 0.
step = 0

transiciones =  []
data = []

while not done:
    q_values = Q(torch.Tensor(obs).to(device))
    action_probs = q_values.detach().numpy()
    action = np.argmax(action_probs)

    transiciones.append(action)
    
    index = env.index
    next_obs, reward, done = env.step(action)
    
    data.append([index, env.get_price(index)[0], env.get_price(index)[1],  env.get_price(index)[4], reward, env.market_state, env.last_state, env.max_price,action_probs[0], action_probs[1] ])

    total_reward += reward 
    obs = next_obs
    
    step += 1
    
    if done is True:
        print("\nTest finalizado con {} iteraciones, con una Recompensa = {}".
              format(step+1, round(total_reward, 2)))
        data = np.array(data)
        indices_de_cambio = detectar_cambios(transiciones)
                
        for i in range(0, len(indices_de_cambio)):
            initial_date = env.get_price(indices_de_cambio[i][1])[0]
            finish_date = env.get_price(indices_de_cambio[i][2])[0]
            if indices_de_cambio[i][0] == 0:
                order_type = "nan"
            elif indices_de_cambio[i][0]  == 1:
                order_type = "buy"
            elif indices_de_cambio[i][0] == 2:
                order_type = "sell"
            buffer.store(initial_date, finish_date, order_type, 0, 0)
        buffer.export_buffer()


"""        
buffer_get = buffer.get_buffer()

    if env.market_state == 0 and (action == 1 or action == 2):
        #guardo fecha de apertura
        initial_op_index = env.index

    elif env.market_state == 1 and (action == 0 or env.market_state == 2):
        #guardo fecha de cierre
        finish_op_index = env.index
        #guardo la operación en el buffer
        initial_date = str(env.get_price(initial_op_index)[0])
        finish_date = str(env.get_price(finish_op_index)[0])
        
        if env.market_state == 2:
            initial_op_index = env.index
        
        buffer.store(initial_date, finish_date, "indefinido", 0, env.market_state)
    elif env.market_state == 2 and (action == 0 or env.market_state == 1):
        #guardo fecha de cierre
        finish_op_index = env.index
        #guardo la operación en el buffer

        initial_date = str(env.get_price(initial_op_index)[0])
        finish_date = str(env.get_price(finish_op_index)[0])
        
        if env.market_state == 1:
            initial_op_index = env.index
            
        buffer.store(initial_date, finish_date, "indefinido", 0, env.market_state)
"""
"""
order_buf = buffer.order_buffer

A_obs = env.reset(initialIndex=13936)

A_done = False
A_total_reward = 0.
A_step = 0

A_q_values = Q(torch.Tensor(A_obs).to(device))
A_action_probs = A_q_values.detach().numpy()
A_action = np.argmax(A_action_probs)
A_index = env.index

A_next_obs, A_reward, A_done = env.step(A_action)


A_total_reward += A_reward
A_obs = A_next_obs

A_step += 1
"""
# ---------------------------------------------------
# COMPARACIONES DE STANDAR SCALER
"""
from sklearn.preprocessing import MinMaxScaler

features_standar_full = features_full[:,6]
features_standar_full_transform = np.reshape(features_standar_full, (-1,1))

#Standar Scaler
sc_j = StandardScaler()
features_sc =  sc.fit_transform(features_standar_full_transform)

#Max Min Scaler
sc_maxmin = MinMaxScaler()
features_minmax = sc_maxmin.fit_transform(features_standar_full_transform)
features_standar_full_transform = sc.transform(features_standar_full)

#Crear array con scalers individuales para cada columnar
features_xscalers = [MinMaxScaler() for _ in range(features_full.shape[1])]

#se concatenan las columnas escaladas para generar de nuevo el array
features_xescalados = np.hstack([scaler.fit_transform(features_full[:, i].reshape(-1, 1)) for i, scaler in enumerate(features_xscalers)])

import joblib

# se guardan las columnas escaladas
for i, scaler in enumerate(features_xscalers):
    columna_escalada = scaler.fit_transform(features_full[:, i].reshape(-1, 1))
    joblib.dump(scaler, f'minmax_scalers/scaler_columna_{i}.pkl')


scalers = []
# Cargar todos los escaladores guardados
for i in range(features_full.shape[1]):
    scaler = joblib.load(f'minmax_scalers/scaler_columna_{i}.pkl')
    scalers.append(scaler)
# luego se hace el np.hstack
"""