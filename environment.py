



import pandas as pd
import numpy as np
import random
import datetime





class Environment(object):
    def __init__(self,
                observationShape = int,
                initialIndex: int=0,
                price_array = [],
                features_array = [],
                features_full = []
                #initialBalance: float=10000.,
                #symbolSFee: float=3.,
                #symbol: str="XAUUSD",
                #symbolDecimal: int=2
                ):
        
        
        # acciones = [fuera de mercado, compra, venta]
        self.actions = [0, 1]
        self.action_shape = len(self.actions)
        
        #self.initial_index = initialIndex
        self.index = initialIndex
        
             
        #self.reward = 0.
        #self.current_profit = 0.
        #[fuera de mercado, compra, venta]
        self.market_state = 0
        self.done = False
        
        #self.balance = initialBalance
        #self.max_balance = initialBalance
        assert price_array.shape[0] == features_array.shape[0] , "the length of the dataframe and the array are not the same" 
   
        self.price_array = price_array
        self.features_array = features_array
        self.observation_shape = observationShape
        
        self.features_full = features_full
        #self.current_symbol = symbol
        #self.symbol_decimal = symbolDecimal
        
        
        self.initial_price_order = 0
        self.finish_price_order = 0

        self.last_state = 0
        self.trailing = 0
        self.date_open = None
        self.max_price = 0
        self.period_count = 0
        self.sum_diferences = 0
        
        
        
        

    """
    STEP
    - Solo cambio de estado con índice - features y precios
    - guarda state y next_state
    
    """
    def step(self, action):
        current_index = self.index
        next_index = self.index + 1
        self.last_state = self.market_state
        
        if next_index == len(self.features_array)-1:
            self.done = True
        
        self.price_obs = self.price_array[current_index]
        self.price_next_obs = self.price_array[next_index]
        

        self.market_state = action
        
        reward = self.calculate_reward(current_index, next_index)
        #self.reward += reward
        
        
        self.index = next_index
        
        return self.features_array[next_index], reward, self.done
        


    def calculate_reward(self, index, next_index):
        
        reward = 0
        #current_close_price = self.get_price(index)[4]
        #change = next_close_price - current_close_price 
        
        
        """
        next_Open_price = self.get_price(next_index)[4]
        next_Open_price = self.get_price(next_index)[1]
        """
        
        close_price = self.get_price(index)[4]
        open_price = self.get_price(index)[1]
        #MA14 = self.features_full[index][4]
        
        SMA3 = self.features_full[index][0]
        SMA6 = self.features_full[index][1]
        SMA14 = self.features_full[index][2]
        SMA25 = self.features_full[index][3]
        SMA50 = self.features_full[index][4]
        
        ANG_Close = self.features_full[index][11]
        
        ANG_MA_6 = self.features_full[index][5]
        ANG_MA_15 = self.features_full[index][6]
        ANG_MA_25 = self.features_full[index][7]
        ANG_MA_50 =self.features_full[index][8]
        
        changes = self.features_full[index][23]
        
        ADX = self.features_full[index][19]
        ADXR = self.features_full[index][22]
        DI = self.features_full[index][20]
        DM = self.features_full[index][21]
        
        
        
        #SL = 1
        #intervalo = 0.5
        

        if self.last_state == 0 and self.market_state == 1:
            self.initial_price_order = close_price
            self.sum_diferences = 0
            self.period_count = 0
            #self.max_price = close_price
            #self.trailing = SMA6 - intervalo
            #self.date_open = self.get_price(index)[0].to_pydatetime()
            if DI > DM and SMA6 > SMA14:
                reward = 0.2
            else:
                reward = -1
            
        
        
        
        if self.market_state == 0 and self.last_state == 1:
            self.sum_diferences += close_price - open_price
            self.period_count += 1
            
            if self.sum_diferences >= 1:
                self.sum_diferences = 0
                self.period_count = 0
                reward = 0.3
            elif self.sum_diferences <= 0:
                self.sum_diferences = 0
                self.period_count = 0
                reward = -1
            else:
                self.sum_diferences = 0
                self.period_count = 0
                reward = -0.2
 
                
        if self.market_state == 1 and self.last_state == 1:
            self.sum_diferences += close_price - open_price
            self.period_count += 1
            reward = round(changes, 2)
            
            if self.period_count == 5:
                self.sum_diferences = 0
                self.period_count = 0


            
            #if DI > DM and SMA6 > SMA50 and self.market_state == 1 and self.last_state == 1:
                
                
            """
            if SMA6 > SMA50:
                if self.period_count == 5:
                    if self.sum_diferences >= 1:
                        self.sum_diferences = 0
                        self.period_count = 0
                        reward = 0.5
                    else:
                        self.sum_diferences = 0
                        self.period_count = 0
                        reward = -1.5
            elif SMA6 < SMA50:
                if self.period_count == 5:
                    if self.sum_diferences >= 1:
                        self.sum_diferences = 0
                        self.period_count = 0
                        reward = 0.5
                    else:
                        self.sum_diferences = 0
                        self.period_count = 0
                        reward = -1.5
            """
        
        

        
                        
        """
        if close_price < self.initial_price_order and self.market_state == 1:
            reward = -0.4
            if close_price < self.trailing:
                reward = -0.75
            if close_price < (self.initial_price_order - SL):
                reward = -1.5
        """
        
        #OPERACIONES 1 VELA
        # REVISAR PORQUE ESTOY FAJANDO UNA SIETUACION DE CIERRE NO POR INDICADORES
        
        #if self.market_state == 0 and self.last_state == 1 and self.date_open != None:
        #    if (self.date_open + datetime.timedelta(minutes=1)) == self.get_price(index)[0].to_pydatetime():
        #        reward = -0.55
        
        
        return reward

    """
    TAKE ACTION
    - Dispara cambio de datos de ordenes
    - Incluido en el step (por ahora)
    """

    """
    
    GET_FEATURE
    - busqueda por step/index
    
    GET_PRICE
    - busqueda por step/index
    
    """

    def get_current_state(self):
        return self.get_feature(self.index), self.reward, self.done

    def get_feature(self, index):
        return self.features_array[index]
    
    def get_price(self, index):
        return self.price_array[index]
    
    def reset(self, initialIndex: int=0):
        self.index = initialIndex
        self.market_state = 0
        self.done = False
        self.initial_price_order = 0
        
        self.last_state = 0
        self.trailing = 0
        self.date_open = None
        self.max_price = 0
        self.period_count = 0
        self.sum_diferences = 0
        
        return self.get_feature(self.index)
    
    


"""

DATA SET
- Carga de dataset de precios
- carga de dataset de features

"""
"""
import numpy as np
from features import feature_engineering
from datetime import datetime

#dailyBars viene de exampleTrendBars
df = pd.DataFrame(np.array(dailyBars),columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

features_list =  feature_engineering(df).dropna()

prices_list = np.array(df.iloc[features_list.index[0]:])

features_new_list = features_list.reset_index(drop=True)
features_list = np.array(features_new_list)
    
env = Environment(observationShape= features_list.shape[1], price_array = prices_list, features_array= features_list)

env.index = 11952
index = env.index
obs_feature = env.get_feature(index)
obs_price = env.get_price(index)

#ambos indicadores alcistas - Vela Alcista
#reward 42/43 action = 0 -> -3.3699 (está bien)
#reward 42/43 action = 1 -> 3.3699 (está bien - vela alcista)
#reward 42/43 action = 2 -> -13.4799 (está bien - ambos indicadores alcistas)

#ambos indicadores alcistas - Vela Bajista
#reward 44/45 action = 0 -> -2.61 (està bien)
#reward 44/45 action = 1 ->  -5.22 (està bien - vela bajista - ambos indicadores alcistas)
#reward 44/45 action = 2 ->  2.61 (està bien - vela bajista)

#1 indicador bajista - 1 indicador alcista - Vela bajista
#reward 89/90 action = 0 -> -3.07 (està bien)
#reward 89/90 action = 1 -> -12.28 (está bien - media alcista/ADX bajista)
#reward 89/90 action = 2 -> 3.07 (està bien - vela bajista)

# indicadores Bajistas - Vela alcista
#reward 97/98 action = 0 -> -1.7399 (està bien)
#reward 97/98 action = 1 -> 1.7399 (està bien)
#reward 97/98 action = 2 -> -3.4799 (està bien)


#obs = env.reset(initialIndex=random.randrange(start=0,stop=100))



action = 0
obs_next_obs, obs_reward, obs_done = env.step(action)

obs = obs_next_obs


test_current_close_price = env.get_price(0)[4]
test_next_close_price = env.get_price(1)[4]
test_change = test_next_close_price - test_current_close_price 


"""


"""AGREGAR EN TRAIN o en una clase aparte Y NO EN ENV"""

"""
ORDENES
- ver cálculo más simple
- estado
- index de inicio
- index de fin
- sumatoria de cambios mientras se mantuvo -> para profit

buffer de ordenes para calculo de profit -> descarga de operaciones
"""
"""

self.ops_buffer = list()
if self.market_state == 0 and (action == 1 or action == 2):
    self.initial_index = current_index
elif (self.market_state == 1 or self.market_state == 2) and action == 0:
    self.finish_index_order = current_index
    save_order_buffer(action)
    def save_order_buffer(self):
        
        if self.market_state == 1:
            
            
        elif self.market_state == 2:


from collections import deque, namedtuple

OrdersBuffer = namedtuple("Experience", ['obs', 'action', 'reward', 'next_obs', 'done'])


class ExperienceDeque(object):
    def __init__(self, capacity = int(1e6)):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def store(self, exp):
        self.memory.append(exp)
        
    def get_size(self):
        return len(self.memory)
"""
















