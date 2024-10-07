


from environment import Environment
from DQN import DQLearner
import numpy as np

from dataSetLoader import load_dataset_sql, state_3d_creator
from features import feature_engineering
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import torch

start_date = '2022-05-01 00:00:00'
end_date = '2022-05-20 00:00:00'
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
    print("Error al cargar el archivo scaler, se genera uno nuevo")


if sc_load:
    features_list = sc.transform(features_list)
else:
    features_list = sc.fit_transform(features_list)
    joblib.dump(sc, 'standar_scaler/scaler.pkl')
"""



#scalers = [MinMaxScaler() for _ in range(features_list.shape[1])]


minmax_load = False
scalers = []

try:
    for i in range(features_list.shape[1]):
        scaler = joblib.load(f'minmax_scalers/scaler_columna_{i}.pkl')
        scalers.append(scaler)
        minmax_load = True
except FileNotFoundError:
    print("Error al cargar el archivo scaler, se genera uno nuevo")

if minmax_load:
    features_list = np.hstack([scaler.fit_transform(features_list[:, i].reshape(-1, 1)) for i, scaler in enumerate(scalers)])
else:

    features_scalers = [MinMaxScaler() for _ in range(features_list.shape[1])]
    features_concat = []
    for i, scaler in enumerate(features_scalers):
        columna_escalada = scaler.fit_transform(features_list[:, i].reshape(-1, 1))
        features_concat.append(columna_escalada)
        joblib.dump(scaler, f'minmax_scalers/scaler_columna_{i}.pkl')
    features_list = np.hstack(features_concat)



#features_standar_full = features_full[:,6]
#features_standar_full_transform = np.reshape(features_standar_full, (-1,1))
#sc_maxmin = MinMaxScaler()
#features_minmax = sc_maxmin.fit_transform(features_standar_full_transform)





env = Environment(observationShape= features_list.shape[1] ,price_array = prices_list, features_array= features_list, features_full=features_full)

episode_rewards = list()

MAX_NUM_EPISODES = 5000
agent = DQLearner(environment=env, maxMemory=1010 ,MAX_EPISODES=MAX_NUM_EPISODES, STEPS_PER_EPISODE=len(env.features_array))

#agent = DQLearner(learningRate=0.05, environment=env, maxMemory=13500 ,MAX_EPISODES=MAX_NUM_EPISODES, STEPS_PER_EPISODE=len(env.features_array))
     
summary_filename_prefix = "logs/DQN_"
summary_filename = summary_filename_prefix + datetime.now().strftime("%y-%m-%d-%H-%M")
writer = SummaryWriter(summary_filename)

#writer.add_scalar(self.actor_name + "/critic_loss", critic_loss, self.global_step_num)

try:
    agent.load()
except FileNotFoundError:
    print("ERROR: No existe ningún modelo entrenado para este entorno. Empieza desde 0")

if torch.cuda.is_available() != True:
    print("ERROR, CUDA INACTIVO")
mean_error = 0

for episode in range(MAX_NUM_EPISODES):
    done = False
    obs = env.reset(initialIndex=random.randrange(start=0,stop=100))
    total_reward = 0.0
    step = 0
    previous_mean_ep_rew = agent.best_mean_reward
    checkpoint_to_save = False
    venta = 0
    compra = 0
    fuera = 0

    
    while not done:
        #acá empezarían los steps


        action = agent.get_action_epsilon(obs)
        #action = agent.get_action_softmax(obs, True)
        
        next_obs, reward, done = env.step(action)
        agent.memory_store(obs, action, reward, next_obs, done)
        
        if action == 0:
            fuera += 1
        elif action == 1:
            compra += 1
        elif action == 2:
            venta += 1

        obs = next_obs
        total_reward += reward 

        step += 1
        
        
        
        if step % 1000 == 0 and agent.memory.get_size() > 1000:
           mean_error = agent.replay_experience(1000)
        


        if done is True:
            episode_rewards.append(total_reward)
            if total_reward > agent.best_reward:
                agent.best_reward = total_reward
                
        
            if np.mean(episode_rewards) > agent.best_mean_reward:
                if episode > 400:
                    agent.best_mean_reward = np.mean(episode_rewards)
                    previous_mean_ep_rew = np.mean(episode_rewards)
                    checkpoint_to_save = True
            else:
                checkpoint_to_save = False

                
                
                
            print("\nEpisodio#{} finalizado con {} iteraciones, Recompensa = {}, Mejor recompensa = {}, Recompensa media = {}, Mejor recompensa media = {}".
                  format(episode, step+1, round(total_reward, 2), round(agent.best_reward, 2), round(np.mean(episode_rewards), 2), round(agent.best_mean_reward, 2)))
            

            
            """
            print("\nEpisodio#{} finalizado con {} iteraciones. Balance Final = {}, Recompensa = {}, Recompensa media = {}, Mejor recompensa = {}".
                  format(episode, step+1, round(env.balance, 2) ,round(total_reward, 2), round(np.mean(episode_rewards), 2), round(max_reward, 2)))
            """
            
            # vamos a hacer experiencias en función del tamaño de la capacidad
            # en este caso, despues de haber acumulado más de 10.000 va a usar 3200 esperiencias
            """Este rejugado hay varias formas de hacerlo, ver otras opciones como el de tenwordfow2"""
            """
            if agent.memory.get_size()>=13000: 
                mean_error = agent.replay_experience(13000)

                if checkpoint_to_save:
                    agent.save()
                    
                writer.add_scalar("fuera", fuera, episode)
                writer.add_scalar("compra", compra, episode)
                writer.add_scalar("venta", venta, episode)
                
                
                mean_error = float(mean_error)

                writer.add_scalar("total_reward", total_reward, episode)
                writer.add_scalar("max_reward", agent.best_reward, episode)
                writer.add_scalar("epsilon", agent.epsilon_dec, episode)
                writer.add_scalar("mean_reward", round(np.mean(episode_rewards), 2), episode)
                writer.add_scalar("mean_best_reward", round(agent.best_mean_reward, 2), episode)
                writer.add_scalar("mean_error", round(mean_error, 4), episode)
            break
            """
            if checkpoint_to_save:
                agent.save()
                    
            writer.add_scalar("fuera", fuera, episode)
            writer.add_scalar("compra", compra, episode)
            writer.add_scalar("venta", venta, episode)
            
            
            mean_error = float(mean_error)

            writer.add_scalar("total_reward", total_reward, episode)
            writer.add_scalar("max_reward", agent.best_reward, episode)
            writer.add_scalar("epsilon", agent.epsilon_dec, episode)
            writer.add_scalar("mean_reward", round(np.mean(episode_rewards), 2), episode)
            writer.add_scalar("mean_best_reward", round(agent.best_mean_reward, 2), episode)
            writer.add_scalar("mean_error", round(mean_error, 4), episode)


"""
env_features = env.features_array
env_prices = env.price_array

env_price = env.get_price(0)[4]

print(env_prices[0][4])
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
            
"""