# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:32:58 2023

@author: Solido
"""



import torch
import numpy as np
from brain import Brain
from decay_schedule import LinearDecaySchedule
from experience_memory import ExperienceDeque, Experience
import random


class DQLearner(object):
    def __init__(self, 
                 environment, 
                 learningRate = 0.005, 
                 gamma = 0.98,
                 maxMemory = 20000,
                 STEPS_PER_EPISODE= int,
                 MAX_EPISODES= int):
        
        self.obs_shape = environment.observation_shape
        self.action_shape = environment.action_shape
        self.learning_rate = learningRate
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Q = Brain(self.obs_shape, self.action_shape).to(self.device)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = self.learning_rate)
        self.best_mean_reward = -float("inf")
        self.best_reward = -float("inf")
        
        #Factor de descuento
        self.gamma = gamma
        # ALPHA => ratio de aprenizaje del agente (learning_rate)
        self.memory = ExperienceDeque(maxMemory)
        
        self.step_per_episode = STEPS_PER_EPISODE
        """Variables para epsilon greedy"""
        # para exploración/explotación (desuso por el softmax)
        self.epsilon_max = 1.0
        self.epsilon_min = 0.1
        self.train_epsilon = 0.9
        self.epsilon_decay = LinearDecaySchedule(initial_value= self.epsilon_max,
                                                 final_value= self.epsilon_min,
                                                 max_steps= self.train_epsilon*MAX_EPISODES*self.step_per_episode)
        self.step_num = 0
        self.policy = self.epsilon_greedy_Q
        self.epsilon_dec = 0
    
    
    """Ejemplo de epsilon Greedy"""
    def get_action_epsilon(self, obs):
        return self.policy(obs)
    
    def epsilon_greedy_Q(self, obs):
        #torch_obs = torch.Tensor(obs.reshape(1, -1))
        self.epsilon_dec = self.epsilon_decay(self.step_num)
        if random.random() < self.epsilon_dec:
            action = random.choice([a for a in range(self.action_shape)])
        else:
            #q_values = self.Q(torch.Tensor(obs).to(self.device))
            
            q_values = self.Q(torch.Tensor(obs).to(self.device))
            #action_probs = q_values.cpu().detach().numpy().squeeze()
            action_probs = q_values.cpu().detach().numpy()
            #print(str(action_probs) + str(np.argmax(action_probs)))
            action = np.argmax(action_probs)
        self.step_num+=1
        return action


    """Trabajo con SOFTMAX"""
    # la obs puede venir en formato lista y es necesario convertirlo a un array fila par convertirlo en tensor
    def get_action_softmax(self, obs, explore: bool=True): 
        # torch_obs = torch.Tensor(obs.reshape(1, -1))
        #q_values = self.Q(torch.Tensor(obs).to(self.device))
        
        q_values = torch.nn.functional.softmax(self.Q(torch.Tensor(obs).to(self.device)), dim=0)
        action_probs = q_values.cpu().detach().numpy().squeeze()
        #print(action_probs)
        if explore:
            # Seleccionar la acción según las probabilidades del softmax
            selected_action = np.random.choice(range(self.action_shape), p=action_probs)
        else:
            selected_action = np.argmax(action_probs)
        return selected_action
    
        
    def replay_experience(self, batch_size):
        experience_batch = self.memory.random_batch(batch_size)
        #experience_batch = self.memory.continuous_sample(batch_size)
        error = self.learn_from_batch_experience(experience_batch)
        return error.cpu().detach().numpy()
        
    def learn_from_batch_experience(self, experiences):
        """
        Parameters
        ----------
        experiences : TYPE
            DESCRIPTION.
            Fragmento de recuerdos anteriores

        Returns
        -------
        None.
        
        info = https://cursos.frogamesformacion.com/courses/take/ia-python/lessons/34044429-aprender-de-la-experiencia-previa
        """
        batch_exp = Experience(*zip(*experiences))
        obs_batch = torch.Tensor(np.array(batch_exp.obs)).to(self.device)
        action_batch = torch.Tensor(np.array(batch_exp.action)).to(self.device)
        reward_batch = torch.Tensor(np.array(batch_exp.reward)).to(self.device)
        next_obs_batch = torch.Tensor(np.array(batch_exp.next_obs)).to(self.device)
        done_batch = torch.tensor(np.array(batch_exp.done),dtype=bool).to(self.device)
        
        nptile = np.tile(self.gamma, len(next_obs_batch))
        npTileTensor = torch.Tensor(nptile).to(self.device)
        
        #print(next_obs_batch.shape)
        paraPrint = self.Q(next_obs_batch).to(self.device)
        #print(paraPrint)
        #print(paraPrint.shape)
        predictNextObs = paraPrint.max(dim=1)[0].to(self.device)
        

        
        """
        print(predictNextObs)
        print(reward_batch.shape)
        print(done_batch.shape)
        print(npTileTensor.shape)
        """

        td_target = reward_batch + ~done_batch * \
                 npTileTensor * \
                 predictNextObs

             
        td_target = td_target.to(self.device)
        td_target = td_target.float().unsqueeze(1).to(self.device)

        tensor_obs = obs_batch.squeeze(dim=1).to(self.device)
        
        # REVISAR ENTROPIA CRUZADA EN TORCH

        td_error =  torch.nn.functional.mse_loss(
                self.Q(tensor_obs).gather(1,action_batch.view(-1,1).long()),
                td_target).to(self.device)
        
        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        self.Q_optimizer.step()
        
        #td_error.item() da el valor de error
        """
        td_error.mean().backward()
        self.Q_optimizer.step()
        self.Q_optimizer.zero_grad()
        """
        
        return td_error.mean()
        
    def memory_store(self, obs, action, reward, next_obs, done):
        self.memory.store(Experience(obs, action, reward, next_obs, done))
        
        
        
    def save(self):
        #model_save_name = 'model.pt'
        #path = F"/content/drive/My Drive/{model_save_name}" 
        file_name = "models/DQN_model.ptm"
        agent_state = {"Q": self.Q.state_dict(),
                       "best_mean_reward": self.best_mean_reward,
                       "best_reward": self.best_reward}
        torch.save(agent_state, file_name)
        print("Estado del agente guardado en : ", file_name)
        
        
    def load(self):
        #path = F"/content/drive/My Drive/trained_models/model.pt"
        file_name = "models/DQN_model.ptm"
        agent_state = torch.load(file_name, map_location = lambda storage, loc: storage)
        self.Q.load_state_dict(agent_state["Q"])
        self.Q.to(self.device)
        self.best_mean_reward = agent_state["best_mean_reward"]
        self.best_reward = agent_state["best_reward"]
        print("Cargado del modelo Q desde", file_name,
              "que hasta el momento tiene una mejor recompensa media de: ",self.best_mean_reward,
              " y una recompensa máxima de: ", self.best_reward)





"""
from environment import Environment, env_states
env = Environment()


env = Environment(observationShape= features_list.shape[1] ,price_array = prices_list, features_array= features_list)

MAX_NUM_EPISODES = 10
agent = DQLearner(environment=env, maxMemory=82000 ,MAX_EPISODES=MAX_NUM_EPISODES, STEPS_PER_EPISODE=len(env.features_array))

done = False
obs = env.reset()
total_reward = 0.0
step = 0


obs = env.get_current_state()[0]
#usaremos el de softmax
actionEpsilon = agent.get_action_epsilon(obs)


actionSoftmax = agent.get_action_softmax(obs, True)


q_values = torch.nn.functional.softmax(agent.Q(torch.Tensor(obs).to(agent.device)), dim=0)
action_probs = q_values.cpu().detach().numpy().squeeze()


selected_action = np.random.choice(range(agent.action_shape), p=action_probs)


q_values = agent.Q(torch.Tensor(obs).to(agent.device))
action_probs = q_values.detach().numpy()
action = np.argmax(action_probs)







next_obs, reward, done = env.step(actionSoftmax)

torch_obs = torch.Tensor(obs).to(agent.device)
torch_next_obs = torch.Tensor(next_obs).to(agent.device)

asd = agent.Q(torch_next_obs).cpu().detach().numpy()
#formula de bellman
# R + gamma * max Q(s',a') 
td_target = reward + agent.gamma * torch.max(agent.Q(torch_next_obs))
#q_actual = self.Q(obs)

# (R + gamma * max Q(s',a')) - Qt-1(s,a)
#td_error = td_target - q_actual
qdetach = np.array(agent.Q(torch_obs).cpu().detach().numpy().flatten()[actionSoftmax])
qdata = torch.from_numpy(qdetach).float().to(agent.device)
td_error = torch.nn.functional.mse_loss(qdata, td_target)
        
obs = next_obs
agent.Q_optimizer.zero_grad()
td_error.backward()
agent.Q_optimizer.step()










q_values = agent.Q(torch.Tensor(obs).to(agent.device))
action_probs = q_values.cpu().detach().numpy().squeeze()
sigmoid = torch.nn.functional.sigmoid( q_values)
action_sigmoid = sigmoid.cpu().detach().numpy().squeeze()
softmax = torch.nn.functional.softmax(q_values, dim=1)
action_softmax = softmax.cpu().detach().numpy().squeeze()

soft_sigmoid = torch.nn.functional.softmax(sigmoid, dim=1)
action_softmax_sigmoid = soft_sigmoid.cpu().detach().numpy().squeeze()

q_values_nextObs = agent.Q(torch.Tensor(next_obs).to(agent.device))
action_probs_nextObs = q_values.cpu().detach().numpy().squeeze()

# Seleccionar la acción según las probabilidades del softmax
selected_action = np.random.choice(range(4), p=action_probs)

selected_action = np.argmax(action_probs)



next_obs, reward, done = env.step(actionSoftmax)





agent.learn(obs, action, reward, next_obs)

agent.memory_store(obs, action, reward, next_obs, done)

memoria = agent.memory
memoria.get_size()

agent.Q.state_dim

Q = agent.Q(torch.Tensor(next_obs).to(agent.device))
qdetach = np.array(Q.cpu().detach().numpy())
Qmax = torch.max(Q).to(agent.device)
td_target = torch.Tensor(np.array(reward)).to(agent.device) + torch.Tensor(np.array(0.98)).to(agent.device) * Qmax


td_target2 = reward + 0.98 * float(torch.max(Q).cpu().detach().numpy())


qdata = torch.from_numpy(qdetach).float()

tensor2 = torch.from_numpy(obs).float()

action_probs = tensor1.detach().numpy().flatten()

softmax = torch.nn.functional.softmax(tensor2, dim=1)

softmaxResult = softmax.detach().numpy().flatten()


td_target = 100 + 0.98 * torch.max(softmax)


torchmax = torch.max(softmax)
nwtor = softmax[0,2]
torchmaxarray = torchmax.detach().numpy().flatten()

# Seleccionar la acción según las probabilidades del softmax
selected_action = np.random.choice(range(5), p=softmaxResult)

selected_action2 = np.argmax(softmaxResult)
maxim = np.max(softmaxResult)
"""
