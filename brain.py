# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:34:11 2023

@author: Langley Juan Manuel
"""

"""
:param input_shape: Tamaño o forma de los datos de entrada
:param output_shape: Tamaño o forma de los datos de salida
:param device: El dispositivo ('cpu' o 'cuda') que la SLP debe utilizar para almacenar los inputs a cada iteración
"""

import torch

class Brain(torch.nn.Module):
    def __init__(self, state_dim, action_dim, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        super(Brain, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        """
        self.linear1 = torch.nn.Linear(self.state_dim, 256)
        self.linear2 = torch.nn.Linear(256, 512)
        self.out = torch.nn.Linear(512, self.action_dim)
        """
        self.device = device
        
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.action_dim)
            )
           

    def forward(self, x):
        x = self.model(x)
        """
        #x = torch.from_numpy(x).float().to(self.device)
        x = torch.nn.functional.relu(self.linear1(x)) ##Función de activación RELU
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.out(x)
        """
        return x

# model.train()

    
