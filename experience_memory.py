#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: juan
"""

from collections import namedtuple
import random
from collections import deque

Experience = namedtuple("Experience", ['obs', 'action', 'reward', 'next_obs', 'done'])


class ExperienceDeque(object):
    def __init__(self, capacity = int(1e6)):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def store(self, exp):
        self.memory.append(exp)
        
    def get_size(self):
        return len(self.memory)
    
    def continuous_sample(self, batch_size):
          batch = []
          if batch_size > len(self.memory):
              batch_size = len(self.memory)
              
          # como el deque siempre almacena las ultimas observaciones al final, para llenar el batch
          # recorremos esas observaciones y las agregamos al batch
          for i in range(len(self.memory) - batch_size, len(self.memory)):
              batch.append(self.memory[i]) 
          return batch
      
    def random_batch(self, batch_size):
        assert batch_size <= self.get_size(), "El tamaño de la muestra es superior a la memoria disponible"
        return random.sample(self.memory, batch_size)
 
    
"""
memoria = ExperienceMemory(capacity=100)
for i in range(0, 105):
    memoria.store(i)

memoria.get_size()
lista = memoria.memory

memoria2 = ExperienceDeque(capacity=100)
for i in range(0, 50):
    memoria2.store(i)

memoria2.store(999)
lista = memoria2.random_sample(5)
memoria2.get_size()

example = memoria2.random_sample(10)
"""