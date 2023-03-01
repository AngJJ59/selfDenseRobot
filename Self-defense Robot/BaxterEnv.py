#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:42:01 2022

@author: benderang
"""

import numpy as np
import gym
from gym.utils import seeding
from gym import spaces, logger
import time


from rl_baxtermonitorSimModel import BaxterHeadMonitorModel

class BaxterEnv(gym.Env):
    
      metadata = {'render.modes': ['human']}
    
      def __init__(self, action_type='descrete'):
           super(BaxterEnv, self).__init__()
           self.action_type = action_type
           self.push_force = 0
           self.q = [0.0, 0.0]
           self.q_last = [0.0, 0.0]
      
           self.theta_max = 40*np.pi / 360
           self.cart_pos_max = 0.8