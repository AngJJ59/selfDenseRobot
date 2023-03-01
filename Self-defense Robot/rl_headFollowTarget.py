#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:52:43 2022

@author: benderang
"""

import sys
import gym


class BaxterHeadMonitorModel():
      def __init__(self, name='selfdefenseBaxter'):
          super(self.__class__, self).__init__()
          self.name = name
          self.client_ID = None
          
          
          self.monitorJointHandle = None
      
      def initializeSimModel(self, client_ID):
          print('h')
          