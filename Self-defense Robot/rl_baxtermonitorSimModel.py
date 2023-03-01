#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:52:43 2022

@author: benderang
"""



import sim 


class BaxterHeadMonitorModel():
      def __init__(self, name='selfdefenseBaxter'):
          super(self.__class__, self).__init__()
          self.name = name
          self.client_ID = None
          
          
          self.monitorJointHandle = None
      
      def initializeSimModel(self, client_ID):
          try:
             print ('Connected to remote API server')
             client_ID != -1
          except:
                 print ('Failed connecting to remote API server')

          self.client_ID = client_ID
          
          return_code, self.monitor_joint_handle = sim.simxGetObjectHandle(client_ID, 'monitorJointHandle', sim.simx_opmode_blocking)
          if return_code == sim.simx_return_ok:
              print('get minitor joint handle ok')
              
              
          # Get the joint position
          return_code, q = sim.simxGetJointPosition(self.client_ID, self.monitor_joint_handle, sim.simx_opmode_streaming)
          
          
          # Set the initialized position for each joint
          self.setJointTorque(0)
          
          
      def getJointPosition(self, joint_name):
  
           q = 0
           if joint_name == 'monitorJointHandle':
              return_code, q = sim.simxGetJointPosition(self.client_ID, self.monitor_joint_handle, sim.simx_opmode_buffer)
           else:
               print('Error: joint name: \' ' + joint_name + '\' can not be recognized.')
      
           return q
         
            
      def setJointTorque(self, torque):
            if torque >= 0:
                sim.simxSetJointTargetVelocity(self.client_ID, self.monitor_joint_handle, 1000, sim.simx_opmode_oneshot)
            else:
                sim.simxSetJointTargetVelocity(self.client_ID, self.monitor_joint_handle, -1000, sim.simx_opmode_oneshot)
    
            sim.simxSetJointMaxForce(self.client_ID, self.monitor_joint_handle, abs(torque), sim.simx_opmode_oneshot)
          
          
          
              
              
              
          
                          
        
          