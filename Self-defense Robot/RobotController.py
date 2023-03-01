import sim as vrep
import sys

vrep.simxFinish(-1)

clientID=vrep.simxStart("127.0.0.1",19999,True,True,5000,5) 







if clientID != -1:
    print("hello")
    print("Connected to the remote API server")
else:
    print("Connetion failed")
    sys.exit("Could not connect")




