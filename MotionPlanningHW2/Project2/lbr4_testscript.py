import time
import vrepWrapper
import numpy as np

sh = True
batched = True
normal = True


try:
    import SharedArray as sa
except:
    print('switching to not using shared memory')
    sa = None
    sh = False


simulator= vrepWrapper.vrepWrapper(sh,sa)
#demonstrate collission checking
#simulator.vrepStop()

num = 1000
states = np.pi * np.random.random((num,7))


#benchmark using shared memory
if(sh):
    shm_collision = [0] * num
    start = time.time()
    for i in range(num):
        state = np.reshape(states[i,:],(1,-1))        
        #time.sleep(0.01)
        collide = simulator.test_collisions(state)
        shm_collision[i] = collide
        
    sh_time = (time.time() - start)/num 
    print(sh_time)



#benchmark passing num states at once
if(batched):
    start = time.time()
    which_collides = simulator.checkCollission(states)
    batched_time = (time.time() - start)/num 
    print(batched_time)

  
#benchmark passing num states one at a time
if(normal):
    simulator.setCollision(False)
    tcp_collision = [0] * num
    start = time.time()
    for i in range(num):
        state = np.reshape(states[i,:],(1,-1))        
        tcp_collision[i] = simulator.test_collisions(state)
    normal_time = (time.time() - start)/num 
    print(normal_time)

if(batched and normal):
    print('batched gives you ' + str(round(normal_time/batched_time,2))+ 'x speed improvement')

if(sh and normal):
    print('my shared memory gives you ' + str(round(normal_time/sh_time,2)) + 'x speed improvement')

simulator.vrepReset()

start = time.time()
points = np.random.random((50,3))
simulator.addPoint(points)


#demonstrate drawing edges
lines = np.zeros((1000,6))
for i in range(1000):
    startIndex = np.random.randint(0,49)
    endIndex = np.random.randint(0,49)
    lines[i,:3] = points[startIndex,:]
    lines[i,3:] = points[endIndex,:]


simulator.addLine(lines)
print(time.time() - start )


#
time.sleep(10)
#
simulator.vrepStop()
print("Simulation complete.")
