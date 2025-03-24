function collision_setup()
    print('collision etup')
    jointHandles={-1,-1,-1,-1,-1,-1,-1}
    for i=1,7,1 do
        jointHandles[i]=sim.getObjectHandle('LBR4p_joint'..i)
    end
    linkHandles = {-1,-1,-1,-1,-1,-1,-1,-1} 
    for i=1,7,1 do
        name = 'LBR4p_link'..i+1
        linkHandles[i] = sim.getObjectHandle(name)
    end
    linkHandles[8] = sim.getObjectHandle('BarrettHand')
    table = sim.getObjectHandle('diningTable')
    floor = sim.getObjectHandle('Floor_element')

end
function shm_collision(path)
    print('trying new')
    py = require 'python'
    print('required python')
    py.execute('import os;cwd = os.getcwd();print(cwd)')
    print('got cwd')
    chdir = py.import "os".chdir
    chdir(path) 
    print('changed cmd')
    py.execute('import os;cwd = os.getcwd();print(cwd)')
    test = py.import "sum".test
    test()
    print('tested')

    shared_setup = py.import "sum".setup
    success = shared_setup()
    if(success == 0)then
        print('shm setup failed')
        py = nil
        package.loaded["python"] = nil
        while true do 
            sim.switchThread()
        end
    end
    print('setup shared')
    getJoints = py.import "sum".getJointPos
    sendData = py.import "sum".sendCollisionData
    print('import everything')
    count = 0
    while true do
        data = getJoints()
        if(data[0] >0.5) then
            count = count +1
            --print(count)
            --print('start check')
            collisions,fk = collision_run({1},data,'','')
            --print('finished check')
            sendData(fk,collisions)
            --print(collisions[1])
        end
    end
end
function collision_run(inInts,angles,dummy,dummy1)
    num_states = inInts[1]
    jointPositions={-1,-1,-1,-1,-1,-1,-1}
    for i=1,7,1 do
        jointPositions[i] = sim.getJointPosition(jointHandles[i])
    end

    collisions = {}
    fk = {}
    for j=0,num_states-1,1 do
        
        for i=1,7,1 do
            sim.setJointPosition(jointHandles[i],angles[i + j*7])
        end
        collides = 0
        index = 8
        while collides == 0 and index>0 do
            collides = sim.checkCollision(linkHandles[index],table) 
            index = index -1
        end
        index = 8
        while collides == 0 and index>0 do
            collides = sim.checkCollision(linkHandles[index],floor) 
            index = index -1
        end
        pose = sim.getObjectPosition(linkHandles[8],-1)
        fk[j*3 + 1] = pose[1]
        fk[j*3 + 2] = pose[2]
        fk[j*3 + 3] = pose[3]

        collisions[j+1] = collides
    end

    for i=1,7,1 do
        sim.setJointPosition(jointHandles[i],jointPositions[i])
    end
    return collisions,fk,{},''
end
