function draw_setup()
	print('draw setup')
	red = {1,0,0}
	green = {0,1,0}
	blue = {0,0,1}
	plan = sim.addDrawingObject(sim.drawing_lines,8,0,-1,1000000,green)
	points = sim.addDrawingObject(sim.drawing_points,5,0,-1,1000000,blue)
	edges = sim.addDrawingObject(sim.drawing_lines,1,0,-1,1000000,blue)
	importantPoints = sim.addDrawingObject(sim.drawing_points,10,0,-1,1000000,red)
end

function addPoints(inInts,pointData,dummy,dummy1)
	num_states = inInts[1]
	print(num_states)
	if inInts[2] == 0 then
		targetPoints = points
	else 
		targetPoints = importantPoints
	end
    for j=0,num_states-1,1 do
    	data = {pointData[1 + 3*j],pointData[2 + 3*j],pointData[3 + 3*j]}
		sim.addDrawingObjectItem(targetPoints,data)
	end
	return {}, {}, {}, ''
end

function addLines(inInts,pointData,dummy,dummy1)
	num_states = inInts[1]
	print(num_states)
	if inInts[2] == 0 then
		targetLines = edges
	else 
		targetLines = plan
	end
    for j=0,num_states-1,1 do
    	data = {	
    	pointData[1 + 6*j],pointData[2 + 6*j],
    	pointData[3 + 6*j],pointData[4 + 6*j],
   		pointData[5 + 6*j],pointData[6 + 6*j]}

		sim.addDrawingObjectItem(targetLines,data)
	end
	return {}, {}, {}, ''
end