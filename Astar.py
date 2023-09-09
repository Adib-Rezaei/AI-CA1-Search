import time 
import heapq

def execution_time_wrapper(func):
	def inner(*args, **kwargs):
		tic = time.time()

		result, a = func(*args, **kwargs)

		toc = time.time()
		print(f'Result: {result}')
		print(f'Compute time: {round(1000 * (toc - tic), 6)}ms')
		return result, a

	return inner

class State:
	def __init__(self, doctors, spells=[], pills=[], f=0, g=0, h=0, parent=None):
		self.doctors = doctors
		self.spells = spells
		self.pills = pills 
		self.f = f 
		self.g = g 
		self.h = h
		self.parent = parent 

	def __eq__(self, object):
		if(isinstance(object, State)):
			return self.spells == object.spells and \
				self.pills == object.pills and \
				self.doctors == object.doctors

	def __str__(self):
		return f" doctors: {self.doctors}\n spells: {self.spells}\n pills: {self.pills}\n f: {self.f} g: {self.g} h: {self.h}\n"

	def __hash__(self) -> int:
		return hash(f" doctors: {self.doctors}\n spells: {self.spells}\n pills: {self.pills}\n")

	def __lt__(self, obj):
		return self.f < obj.f


def get_pair_input():
	a, b = map(int, input().strip().split())
	return a, b

n, m = get_pair_input()
c, k = get_pair_input()
grid = [[0 for i in range(m)] for j in range(n)]
row = n-1
all_spells = []
for i in range(c):
	x, y = get_pair_input()
	grid[row-x][y] = 1      # add spells
	all_spells.append((row-x, y))

for i in range(k):
	x, y = get_pair_input()
	grid[row-x][y] = 2      # add pills

d = int(input())
for i in range(d):
	x, y = get_pair_input()
	grid[row-x][y] = 3      # add walls

import numpy as np 
print(np.matrix(grid))
print()



x_dir = [ -1, 0, 1, 0]
y_dir = [ 0, 1, 0, -1]

def manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def heuristic(state):
	doctors = state.doctors
	spells_to_explore = [spell for spell in all_spells if (spell not in state.spells)]
	sum = 0
	for doctor in doctors:
		mini = manhattan(*doctor, *(0, m-1))
		for spell in spells_to_explore:
			mini = min(mini, manhattan(*doctor, *spell))
		sum += mini 

	return sum 

def is_position_valid(x, y):
	if (x < 0 or y < 0 or x >= n or y >= m or grid[x][y] == 3):
		return False

	return True

def apply_event(state, grid, x, y):
	if(grid[x][y] == 1):
		if not (x, y) in state.spells:
			state.spells.append((x, y))

	elif(grid[x][y] == 2):
		if not (x,y) in state.pills:
			state.pills.append((x, y))
			state.doctors.append((0, 0))

def create_child_state(prev_state, x, y, j):

	# create new state
	state = State(doctors=prev_state.doctors[:],
	spells=prev_state.spells[:], pills=prev_state.pills[:],
	parent=prev_state, g=prev_state.g+1)

	# update doctor location
	state.doctors[j] = (x, y)

	apply_event(state, grid, x, y)
	
	state.h = heuristic(state)
	state.f = state.h + state.g
	
	return state

def goal_test(state):
	if len(state.spells) != c:
		return False 

	for doctor in state.doctors:
		if doctor != (0, m-1):
			return False

	return True 
	
def get_neighbors(doctors, j):
	neighbors = []
	for i in range(4):
		adjx = doctors[j][0] + x_dir[i]
		adjy = doctors[j][1] + y_dir[i]
		neighbors.append((adjx, adjy))
	return neighbors


@execution_time_wrapper
def astar(grid, start_state):

	openset = []
	closedset = set()
	heapq.heappush(openset, start_state)
	closedset.add(start_state)

	apply_event(start_state, grid, n-1, 0)

	while len(openset) > 0:
		state = heapq.heappop(openset)
		if goal_test(state):
			return state, len(closedset)

		doctors = state.doctors
		for j in range(len(doctors)):
			neighbors = get_neighbors(doctors, j)
			for adjx, adjy in neighbors:
				if (is_position_valid(adjx, adjy)):
					child_state = create_child_state(state, adjx, adjy, j)
					if child_state not in closedset:
						heapq.heappush(openset, child_state)
						closedset.add(child_state)


start_state = State(doctors=[(n-1, 0)])
start_state.f = start_state.h = heuristic(start_state)
goal_state, explored_states = astar(grid, start_state)
print("explored: ", explored_states)
print(goal_state)
def print_path(goal_state):
    print("path from source to goal: \n")
    while goal_state:
        print(goal_state)
        goal_state = goal_state.parent

# print_path(goal_state)
# print("explored number: ", closedset)