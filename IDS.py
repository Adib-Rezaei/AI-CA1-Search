import time 


def execution_time_wrapper(func):
    def inner(*args, **kwargs):
        tic = time.time()

        result = func(*args, **kwargs)

        toc = time.time()
        print(f'Result: {result}')
        print(f'Compute time: {round(1000 * (toc - tic), 6)}ms')
        return result

    return inner

class State:
	def __init__(self, doctors, spells=[], pills=[], cost=0, parent=None):
		self.doctors = doctors
		self.spells = spells
		self.pills = pills 
		self.cost = cost
		self.parent = parent 

	def __eq__(self, object):
		if(isinstance(object, State)):
			return self.spells == object.spells and \
				self.pills == object.pills and \
				self.doctors == object.doctors

	def __str__(self):
		return f" doctors: {self.doctors}\n spells: {self.spells}\n pills: {self.pills}\n cost: {self.cost}\n"

	def __hash__(self) -> int:
		return hash(str(self))


def get_pair_input():
    a, b = map(int, input().strip().split())
    return a, b

n, m = get_pair_input()
c, k = get_pair_input()
grid = [[0 for i in range(m)] for j in range(n)]
row = n-1
for i in range(c):
    x, y = get_pair_input()
    grid[row-x][y] = 1      # add spells

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
    cost=prev_state.cost+1, parent=prev_state)

    # update doctor location
    state.doctors[j] = (x, y)

    apply_event(state, grid, x, y)
    
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


def DLS(grid, state ,maxDepth, explored):

    if goal_test(state):
        global goal_state 
        goal_state = state
        return True

    explored.add(state)
    if maxDepth <= 0 : return False

    doctors = state.doctors
    for j in range(len(doctors)):
        neighbors = get_neighbors(doctors, j)
        for adjx, adjy in neighbors:
            if (is_position_valid(adjx, adjy)):
                child_state = create_child_state(state, adjx, adjy, j)
                if(child_state not in explored):
                    if(DLS(grid, child_state ,maxDepth-1, explored)):
                        return True
    return False

@execution_time_wrapper
def IDS(grid, state, maxDepth):
    states_explored_sum = 0
    explored = set()
    apply_event(state, grid, n-1, 0)
    for depth in range(maxDepth):
        if (DLS(grid, state, depth, explored)):
            states_explored_sum += len(explored)
            return states_explored_sum
        explored = set()
    return states_explored_sum


start_state = State(doctors=[(n-1, 0)])
explored = set()
states_explored_sum = IDS(grid, start_state, n*m)
# print(len(explored))
# for e in explored:
#     print(e)
def print_path(goal_state):
    print("path from source to goal: \n")
    while goal_state:
        print(goal_state)
        goal_state = goal_state.parent

print_path(goal_state)
print("explored number: ", states_explored_sum)

