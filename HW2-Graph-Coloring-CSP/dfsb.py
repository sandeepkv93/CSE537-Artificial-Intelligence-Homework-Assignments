import sys
import time

'''
Parses the input file to get N, M, K and adjacency list i.e constraint list
'''
def parse_input_file(input_file):
	with open(input_file,'r') as inp_fi:
		lines = inp_fi.read().splitlines()
	N,M,K = map(int,lines[0].split('\t'))
	constraint_lines = lines[1:]
	return N,M,K,constraint_lines

'''
Generates the Constraint satisfaction problem by parsing and processing the inputs given
'''
def generate_constraint_satisfaction_problem(input_file):
    N,M,K,constraint_lines = parse_input_file(input_file)
    constraints = [[] for x in range(N)]
    domains = [list(range(K)) for a in range(N)]
    for line in constraint_lines:
    	constraint = list(map(int,line.split('\t')))
    	constraints[constraint[0]].append(constraint[1])
    	constraints[constraint[1]].append(constraint[0])
    arcs = [[i,constraint] for i in range(N) for constraint in constraints[i]]
    return N, M, K, domains, constraints, arcs

'''
Prints the solution to the output file.
'''
def print_solution_to_output_file(output_file, solution):
	with open(output_file,'w') as out_fi:
		out_fi.write('No Answer') if solution == 'failure' else out_fi.write('\n'.join([str(a) for a in solution]))

'''
Check if solution is complete by checking if all positions are assigned
'''
def is_solution_complete(assignment):
	return all(a != None for a in assignment)

'''
Check if the assignments are consistent and valid
'''
def is_assignment_valid(N, assignment, constraints):
	return not any(assignment[var] == assignment[constraint] and assignment[constraint] is not None for var in range(N) for constraint in constraints[var])


'''
Arc from X->Y is
consistent only when for every value x of in domain of X there is some allowed y
from domain of Y. This algorithm removes the inconsistent values from domain
of all arcs. If domain X loses a value, neighbors of domain X need to be rechecked
because its incoming arcs may become inconsistent.
'''
def arc_consistency_heuristics(arcs, constraints):
	global prunes, domains
	while arcs:
		arc = arcs.pop(0)
		removed = False
		cond_satisfied = False
		for h_color in domains[arc[0]]:
			cond_satisfied = all(v_color == h_color for v_color in domains[arc[1]])
			if cond_satisfied:
				domains[arc[0]].remove(h_color)
				removed = True
				prunes += 1
			cond_satisfied = False
		if removed == True:
			for neighbor in constraints[arc[0]]:
				if neighbor != arc[1]:
					arcs.append([neighbor, arc[0]])

'''
This algorithm chooses a state with least number of legal values left
Find illegal moves at each position and return the position with least legal moves possible
'''

def minimum_remaining_values_heuristics(K, assignment, constraints):
	global domains
	current_min_rem_val = K
	unassigned_position = 0
	position = 0
	while position < len(assignment):
		if assignment[position] is None:
			illegal_values = {*()}
			for var in constraints[position]:
				if assignment[var] is not None:
					illegal_values.add(assignment[var])
					if int(assignment[var]) in domains[position]:
						domains[position].remove(int(assignment[var]))
			num_illegal_values = len(illegal_values)
			total_remaining = K - num_illegal_values
			if total_remaining == 0: return -1
			elif total_remaining <= current_min_rem_val:
				current_min_rem_val = total_remaining
				unassigned_position = position
		position += 1
	return unassigned_position

'''
Find the positions of colors 
such that placing those color at given position 
affects the neighboring elements the least.
Returns list of positions of colors to assign for improved dfsb

'''

def least_constraining_value_heuristics(K, assignment, constraints, position):
	illegal_values = set()
	if constraints[position] is None:
		return list(range(K))
	colors = K * [0]
	for color in range(K):
		illegal_values.add(color)
		for neighbor in constraints[position]:
			if assignment[neighbor] is None and constraints[neighbor] is not None:
				for n_neighbor in constraints[neighbor]:
					if assignment[n_neighbor] is not None:
						illegal_values.add(assignment[n_neighbor])
				colors[color] += K - len(illegal_values)
				illegal_values.clear()
				illegal_values.add(color)
	return sorted(range(len(colors)), key=lambda i:colors[i], reverse = True)

'''
Improved DFSB Implementation

'''

def dfsb_plus_plus(N, K, constraints, assignment, arcs):
	global num_searches
	if is_solution_complete(assignment): return assignment
	arc_consistency_heuristics(arcs[:], constraints)
	unassigned_position = minimum_remaining_values_heuristics(K, assignment, constraints)
	if unassigned_position == -1: return 'failure'
	num_searches += 1
	colors = least_constraining_value_heuristics(K, assignment, constraints, unassigned_position)
	i = 0
	while i < len(colors):
		assignment[unassigned_position] = colors[i]
		if is_assignment_valid(N, assignment, constraints):
			result = dfsb_plus_plus(N, K, constraints, assignment, arcs)
			if result != 'failure': return result
		assignment[unassigned_position] = None
		i += 1
	return 'failure'


'''

Naive DFSB Implementation
'''
def dfsb(N, K, constraints, assignment, end_time):
	global num_searches
	if time.time() >= end_time: return 'failure'
	if is_solution_complete(assignment): return assignment
	num_searches += 1
	unassigned_position = [i for i in range(len(assignment)) if assignment[i] == None][0]
	color = 0
	while color < K:
		assignment[unassigned_position] = color
		if is_assignment_valid(N, assignment, constraints):
			result = dfsb(N, K, constraints, assignment, end_time)
			if result != 'failure': return result
		assignment[unassigned_position] = None
		color += 1
	return 'failure'


'''
Check whether the input is valid
'''
def input_validation(argv):
	try:
		if len(argv) != 4 or (int(argv[3]) != 0 and int(argv[3]) != 1): raise Exception()
	except Exception:
		print('Invalid Input! Try Again!\nUsage: python3 dfsb.py <INPUT_FILE_PATH> <OUTPUT_FILE_PATH> <#Mode>')
		detail = '''
		where,
		#Mode = 0 for DFSB or 1 for DFSB++ 	
		'''
		print(detail)
		exit(1)

if __name__ == '__main__':
	input_validation(sys.argv)
	num_searches = 0
	prunes = 0
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	algorithm = int(sys.argv[3])
	'''
	N: Number of nodes
	M: Number of constraints
	K: Number of colors
	constrainst: Adjacency list
	'''
	N, M, K, domains, constraints, arcs = generate_constraint_satisfaction_problem(input_file)
	assignment = [None] * N
	start_time = time.clock() * 1000.0
	# Based on mode, call either dfsb or dfsb++
	solution = dfsb(N, K, constraints, assignment, time.time() + 60) if algorithm == 0 else dfsb_plus_plus(N, K, constraints, assignment, arcs)
	end_time = time.clock() * 1000.0
	print_solution_to_output_file(output_file,solution)
	if(solution == 'failure'): print(solution)
	print("Time taken: " + str(end_time - start_time) + "ms")
	print("# of num_searches: " + str(num_searches))
	print("# of prunes: " + str(prunes))