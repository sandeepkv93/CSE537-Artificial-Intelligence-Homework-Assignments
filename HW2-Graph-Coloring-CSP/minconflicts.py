from sys import argv
from time import clock
import random
'''
Parses the input file to get N, M, K and adjacency list i.e constraint list
'''


def parse_input_file(input_file):
    with open(input_file, 'r') as inp_fi:
        lines = inp_fi.read().splitlines()
    N, M, K = map(int, lines[0].split('\t'))
    constraint_lines = lines[1:]
    return N, M, K, constraint_lines


'''
Generates the Constraint satisfaction problem by parsing and processing the inputs given
'''


def generate_constraint_satisfaction_problem(input_file):
    N, M, K, constraint_lines = parse_input_file(input_file)
    constraints = [[] for x in range(N)]
    for line in constraint_lines:
        constraint = list(map(int, line.split('\t')))
        constraints[constraint[0]].append(constraint[1])
        constraints[constraint[1]].append(constraint[0])
    return N, M, K, constraints


'''
Prints the solution to the output file.
'''


def print_solution_to_output_file(output_file, solution):
    with open(output_file, 'w') as out_fi:
        out_fi.write('No Answer') if solution == 'failure' else out_fi.write(
            '\n'.join([str(a) for a in solution]))


'''
Check whether the assignment is valid and successful
'''


def is_assignment_consistent(N, assignment, constraints):
    return not any(assignment[var] == assignment[constraint] and
                   assignment[constraint] is not None
                   for var in range(N)
                   for constraint in constraints[var])


'''
Minimum conflicts search algorithm
initalize assignment greedily calling minimize_the_conflicts
then choose random conflicted position and assign a value using calling minimize_the_conflicts
'''


def minimum_conflicts(N, M, K, constraints, maximum_iteration):
    global num_searches, assignment
    for i in range(0, N):
        assignment[i] = minimize_the_conflicts(i, assignment, constraints, K)
    for i in range(maximum_iteration):
        if is_assignment_consistent(N, assignment, constraints):
            return True
        var = random.choice([
            i for i in range(N) for constraint in constraints[i]
            if assignment[i] == assignment[constraint]
        ])
        assignment[var] = minimize_the_conflicts(var, assignment, constraints,
                                                 K)
        num_searches += 1
    return False


'''
Chooses a color in such a way as to minimize the number of conflicts. 
If multiple possibilities exist, it chooses a color randomly.
'''


def minimize_the_conflicts(var, assignment, constraints, K):
    conflicts = [
        len([1
             for neighbor in constraints[var]
             if assignment[neighbor] == i])
        for i in range(K)
    ]
    return random.choice(
        [i for i, v in enumerate(conflicts) if v == min(conflicts)])


'''
Check whether the input is valid
'''


def input_validation(arg):
    try:
        if len(arg) != 3: raise Exception()
    except Exception:
        print(
            'Invalid Input! Try Again!\nUsage: python3 minconflicts.py <INPUT_FILE_PATH> <OUTPUT_FILE_PATH> '
        )
        exit(1)


if __name__ == '__main__':
    input_validation(argv)
    input_file = argv[1]
    output_file = argv[2]
    '''
	N: Number of nodes
	M: Number of constraints
	K: Number of colors
	constrainst: Adjacency list
	'''
    N, M, K, constraints = generate_constraint_satisfaction_problem(input_file)
    assignment = [None] * N
    current = 0
    failure = True
    num_searches = 0
    start = clock() * 1000.0
    while True:
        if minimum_conflicts(N, M, K, constraints, 120000):
            print_solution_to_output_file(output_file, assignment)
            failure = False
            break
        print('Random restart')
        print(num_searches)
        assignment = [None] * N
        # exit after two minutes
        if (clock() * 1000.0) - start < 120000:
            failure = True
            break
    if failure:
        print('failure')
        print_solution_to_output_file(output_file, 'failure')
    end = clock() * 1000.0
    print("Time taken: " + str(end - start) + "ms")
    print("# of Searches: " + str(num_searches))
