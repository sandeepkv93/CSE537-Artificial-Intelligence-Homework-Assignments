from sys import argv, exit
from heapq import heappush, heappop
from time import clock


class Graph_Node:
    puzzle_format = None
    heuristic_type = None
    explored_states = 0
    visited_states = {}
    '''
	Heuristic Switch
    '''

    @staticmethod
    def find_heuristic_function(self):
        if Graph_Node.heuristic_type == 1:
            return lambda: self.manhattan_distance_heuristic()
        elif Graph_Node.heuristic_type == 2:
            return lambda: self.misplaced_tiles_heuristic()

    '''
	Initialize the Graph_Node class
    '''

    def __init__(self, state, ancestor, action):
        self.state = state
        self.ancestor = ancestor
        self.action = action

        if not ancestor:
            self.path_cost = 0
            self.depth = 0
        else:
            self.path_cost = ancestor.path_cost + 1
            self.depth = ancestor.depth + 1
        self.heuristic_cost = Graph_Node.find_heuristic_function(self)()
        self.total_cost = self.path_cost + self.heuristic_cost

    '''
	Comparator for heapq

    '''

    def __lt__(self, node):
        return self.path_cost < node.path_cost

    '''
	For moving the tile in 4 different directions
    '''

    def move(self, empty_slot, direction):
        current_state = list(self.state)
        if direction == 'left':
            current_state[empty_slot], current_state[
                empty_slot - 1] = current_state[empty_slot -
                                                1], current_state[empty_slot]
        elif direction == 'right':
            current_state[empty_slot], current_state[
                empty_slot + 1] = current_state[empty_slot +
                                                1], current_state[empty_slot]
        elif direction == 'up':
            current_state[empty_slot], current_state[
                empty_slot - Graph_Node.puzzle_format] = current_state[
                    empty_slot -
                    Graph_Node.puzzle_format], current_state[empty_slot]
        elif direction == 'down':
            current_state[empty_slot], current_state[
                empty_slot + Graph_Node.puzzle_format] = current_state[
                    empty_slot +
                    Graph_Node.puzzle_format], current_state[empty_slot]
        return current_state

    '''
	Check if the node is rightmost node
    '''

    @staticmethod
    def is_rightmost_edge(empty_slot):
        return (empty_slot + 1) % Graph_Node.puzzle_format == 0

    '''
	Check if the node is leftmost node
    '''

    @staticmethod
    def is_leftmost_edge(empty_slot):
        return (empty_slot + 1) % Graph_Node.puzzle_format == 1

    '''
	Check if the node is topmost node
    '''

    @staticmethod
    def is_topmost_edge(empty_slot):
        return (empty_slot + 1) <= Graph_Node.puzzle_format

    '''
	Check if the node is bottommost node
    '''

    @staticmethod
    def is_bottommost_edge(empty_slot):
        return (empty_slot +
                1) > (Graph_Node.puzzle_format**2) - Graph_Node.puzzle_format

    def find_possible_transitions(self):
        possible_transitions_list = []
        empty_slot = self.state.index(0)
        if not Graph_Node.is_rightmost_edge(empty_slot):
            possible_transitions_list.append(
                ['R', self.move(empty_slot, 'right')])
        if not Graph_Node.is_leftmost_edge(empty_slot):
            possible_transitions_list.append(
                ['L', self.move(empty_slot, 'left')])
        if not Graph_Node.is_topmost_edge(empty_slot):
            possible_transitions_list.append(['U', self.move(empty_slot, 'up')])
        if not Graph_Node.is_bottommost_edge(empty_slot):
            possible_transitions_list.append(
                ['D', self.move(empty_slot, 'down')])
        return possible_transitions_list

    '''
	Misplaced tiles heuristic algorithm
	Finds total number of misplaced tiles in a current state
    '''

    def misplaced_tiles_heuristic(self):
        return len(
            list(
                filter(lambda x: x != 0 and x != self.state.index(x) + 1,
                       self.state)))

    '''
	Manhattan Distance heuristic algorithm
	Finds the total of manhattan distance of each tile with the goal state
	http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
		function heuristic(node) =
    		dx = abs(node.x - goal.x)
    		dy = abs(node.y - goal.y)
    		return D * (dx + dy)
    			where D can be assumed to be 1
    '''

    def manhattan_distance_heuristic(self):
        manhattan_distance = 0
        for each_node in self.state:
            if each_node != 0:
                cur_x = self.state.index(each_node) % Graph_Node.puzzle_format
                cur_y = self.state.index(each_node) // Graph_Node.puzzle_format
                goal_x = (each_node - 1) % Graph_Node.puzzle_format
                goal_y = (each_node - 1) // Graph_Node.puzzle_format
                dx = abs(cur_x - goal_x)
                dy = abs(cur_y - goal_y)
                manhattan_distance += (dx + dy)
        return manhattan_distance

    '''
    Chck if the current state is the goal state
    '''

    def goal_test(self):
        return self.state == list(range(
            Graph_Node.puzzle_format**2))[1:] + list(
                range(Graph_Node.puzzle_format**2))[:1]


'''
Checking the actions are reversible actions

'''


def check_for_valid_transitions(node, transition):
    if node.ancestor == None:
        return True
    if transition[1] != node.ancestor.state:
        return True


'''

a star search algorithm
uses a priority queue to push the states.
for each node find all possible transitions and put them in priority queue and pop them to check for goals state
'''


def a_star_search(opening_state):
    frontier = []
    heappush(frontier, (opening_state.total_cost, opening_state))
    while frontier:
        current = heappop(frontier)[1]
        cur_state_in_string = ','.join(str(a) for a in current.state)
        Graph_Node.visited_states[
            cur_state_in_string] = Graph_Node.visited_states.get(
                cur_state_in_string, 0) + 1
        if current.goal_test():
            return current
        for transition in current.find_possible_transitions():
            if check_for_valid_transitions(current, transition):
                if ','.join(str(a) for a in transition[1]
                           ) not in Graph_Node.visited_states:
                    new_node = Graph_Node(transition[1], current, transition[0])
                    heappush(frontier, (new_node.total_cost, new_node))
    return False


'''
Recursive Depth First Search with limit considerations
'''


def dfs(node, limit):
    Graph_Node.explored_states += 1
    if node.goal_test(): return node
    elif node.total_cost > limit: return node.total_cost
    else:
        threshold, found_new_limit = float("inf"), False
        for transition in node.find_possible_transitions():
            if check_for_valid_transitions(node, transition):
                next_depth = dfs(
                    Graph_Node(transition[1], node, transition[0]), limit)
                if type(next_depth) == int:
                    found_new_limit, threshold = True, min(
                        threshold, next_depth)
                elif next_depth.__class__.__name__ == 'Graph_Node':
                    return next_depth
        if found_new_limit:
            return threshold
        else:
            return False


'''
IDA Search. Calls recursive dfs with limit parameter till reciveing the goal state
'''


def id_a_star_search(opening_state):
    limit = opening_state.total_cost
    recur_dfs = True
    while recur_dfs:
        result = dfs(opening_state, limit)
        if result.__class__.__name__ == 'int':
            limit = result
        elif result.__class__.__name__ == 'bool' and not result:
            recur_dfs = False
        elif result.__class__.__name__ == 'Graph_Node':
            return result
    return recur_dfs


'''
To output the solution
'''


def create_solution_file(algo_id, Node, output_solution_file):
    if algo_id == 1:
        print(
            "States Explored: " + str(sum(Graph_Node.visited_states.values())))
    elif algo_id == 2:
        print("States Explored: " + str(Graph_Node.explored_states))
    print("Solution Depth: " + str(Node.depth))
    path = []
    while Node.ancestor != None:
        path.append(Node.action)
        Node = Node.ancestor
    with open(output_solution_file, 'w') as output_file:
        output_file.write(','.join(path[::-1]))


'''
For choosing thhe search Algorithm
'''


def choose_search_algo(algo_id, opening_state):
    if algo_id == 1:
        return lambda: a_star_search(opening_state)
    else:
        return lambda: id_a_star_search(opening_state)


'''
To generate the puzzle array from input file
'''


def genrate_puzzle_array(input_puzzle_file):
    puzzle = []
    with open(input_puzzle_file, 'r') as file:
        for line in file:
            puzzle += [
                0 if node.strip() == '' else int(node.strip())
                for node in line.split(',')
            ]
    return puzzle


'''
Input Validation
'''


def input_validation(argv):
    try:
        if len(argv) != 6:
            raise Exception()
        if int(argv[1]) != 1 and int(argv[1]) != 2:
            raise Exception()
        if int(argv[3]) != 1 and int(argv[3]) != 2:
            raise Exception()
    except Exception:
        print(
            'Invalid Input! Try Again!\nUsage: python3 puzzleSolver.py <#Algorithm> <N> <H> <INPUT_FILE_PATH> <OUTPUT_FILE_PATH>\node'
        )
        detail = '''
		where,
			#Algorithm: 1 = A* and 2 = Memory bounded variant.
			N: 3 = 8-puzzle 4 = 15-puzzle format.
			H: 1 = Heuristic 1 and 2 = Heuristic 2.
			INPUT_FILE_PATH = The path to the input file.
			OUTPUT_FILE_PATH = The path to the output file.
		'''
        print(detail)
        exit(1)


if __name__ == "__main__":
    input_validation(argv)
    algo_id = int(argv[1])
    Graph_Node.puzzle_format = int(argv[2])
    Graph_Node.heuristic_type = int(argv[3])
    input_puzzle_file = argv[4]
    output_solution_file = argv[5]
    opening_state = Graph_Node(
        genrate_puzzle_array(input_puzzle_file), None, None)
    print("Initial State: " + ' '.join(map(str, opening_state.state)))
    start = clock() * 1000.0
    search_algorithm = choose_search_algo(algo_id, opening_state)
    create_solution_file(algo_id, search_algorithm(), output_solution_file)
    end = clock() * 1000.0
    print("Solution Time: " + str(end - start) + "ms")
