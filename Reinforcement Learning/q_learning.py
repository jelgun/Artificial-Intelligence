import numpy as np
import copy
import matplotlib.pyplot as plt
np.random.seed(62)

# PARAMETERS
R = -0.01 # state rewads
D = 0.9 # discount factor
P = 0.8 # probability


# epsilon for convergences
eps = 1e-3

env = [
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, -10, 10]
]
end_positions = [[0, 3], [3, 1], [3, 2], [3, 3]]

# movements (up, down, right, left)
dir_x = [0, 0, 1, -1]
dir_y = [-1, 1, 0, 0]

# alternate directions agent can take due to probabilities
alter_dirs = [
    [3, 2],
    [2, 3],
    [0, 1],
    [1, 0]
]

# check if the given pos is legal
def legal_pos(x, y):
    if x < 4 and x >= 0 and y < 4 and y >= 0 and not (x == 1 and y == 1):
        return True
    return False

# apply movement to grid cell
def apply_action(x, y, action):
    next_x = x + dir_x[action]
    next_y = y + dir_y[action]
    # if next position is not legal then don't move
    if not legal_pos(next_x, next_y):
        next_x = x
        next_y = y

    return next_x, next_y


def check_convergence(prev_board, next_board):
    for i in range(4):
        for j in range(4):
            if abs(prev_board[i][j] - next_board[i][j]) > eps:
                return False
    return True


def value_iteration(board_values, r, d, p):
    # updated versions of board values
    next_board_values = copy.deepcopy(board_values)

    # iterate through all positions
    for y in range(4):
        for x in range(4):
            # ending position or blocked cell
            if [y, x] in end_positions or (x == 1 and y == 1):
                next_board_values[y][x] = env[y][x]
                continue
            
            action_values = []
            # go over all actions
            for i in range(4):
                # next states' position
                next_x, next_y = apply_action(x, y, i)
                # caculate the Value function

                val = board_values[next_y][next_x]
                if x == next_x and y == next_y:
                    val = 0
                V = p * (r + d * val)
                # consider other movements
                for j in range(2):
                    next_x, next_y = apply_action(x, y, alter_dirs[i][j])
                    val = board_values[next_y][next_x]
                    if x == next_x and y == next_y:
                        val = 0
                    V += (1 - p) / 2 * (r + d * val)
                
                action_values.append(V)
            
            # apply Bellman eq. 
            next_board_values[y][x] = max(action_values)

    return next_board_values


def policy_iteration(board_policies, board_values, r, d, p):
    next_board_policies = copy.deepcopy(board_policies)
    next_board_values = copy.deepcopy(board_values)

    # calculate values for the given policy, repeat until convergence
    while True:
        for y in range(4):
            for x in range(4):
                # ending position or blocked cell
                if [y, x] in end_positions or (x == 1 and y == 1):
                    next_board_values[y][x] = env[y][x]
                    continue
                
                # apply the policy
                next_x, next_y = apply_action(x, y, board_policies[y][x])

                val = board_values[next_y][next_x]
                # check wall
                if x == next_x and y == next_y:
                    val = 0
                
                # caculate the Value function
                V = p * (r + d * val)

                # consider other movements
                direct = board_policies[y][x]
                for j in range(2):
                    next_x, next_y = apply_action(x, y, alter_dirs[direct][j])
                    val = board_values[next_y][next_x]
                    if x == next_x and y == next_y:
                        val = 0
                    V += (1 - p) / 2 * (r + d * val)

                next_board_values[y][x] = V
        
        # check convergence
        if check_convergence(board_values, next_board_values):
            break
        board_values = copy.deepcopy(next_board_values)

    # update policies according to the new values
    for y in range(4):
        for x in range(4):
            # end position or block
            if [y,x] in end_positions or (x == 1 and y == 1):
                continue
            action_values = []
            for i in range(4):
                # next states' position
                next_x, next_y = apply_action(x, y, i)

                val = board_values[next_y][next_x]
                # check wall
                if x == next_x and y == next_y:
                    val = 0
                action_values.append(val)

            # update the current policy
            next_board_policies[y][x] = np.argmax(action_values)

    return next_board_policies, board_values


def q_learning(N, r, d, alpha, exploration, board_values, board_policies):
    # grid cell to state mapper
    cell2state = {}

    # fill cell2state
    k = 0
    for y in range(4):
        for x in range(4):
            if [y, x] not in end_positions and not (x == 1 and y == 1):
                cell2state[(y, x)] = k
                k += 1

    # create Q table
    q = np.zeros((4, 11))
    counter = np.zeros((4, 11))
    utility_errors = []
    policy_errors = []
    # run q learning
    for step in range(N):
        if step % 100 == 0:
            # calculate utility and policy errors
            u_error = 0
            p_error = 0
            for i in range(4):
                for j in range(4):
                    try:
                        state = cell2state[(i, j)]
                        u_error += (np.max(q[:, state]) - board_values[i][j]) ** 2
                        p_error += (np.argmax(q[:, state]) != board_policies[i][j])
                    except:
                        pass
            # get average error of steps
            u_error /= 11
            p_error /= 11
            utility_errors.append(u_error)
            policy_errors.append(p_error)
        
        # reset initial position
        y = 2
        x = 0

        while(True):
            state = cell2state[(y, x)]
            action = np.argmax(q[:, state])
            rnd = np.random.rand()
            if rnd < exploration:
                action = np.random.randint(4)
            
            rnd = np.random.rand()
            if rnd < 0.8:
                rnd_action = action
            elif rnd < 0.9:
                rnd_action = alter_dirs[action][0]
            else:
                rnd_action = alter_dirs[action][1]

            next_x, next_y = apply_action(x, y, rnd_action)
            if [next_y, next_x] in end_positions:
                action_value = env[next_y][next_x]
            else:
                next_state = cell2state[(next_y, next_x)]
                action_value = np.max(q[:, next_state])
            
            # if border
            if x == next_x and y == next_y:
                action_value = 0

            counter[action][state] += 1
            alpha = 1 / counter[action][state]
            q[action][state] = (1-alpha)*q[action][state] + alpha*(r + d*action_value)
            
            x, y = next_x, next_y
            if [y, x] in end_positions:
                break

    # print q table
    for i in range(4):
        output = ""
        for j in range(11):
            output += "{0: <8}| ".format(round(q[i][j], 3))
        print("{0: <5}| ".format("a" + str(i)), output)

    # optimal policy
    opt_policy = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            try:
                state = cell2state[(i, j)]
                opt_policy[i][j] = np.argmax(q[:, state])
            except:
                pass

    # plot utility errors
    plt.plot(range(0, N, 100), utility_errors)
    plt.xlabel('Experiments')
    plt.ylabel('Utility Error')
    plt.savefig('a.png')

    # plot policy errors
    plt.clf()
    plt.plot(range(0, N, 100), policy_errors)
    plt.xlabel('Experiments')
    plt.ylabel('Policy Error')
    plt.savefig('b.png')


    return opt_policy


def print_board_values(board):
    output = ""
    k = 0
    for x in range(4):
        for y in range(4):
            if [x, y] not in end_positions and not (x == 1 and y == 1):
                board[x][y] *= 1.0
                output += "{0: <8}| ".format(round(board[x][y], 3))
                k += 1

    return output

def print_policy(board):
    for i in range(4):
        output = "|"
        for j in range(4):
            if board[i][j] == 0:
                output += ' ^'
            elif board[i][j] == 1:
                output += ' v'
            elif board[i][j] == 2:
                output += ' >'
            else:
                output += ' <'
            output += ' |'
        print(output)


def print_states():
    output = "     |  "
    for i in range(11):
        output += "{0: <8}| ".format(f"s{i}")
    print(output)
    print('-'*117)


# Value Iteration
# initialize the board with 0s
board_values = [[0, 0, 0, 0] for i in range(4)]
print("Value Iteration:\n")
print_states()
print("{0: <5}| ".format("it0"), print_board_values(board_values))

i = 1
# repeat until convergence
while True:
    # calculate next board values
    next_board_values = value_iteration(board_values, R, D, P)
    print("{0: <5}| ".format("it" + str(i)), print_board_values(next_board_values))

    # stop the iteration if converges
    if check_convergence(board_values, next_board_values):
        break
    board_values = next_board_values
    i += 1

vi_values = copy.deepcopy(board_values)

# Policy Iteration
# initialize the policies with up direction
board_policies = [[0, 0, 0, 0] for i in range(4)]

print("\nPolicy Iteration:\n")
print_states()
board_values = [[0, 0, 0, 0] for i in range(4)]
print("{0: <5}| ".format("it0"), print_board_values(board_values))


i = 1
# repeat until convergence
while True:
    # calculate next board policies
    next_board_policies, next_board_values = policy_iteration(board_policies, board_values, R, D, P)
    print("{0: <5}| ".format("it" + str(i)), print_board_values(next_board_values))

    if board_policies == next_board_policies:
        break

    board_policies = next_board_policies
    board_values = next_board_values
    i += 1

print_policy(board_policies)

print("\nQ Learning")
print_states()
q_policy = q_learning(18201, R, D, 0.1, 0.1, vi_values, board_policies)
print_policy(q_policy)
