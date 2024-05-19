import random
import math

class State:
    def __init__(self, pieces=None, enemy_pieces=None):
        self.pieces = pieces if pieces != None else [0] * 9
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [0] * 9

    def piece_count(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count += 1
        return count
    
    # function to check 3 pieces connected()
    def is_comp(self, x, y, dx, dy):
        for i in range(3):
            if y < 0 or 2 < y or x < 0 or 2 < x or self.enemy_pieces[x + y * 3] == 0:
                return False
            x = x + dx
            y = y + dy
        return True

    # handle the case of lose
    def is_lose(self):
        # handle the case of diagonal
        if self.is_comp(0, 0, 1, 1) or self.is_comp(0, 2, 1, -1):
            return True
        
        for i in range(3):
            # handle the case of vertical & horizontal
            if self.is_comp(0, i, 1, 0) or self.is_comp(i, 0, 0, 1):
                return True
            
        return False
    
    # handle the case of lose
    def is_draw(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == 9
    
    # handle the case of draw
    def is_done(self):
        return self.is_lose() or self.is_draw()
    
    # get next state
    def next(self, action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_pieces, pieces)
    
    # list of available positions for placing pieces
    def legal_actions(self):
        actions = []
        for i in range(9):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions
    
    # check who is first player
    def is_first_player(self):
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)
    
    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        str = ''

        for i in range(9):
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i]:
                str += ox[1]
            else:
                str += '-'

            if i % 3 == 2:
                str += '\n'
        
        return str

# select action in random
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions) - 1)]

# calculate score with alpha beta mini-max algorithm
def alpha_beta(state, alpha, beta):
    if state.is_lose():
        return -1

    if state.is_draw():
        return 0

    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha)
        if score > alpha:
            alpha = score

        if alpha >= beta:
            return alpha

    return alpha

# select action in alpha-beta algorithm
def alpha_beta_action(state):
    best_action = 0
    alpha = -float('inf')
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha)
        if score > alpha:
            best_action = action
            alpha = score

    return best_action

def playout(state):
    if state.is_lose():
        return -1

    if state.is_draw():
        return 0
    
    return -playout(state.next(random_action(state)))


def argmax(collection):
    return collection.index(max(collection))

# select action in mcts
def mcts_action(state):
    class node:
        def __init__(self, state):
            self.state = state
            self.w = 0
            self.n = 0
            self.child_nodes = None

        def evaluate(self):
            if self.state.is_done():
                value = -1 if self.state.is_lose() else 0 

                self.w += value
                self.n += 1
                return value

            if not self.child_nodes:
                value = playout(self.state)

                self.w += value
                self.n += 1

                if self.n == 10:
                    self.expand()
                return value

            else:
                value = -self.next_child_node().evaluate()

                self.w += value
                self.n += 1
                return value

        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(node(self.state.next(action)))

        # return child node which had the best UCB1
        def next_child_node(self):
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            # calculate the UCB1 
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(-child_node.w / child_node.n + 2 * (2 * math.log(t) / child_node.n) ** 0.5)

            return self.child_nodes[argmax(ucb1_values)]

    root_node = node(state)
    root_node.expand()

    for _ in range(100):
        root_node.evaluate()

    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]



if __name__ == '__main__':
    state = State()

    while True:
        if state.is_done():
            break

        action = random_action(state)
        state = state.next(action)

        print(state)
