import random
import random
import numpy as np


class Snake:
    def __init__(self, color, pos, file_name=None):
        self.color = color
        self.head = Cube(pos, color=color)
        self.body = [self.head]
        self.dirnx = 0
        self.dirny = 1
        self.turns = {}
        self.num_of_wins = 0
        self.total_reward = 0
        
        self.q_table = np.load(file_name, allow_pickle=True).item() if file_name and os.path.exists(file_name) else {}

        self.lr = 0.1 
        self.discount_factor = 0.9
        self.epsilon = 0.95
        
        
        
    def get_state(self, snack, other_snake):
        head_x, head_y = self.head.pos
        snack_x, snack_y = snack.pos
        other_head_x, other_head_y = other_snake.head.pos
        
        # Compute differences (distances) between snake head and food, and snake head and other snake's head
        diff_snack_x = abs(head_x - snack_x)
        diff_snack_y = abs(head_y - snack_y)
        diff_other_x = abs(head_x - other_head_x)
        diff_other_y = abs(head_y - other_head_y)
        
        
        # State representation using differences
        state = (diff_snack_x, diff_snack_y, diff_other_x, diff_other_y)
        return state


    def get_optimal_policy(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        return np.argmax(self.q_table[state])
    
    
    def make_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            return self.get_optimal_policy(state)

    def update_q_table(self, state, action, next_state, reward):
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(4)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error 
        
        

    def move(self, snack, other_snake):
        state = self.get_state(snack, other_snake)
        action = self.make_action(state)

        if action == 0: # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1: # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2: # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3: # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        # snack, reward, win_self = self.calc_reward(snack, other_snake)
        next_state = self.get_state(snack, other_snake)
        
        
        
        return state, next_state, action
        
    
    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False
    
    
    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self  = False
        other_win = False

    
        if self.check_out_of_board():
            reward -= 100000
            # win_self = False
            other_win = True
            
            self.reset((random.randint(3, 18), random.randint(3, 18)))
    
        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward += 10000
            self.num_of_wins += 1
    
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            reward -= 10000
            win_self = False
            other_win = True
            
            self.reset((random.randint(3, 18), random.randint(3, 18)))
    
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            if self.head.pos != other_snake.head.pos:
                reward -= 10000
                win_self = False
                other_win = True
                
            else:
                if len(self.body) > len(other_snake.body):
                    reward += 10000
                    win_self = True
                    other_win = False
                elif len(self.body) == len(other_snake.body):
                    reward = 0
                else:
                    reward -= 10000
                    win_self = False
                    other_win = True
            self.reset((random.randint(3, 18), random.randint(3, 18)))
    
        distance_to_snack = self.distance_to_snack(snack)
        if not distance_to_snack == 0:
            reward += 1 / distance_to_snack*10000

        # کاهش epsilon
        if self.epsilon > 0.1:
            self.epsilon = self.epsilon * epsilon_reduction
            
        self.total_reward += reward
        
        
        return snack, reward, win_self , other_win   
    
    
        
        

    def avrage_distance_to_other_snake(self, other_snake):
        sum = 0
        for cube in other_snake.body:
            sum += self.distance_to_snack(cube)
        return sum / len(other_snake.body)
    
    def distance_to_snack(self, snack):
        head_x, head_y = self.head.pos
        snack_x, snack_y = snack.pos
        return abs(head_x - snack_x) + abs(head_y - snack_y)

    
    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = [self.head]
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1
        self.num_of_wins = 0
        

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny
        

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table, allow_pickle=True)
        