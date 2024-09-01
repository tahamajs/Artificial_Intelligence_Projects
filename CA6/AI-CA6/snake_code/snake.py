from cube import Cube
from constants import *
from utility import *

import random
import random
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import copy


class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name=None):
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        try:
            with open(file_name, 'rb') as f:
                self.q_table = pickle.load(f)
        except:
            self.q_table = dict()

        self.lr = 0.1
        self.discount_factor = 0.7
        self.epsilon = 0.01
        
        self.hist = []
        self.hist_reward = []
        self.file_name = file_name

    def get_optimal_policy(self, state):
        return np.argmax(self.q_table[state])

    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon or state not in self.q_table.keys():
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)
        return action

    def update_q_table(self, state, action, next_state, reward):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
            self.q_table[state][state[1]] = 1
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(4)
            self.q_table[next_state][next_state[1]] = 1
        sample = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (sample - self.q_table[state][action])
    
    def create_state(self, snack, other_snake):
        neighbor = self.get_neighbor(3, other_snake)
        snake_side = self.calc_snake_side(snack)
        return (neighbor, snake_side)
        
    def get_neighbor(self, size, other_snake):
        distance = (size - 1) // 2
        tmp = np.array(range(distance + 1))
        tmp = np.union1d(tmp, -tmp)
        
        output = []
        for i in tmp + self.head.pos[0]:
            for j in tmp + self.head.pos[1]:
                if i < 1 or i >= ROWS - 1 or j < 1 or j >= ROWS - 1:
                    output.append(0)
                elif (i, j) in list(map(lambda z: z.pos, self.body)):
                    output.append(0)
                elif (i, j) in list(map(lambda z: z.pos, other_snake.body)):
                    output.append(0)
                elif (i, j) == other_snake.head.pos:
                    output.append(0)
                else:
                    output.append(1)
        return tuple(output)
    
    def calc_snack_distance(self, snack):
        return abs(snack.pos[0] - self.head.pos[0]) + abs(snack.pos[1] - self.head.pos[1])

    def calc_snake_side(self, snack):
        if abs(snack.pos[0] - self.head.pos[0]) > abs(snack.pos[1] - self.head.pos[1]):
            if snack.pos[0] < self.head.pos[0]:
                return 0
            if snack.pos[0] > self.head.pos[0]:
                return 1
        else:
            if snack.pos[1] < self.head.pos[1]:
                return 2
            if snack.pos[1] > self.head.pos[1]:
                return 3
        return -1
        
    def move(self, snack, other_snake):
        self.pre_head = copy.deepcopy(self.head)
        state = self.create_state(snack, other_snake)
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

        new_state = self.create_state(snack, other_snake)
        return state, new_state, action
    
    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False
    
    def calc_snack_reward(self, snack):
        dist1 = abs(snack.pos[0] - self.pre_head.pos[0]) + abs(snack.pos[1] - self.pre_head.pos[1])
        dist2 = abs(snack.pos[0] - self.head.pos[0]) + abs(snack.pos[1] - self.head.pos[1])
        if dist2 - dist1 < 0:
            return SNACK_RATE
        else:
            return -SNACK_RATE * 1.5
    
    
    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False
        
        reward += self.calc_snack_reward(snack)
        
        if self.check_out_of_board():
            win_other = True
            reward += LOSE_REWARD
            reset(self, other_snake, win_other)
        
        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            
            reward += EAT_REWARD
            
            
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            win_other = True
            reward += LOSE_REWARD
            reset(self, other_snake, win_other)
            
            
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            
            if self.head.pos != other_snake.head.pos:
                reward += LOSE_REWARD
                win_other = True
            else:
                if len(self.body) > len(other_snake.body):
                    reward += WIN_REWARD
                    win_self = True
                elif len(self.body) == len(other_snake.body):
                    pass
                else:
                    reward += LOSE_REWARD
                    win_other = True
                    
            reset(self, other_snake, win_other)
        self.hist_reward.append(reward)
        return snack, reward, win_self, win_other

    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1
        if len(self.hist_reward) > 2 and 's1' in self.file_name:
            self.hist.append(np.mean(self.hist_reward))
            self.hist_reward = []
            
            if len(self.hist) % 10 == 9:
                plt.plot(self.hist)
                plt.savefig(self.file_name[:3] + 'img')

            if len(self.hist) % 100 == 99:
                self.lr *= 0.95
                self.epsilon *= 0.92
                print(self.lr, self.epsilon)
            

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
        with open(file_name, 'wb') as f:
            pickle.dump(self.q_table, f)