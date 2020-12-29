import gym
import pygame
import numpy as np
from math import sqrt
ENV_WIDTH = 750
ENV_HEIGHT = 750


class Agent:
    def __init__(self, name, board):
        """
        This is the agent of the digitized environment. The agent is supposed to try riding around the environment and minimize the amount of collisions while riding. First however, the agent has to train in the environment, and learn to minimize the collisions and go from point A to point B smoothly.
        Args:
            name (str): This is the name for the agent and can be anything as long as it's a string. This is just for convenience when visualizing the progress of the agent after training.
        """
        self.board = board
        self.width = 50
        self.height = 25
        self.rewards = []
        self.reward_sum = sum(self.rewards)
        self.collisions = []
        # Data to see correlation between directions and distance measures
        self.direction_history = []
        self.distances = []
        self.agent_position = {'x': 10, 'y': 600}

        self.corners = [[self.agent_position['x'] + self.width, self.agent_position['y'], '1s'], [
            self.agent_position['x'] + self.width, self.agent_position['y'] + self.height, '11']]
        self.line_pos = [self.show_distances(p) for p in self.corners]
        self.angle = 90

    def show_distances(self, pos):
        new_pos = pos[:]
        for obstacle in Obstacle.instances:
            while 0 < new_pos[0] < ENV_WIDTH and 0 < new_pos[1] < ENV_HEIGHT and not (new_pos[0] in range(int(obstacle.x), int(obstacle.x + obstacle.width)) and new_pos[1] in range(int(obstacle.y), int(obstacle.y + obstacle.height))):
                operation = new_pos[2]
                for i, j in enumerate(operation):
                    if j == '1':
                        new_pos[i] += 1
                    elif j == 's':
                        new_pos[i] -= 1
            return new_pos

    def check_collision(self):
        distances = self.return_distances(self.corners, self.line_pos)
        left = distances[0]
        right = distances[1]
        for obstacle in Obstacle.instances:
            if left < 2 or right < 2:
                self.collisions.append({
                    'didCollide': True,
                    'distanceFromObject': left if left < 2 and right > 2 else right if right < 2 and left > 2 else min(left, right),
                    'nameOfCollision': obstacle.name,
                    'width': obstacle.width,
                    'height': obstacle.height
                })
                return True
            else:
                return False

    def return_distances(self, corners, end_line_pos):
        return sqrt((self.line_pos[0][0] - self.corners[0][0])**2 + (self.line_pos[0][1] - self.corners[0][1])), sqrt((self.line_pos[1][0] - self.corners[1][0])**2 + (self.line_pos[1][1] - self.corners[1][1])**2)

    def get_distances(self, corners, line_pos):
        return sqrt((self.line_pos[0][0] - self.corners[0][0])**2 + (self.line_pos[0][1] - self.corners[0][1]))


class Obstacle:
    instances = []

    def __init__(self, x, y, width, height, color, name):
        """
        This is the obstacle class. This class is here so that it's easier to create obstacles of the environment and position them given the size and position attributes as opposed to just hard-coding and drawing them in pygame. In the future, I plan to allow the user to make a couple modifications by a simple sequence of clicks.
        Args:
            width (int): Specifies the width of the object
            height (int): Specifies the height of the object
            color (tuple): Color in RGB tuple format
            name (str): Name of the obstacle - may make this optional but this is convenient for data visualization after training
        """
        self.__class__.instances.append(self)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.name = name


class Iota(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, model):
        """
        This is the environment. This environment is a custom Gym environment. A standard gym environment has threee methods: render, reset, and step. The render method is to update the environment with new positions of objects. Step is to update the agent's position given the action it predicted. Finally, reset is a method that is called when the agent reaches the terminal state. In other words, it resets the agent's position to it's default position and then restarts the training process. The only terminal state of the agent is when the agent collides with another object or at the environment's endpoints.
        """

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=750, shape=(2,), dtype=np.float32)
        pygame.init()
        pygame.display.set_caption("Iota Environment")
        self.board = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))
        self.bed = Obstacle(0, ENV_HEIGHT / 2 - 250 / 2,
                            400, 250, (255, 255, 255), "Bed")
        self.table = Obstacle(0, 0, 450, 200, (255, 255, 255), "Table")
        self.agent = Agent("Iota", self.board)
        hasFinished = False
        self.model = model
        obs = self.reset()
        while not hasFinished:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    hasFinished = True
            action, states = model.predict(obs)
            obs, rewards, dones, info = self.step(action)
            print(obs, self.agent.angle)
            self.render()

    def render(self, mode='human', close=False):

        self.board.fill((0, 0, 0))
        pygame.draw.rect(self.board, self.bed.color, [
                         self.bed.x, self.bed.y, self.bed.width, self.bed.height])
        pygame.draw.rect(self.board, self.table.color, [
                         self.table.x, self.table.y, self.table.width, self.table.height])
        pygame.draw.rect(self.board, (255, 255, 255), [
                         self.agent.agent_position["x"], self.agent.agent_position["y"], self.agent.width, self.agent.height])
        for i, pos in enumerate(self.agent.line_pos):
            pygame.draw.line(self.board, (255, 255, 255),
                             (self.agent.corners[i][0], self.agent.corners[i][1]), (pos[0], pos[1]))

        pygame.display.update()

    def reset(self):

        self.agent.agent_position['x'] = 10
        self.agent.agent_position['y'] = ENV_HEIGHT / 3 + self.bed.height + 100
        self.corners = [[self.agent.agent_position['x'] + self.agent.width, self.agent.agent_position['y'], '1s'], [
            self.agent.agent_position['x'] + self.agent.width, self.agent.agent_position['y'] + self.agent.height, '11']]
        self.line_pos = [self.agent.show_distances(p) for p in self.corners]
        dist1, dist2 = self.agent.return_distances(self.corners, self.line_pos)

        self.render()
        return np.array([dist1, dist2])

    def step(self, action):
        print(action)
        """
        array[4] = [0, 0, 0, 0]
        array[0] = forward
        array[1] = left
        array[2] = reverse
        array[3] = right
        """
        distances = self.agent.return_distances(self.agent.corners, self.agent.line_pos)

        left = distances[0]
        right = distances[1]
        self.agent.distances.append({
            'left': left,
            'right': right
        })
        reward = 0
        if action == 1:
            self.agent.angle -= 90
            if self.agent.angle < 0:
                self.agent.angle = 360
            self.agent.direction_history.append('left')
            self.reset_raycasts(self.agent.angle)
            self.render()
            if left > right:
                reward += 5
            else:
                reward -= 5

        elif action == 2:
            if self.agent.angle == 360: 
                self.agent.angle = 0
            else: self.agent.angle += 90

            self.agent.angle += 90
            if self.agent.angle > 360:
                self.agent.angle = 0

            self.reset_raycasts(self.agent.angle)
            self.render()
            self.agent.direction_history.append('right')
            if left < right:
                reward += 5
            else:
                reward -= 5

        elif action == 0:
            self.agent.direction_history.append('forward')
            if self.agent.angle == 0:
                self.agent.agent_position['y'] -= 10
                self.reset_raycasts(self.agent.angle)
            elif self.agent.angle == 90: 
                self.agent.agent_position['x'] += 10
                self.reset_raycasts(self.agent.angle)
            elif self.agent.angle == 180: 
                self.agent.agent_position['y'] += 10
                self.reset_raycasts(self.agent.angle)
            elif self.agent.angle == 270:
                self.agent.agent_position['x'] -= 10
                self.reset_raycasts(self.agent.angle)
            
            if left + right >= 50:
                reward += 5

            elif self.agent.check_collision():
                reward -= 10
                self.reset() 

            self.render()

        elif action == 1:
            self.agent.direction_history.append('reverse')
            if self.agent.angle == 0:
                self.agent.agent_position['y'] += 10
                self.reset_raycasts(self.agent.angle)
                self.render()
            elif self.agent.angle == 90: 
                self.agent.agent_position['x'] -= 10
                self.reset_raycasts(self.agent.angle)
                self.render()
            elif self.agent.angle == 180: 
                self.agent.agent_position['y'] -= 10
                self.reset_raycasts(self.agent.angle)
                self.render()
            elif self.agent.angle == 270:
                self.agent.agent_position['x'] += 10
                self.reset_raycasts(self.agent.angle)
                self.render()
            
            if left + right <= 50:
                reward += 5

            elif self.agent.check_collision():
                reward -= 10
                self.reset() 
            else:
                reward -= 5

            self.agent.rewards.append({
                'leftDistance': left,
                'rightDistance': right,
                'reward': reward,
            })
        info = {}
        self.render()
            # self.render()
            # print(self.agent.direction_history[-1])
        self.agent.rewards.append(reward)
        return np.array([left, right]), reward, False, info

    def reset_raycasts(self, angle_of_agent):
        if angle_of_agent == 0:
            if self.agent.width > self.agent.height:
                self.agent.width, self.agent.height = self.agent.height, self.agent.width
            self.agent.corners = [[self.agent.agent_position['x'], self.agent.agent_position['y'], 'ss'], [
                    self.agent.agent_position['x'] + self.agent.width, self.agent.agent_position['y'], '1s']]
            self.agent.line_pos = [self.agent.show_distances(p) for p in self.agent.corners]
            self.render()
        elif angle_of_agent == 90:
            if self.agent.width < self.agent.height:
                self.agent.width, self.agent.height = self.agent.height, self.agent.width
            self.agent.corners = [[self.agent.agent_position['x'] + self.agent.width, self.agent.agent_position['y'], '1s'], [
                    self.agent.agent_position['x'] + self.agent.width, self.agent.agent_position['y'] + self.agent.height, '11']]
            self.agent.line_pos = [self.agent.show_distances(p) for p in self.agent.corners]
            self.render()

        elif angle_of_agent == 180:
            if self.agent.width > self.agent.height:
                self.agent.width, self.agent.height = self.agent.height, self.agent.width
            self.agent.corners = [[self.agent.agent_position['x'], self.agent.agent_position['y'] + self.agent.height, 's1'], [
                    self.agent.agent_position['x'] + self.agent.width, self.agent.agent_position['y'] + self.agent.height, '11']]
            self.agent.line_pos = [self.agent.show_distances(p) for p in self.agent.corners]
            self.render()
        elif angle_of_agent == 270:
            if self.agent.width < self.agent.height:
                self.agent.width, self.agent.height = self.agent.height, self.agent.width
            self.agent.corners = [[self.agent.agent_position['x'], self.agent.agent_position['y'], 'ss'], [
                    self.agent.agent_position['x'], self.agent.agent_position['y'] + self.agent.height, 's1']]
            self.agent.line_pos = [self.agent.show_distances(
                    p) for p in self.agent.corners] 

            self.render()