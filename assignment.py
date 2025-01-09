import numpy as np
from matplotlib import pyplot as plt
import random

# Step 1: Define the environment
class RectanglePackingEnv:
    def __init__(self, bin_width, bin_height, rectangles):
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.rectangles = rectangles
        self.state = []
        self.reset()

    def reset(self):
        self.state = np.zeros((self.bin_width, self.bin_height), dtype=int)
        self.rectangles_to_place = self.rectangles.copy()
        return self.state

    def step(self, action):
        rect_idx, x, y = action
        rect_w, rect_h = self.rectangles_to_place[rect_idx]

        if self._can_place_rectangle(x, y, rect_w, rect_h):
            self._place_rectangle(x, y, rect_w, rect_h)
            del self.rectangles_to_place[rect_idx]
            reward = rect_w * rect_h
            done = len(self.rectangles_to_place) == 0
            return self.state, reward, done, {}
        else:
            return self.state, -1, False, {}

    def _can_place_rectangle(self, x, y, w, h):
        if x + w > self.bin_width or y + h > self.bin_height:
            return False
        if np.any(self.state[x:x + w, y:y + h] != 0):
            return False
        return True

    def _place_rectangle(self, x, y, w, h):
        self.state[x:x + w, y:y + h] = 1

# Step 2: Dummy RL Agent for Compatibility
class DummyRLAgent:
    def __init__(self, env):
        self.env = env

    def train(self, timesteps):
        print("Training not implemented. Using random actions.")

    def predict(self, obs):
        if len(self.env.rectangles_to_place) > 0:
            rect_idx = random.randint(0, len(self.env.rectangles_to_place) - 1)
            x = random.randint(0, self.env.bin_width - 1)
            y = random.randint(0, self.env.bin_height - 1)
            return (rect_idx, x, y), None
        return None, None

# Step 3: Integrate Dummy RL framework and environment
def train_packing():
    bin_width, bin_height = 80, 40
    rectangles = [(10, 15), (15, 20), (5, 5), (25, 10), (30, 5)]

    env = RectanglePackingEnv(bin_width, bin_height, rectangles)
    agent = DummyRLAgent(env)
    agent.train(timesteps=10000)

    # Visualize results
    obs = env.reset()
    done = False

    while not done:
        action, _ = agent.predict(obs)
        obs, reward, done, _ = env.step(action)

    visualize_packing(env.state)

# Step 4: Visualize the packing
def visualize_packing(state):
    plt.imshow(state, cmap='gray')
    plt.title("Final Packing")
    plt.show()

if __name__ == "__main__":
    train_packing()

