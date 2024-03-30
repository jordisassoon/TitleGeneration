import gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from transformers import T5Tokenizer, T5ForConditionalGeneration



class TitleGenerationEnv(gym.Env):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.observation_space = gym.spaces.Discrete(len(self.data))
        self.action_space = gym.spaces.Discrete(2)  # 0: generate title, 1: stop episode
        self.current_observation = None
        self.current_episode = None
        self.current_step = None
        self.current_generated_title = None
        self.current_actual_title = None

    def reset(self):
        self.current_episode = self.data.sample()
        self.current_observation = self.current_episode['text'].values[0]
        self.current_step = 0
        self.current_generated_title = ''
        self.current_actual_title = self.current_episode['titles'].values[0]
        return self.current_observation

    def step(self, action):
        if action == 0:
            input_ids = self.tokenizer.encode(self.current_observation, return_tensors='pt')
            output = self.model.generate(input_ids)
            self.current_generated_title = self.tokenizer.decode(output[0], skip_special_tokens=True)
            self.current_step += 1
            done = False
        else:
            done = True

        reward = self.calculate_reward()
        return self.current_observation, reward, done, {}

    def calculate_reward(self):
        rouge = Rouge()
        rouge_score = rouge.get_scores(self.current_generated_title, self.current_actual_title)[0]['rouge-l']['f']
        return rouge_score

data_path = '/path/to/your/dataset.csv'
env = DummyVecEnv([lambda: TitleGenerationEnv(data_path)])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("/path/to/save/model")

