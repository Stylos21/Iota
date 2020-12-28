from Iota import Iota
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy 
import json
# model = DQN.load('pp')
env = Iota()
model = DQN(MlpPolicy, env, verbose=1).learn(total_timesteps=25000)
model.save('DQN')

with open('agentDirectionHistory.json', 'w') as f:
    f.write(json.dumps(env.agent.direction_history))

with open('agentDistances.json', 'w') as f:
    f.write(json.dumps(env.agent.distances))

with open('collisionHistory.json', 'w') as f:
    f.write(json.dumps(env.agent.collisions))

with open('rewardLogs.json', 'w') as f:
    f.write(json.dumps(env.agent.rewards))
# model.load('pp')
    

