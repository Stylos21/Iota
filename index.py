from Iota import Iota
from stable_baselines import DQN, A2C
# from stable_baselines.common.policies import MlpPolicy 
import json
model = A2C.load('50000')
env = Iota(model)
# model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./DQN_IOTA/Test2")
# model.learn(total_timesteps=50000)
# model.save('50000')

# with open('agentDirectionHistory.json', 'w') as f:
#     f.write(json.dumps(env.agent.direction_history))

# with open('agentDistances.json', 'w') as f:
#     f.write(json.dumps(env.agent.distances))

# with open('collisionHistory.json', 'w') as f:
#     f.write(json.dumps(env.agent.collisions))

# with open('rewardLogs.json', 'w') as f:
#     f.write(json.dumps(env.agent.rewards))
# # model.load('pp')
    

