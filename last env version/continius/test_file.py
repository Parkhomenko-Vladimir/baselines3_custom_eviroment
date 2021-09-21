import os
from stable_baselines3 import DQN, PPO, TD3
from CustomEnv import CustomEnv
from CustomCNN import CustomCNN
from SaveOnBestTrainingRewardCallback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor
import torch
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim = 518),
    activation_fn=torch.nn.ReLU,
    net_arch = [dict(pi=[518,128, 32, 2], vf=[518,128, 32, 2])]
)

env = CustomEnv(obstacle_turn = False,
                vizualaze     = False,
                Total_war     = True,
                head_velocity = 0.01,
                steps_limit   = 2000)

log_dir = './saved_models_disc_mult/PPO_T_with_additional_reward/'
os.makedirs(log_dir, exist_ok=True)

callback = SaveOnBestTrainingRewardCallback(check_freq  = 1000,
                                            log_dir     = log_dir,
                                            agent_name  = 'PPO_T_with_additional_reward')

env = Monitor(env, log_dir)

model = PPO(policy          = 'MlpPolicy',
            env             = env,
            learning_rate   = 0.0001,
            n_steps         = 2048,
            batch_size      = 24,
            tensorboard_log = "./tensorboard_logs_disc_mult/",
            policy_kwargs   = policy_kwargs,
            verbose         = 1,
            device          = 'cuda')

obs = env.reset()
action, _ = model.predict(obs)
# model.learn(total_timesteps=1e6,callback=callback)