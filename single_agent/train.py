import os
import copy
import json

import numpy as np
import gym

from torch.utils.tensorboard import SummaryWriter

from rl_agents.DDPG import Agent

# Init. tensorboard summary writer
tb = SummaryWriter(log_dir=os.path.abspath('data/tensorboard/'))


if __name__ == '__main__':

    # Init. Environment
    env = gym.make('Pendulum-v1')
    env.reset()

    # Init. Datapath
    data_path = os.path.abspath('data')

    # Init. Training
    best_score = -np.inf
    score_history = []
    avg_history = []
    distance_history = []
    n_games = 1000
    logging_info = []

    # Init. Agent
    agent = Agent(env=env, datapath=data_path, n_games=n_games)

    for i in range(n_games):
        score = 0
        done = False

        # Initial Reset of Environment
        state = env.reset()

        while not done:
            # Choose agent based action & make a transition
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.memory.add(state, action, reward, next_state, done)

            state = copy.deepcopy(next_state)
            score += reward

            # Optimize the agent
            agent.optimize()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            print(f'Episode:{i}'
                  f'\t ACC. Rewards: {score:3.2f}'
                  f'\t AVG. Rewards: {avg_score:3.2f}'
                  f'\t *** MODEL SAVED! ***')
        else:
            print(f'Episode:{i}'
                  f'\t ACC. Rewards: {score:3.2f}'
                  f'\t AVG. Rewards: {avg_score:3.2f}')

        episode_info = {
            'Episode': i,
            'Total Episodes': n_games,
            'Epidosic Summed Rewards': score,
            'Moving Mean of Episodic Rewards': avg_score
        }

        logging_info.append(episode_info)

        # Add info. to tensorboard
        tb.add_scalars('training_rewards', {'Epidosic Summed Rewards': score,
                       'Moving Mean of Episodic Rewards': avg_score, }, i)

        # Dump .json
        with open(os.path.join(data_path, 'training_info.json'), 'w', encoding='utf8') as file:
            json.dump(logging_info, file, indent=4, ensure_ascii=False)

    # Close tensorboard writer
    tb.close()
