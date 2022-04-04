import os
import numpy as np
import json

import gym

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import torch

from DDPG import Agent


def predict_value(agent: Agent, pca: PCA, state: np.ndarray) -> float:
    position = np.ravel(pca.inverse_transform(state[0]))
    velocity = state[1]
    with torch.no_grad():
        state = torch.as_tensor(np.hstack((position, velocity)), device=agent.actor.device)
        action = torch.as_tensor(agent.choose_action(state), device=agent.actor.device)
        value = agent.critic(state, action)
    return value.item()


if __name__ == '__main__':

    # Init. path
    data_path = os.path.abspath('single_agent/data')

    with open(os.path.join(data_path, 'training_info.json')) as f:
        train_data = json.load(f)

    with open(os.path.join(data_path, 'testing_info.json')) as f:
        test_data = json.load(f)

    # Load all the data frames
    score = [data["Epidosic Summed Rewards"] for data in train_data]
    average = [data["Moving Mean of Episodic Rewards"] for data in train_data]
    test = [data["Test Score"] for data in test_data]

    # Gather data for value plots
    env = gym.make('Pendulum-v1')
    env.reset()

    agent = Agent(env=env, datapath=data_path, training=False)
    agent.load_models()

    state = np.vstack(np.linspace(env.observation_space.low, env.observation_space.high, 500))
    pca = PCA(n_components=1)
    positions = np.ravel(pca.fit_transform(state[:, :2]))
    assert pca.explained_variance_ratio_[0] == 1.0

    x, y = np.meshgrid(positions, state[:, 2])
    z = np.apply_along_axis(lambda _: predict_value(agent, pca, _), 2, np.dstack([x, y]))
    z = z[:-1, :-1]
    z_min, z_max = z.min(), z.max()

    # Generate graphs
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    axes[0].plot(score, alpha=0.5, label='Episodic summation')
    axes[0].plot(average, label='Moving mean of 100 episodes')
    axes[0].grid(True)
    axes[0].set_xlabel('Training Episodes')
    axes[0].set_ylabel('Rewards')
    axes[0].legend(loc='best')
    axes[0].set_title('Training Profile')

    axes[1].boxplot(test)
    axes[1].grid(True)
    axes[1].set_xlabel('Test Run')
    axes[1].set_title('Testing Profile')

    axes[2].pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    axes[2].axis([x.min(), x.max(), y.min(), y.max()])
    axes[2].set_xlabel('Position - Principal Axes 1 (VAR = 1.0)')
    axes[2].set_ylabel('Velocity')
    axes[2].set_title("State-Action Value Estimation")
    fig.colorbar(axes[2].pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max))

    fig.tight_layout()
    plt.savefig(os.path.join(data_path, 'DDPG Agent Profiling.png'))
