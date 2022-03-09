import os
import json

import numpy as np
import matplotlib.pyplot as plt

# Init. path
data_path = os.path.abspath('data')

with open(os.path.join(data_path, 'training_info.json')) as f:
    json_data = json.load(f)

# Load all the data frames
score = [data["Epidosic Summed Rewards"] for data in json_data]
average = [data["Moving Mean of Episodic Rewards"] for data in json_data]

# Generate graphs
plt.figure(1)
plt.plot(score, alpha=0.5, label='Episodic summed rewards')
plt.plot(average, label='Moving mean')
plt.grid(True)
plt.xlabel('Training Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.title('Training Profile')
plt.savefig(os.path.join(data_path, 'Training_Profile.png'))
