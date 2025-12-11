import wandb
import random


# 1. Start a W&B run
wandb.init(project='gpt3', mode='offline')


# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01


# 3. Log metrics over time to visualize performance
for i in range (10):
    wandb.log({"loss": random.random() - i })
