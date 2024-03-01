from agent import Agent
import numpy as np

def default_policy(agent: Agent) -> str:

    reward = agent.known_rewards.astype(int)
    position = agent.position
    print("rewards", reward)
    print("agent", position)

    non_zero_positions = np.where(reward != 0)[0]

    if len(non_zero_positions) = 0:
        next_position = np.argmax(np.abs(np.arange(len(reward)) - position))

        if next_position < position:
            action = "left"
        elif next_position > position:
            action = "right"
        else:
            action = "none"
    
    else:

        last_reward_index = np.where(agent.known_rewards != 0)[0][-1]

        if last_reward_index in non_zero_positions:
            if last_reward_index < position:
                action = "left"
            elif last_reward_index > position:
                action = "right"
            else:
                action = "none"
        else:

            next_position = non_zero_positions[0]

            if next_position < position:
                action = "left"
            elif next_position > position:
                action = "right"
            else:
                action = "none"
    
    return action