from model_based_rl import env

env_instance, num_actions, num_states = env("FrozenLake-v1")
state = env_instance.observation_space.sample()

def transition_probs():
    for action in range(num_actions):
        transitions = env_instance.unwrapped.P[state][action]
        for transition in transitions:
            probability, next_state, reward, done = transition
            print(f"Probability: {probability}, Next State: {next_state}, Reward: {reward}, Done: {done}")
            return probability, next_state, reward, done
