from model_based_rl import env

env_instance, num_actions, num_states = env("FrozenLake-v1")
state = env_instance.observation_space.sample()

def transition_probs():
    for action in range(num_actions):
        P = env_instance.unwrapped.P
        terminal_states = [s for s in P if all(done for _,_,_,done in sum(P[s].values(), []))]
        transitions = P[state][action]
        for transition in transitions:
            probability, next_state, reward, done = transition
            print(f"Probability: {probability}, Next State: {next_state}, Reward: {reward}, Done: {done}")
            return probability, next_state, reward, done, terminal_states
transition_probs()
