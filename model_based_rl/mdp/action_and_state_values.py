from model_based_rl import env
import random


env_instance, num_actions, num_states = env("FrozenLake-v1")
P = env_instance.unwrapped.P
terminal_states = [s for s in P if all(done for _,_,_,done in sum(P[s].values(), []))]
terminal_state = random.choice(terminal_states)

def compute_state_value(state, policy, gamma=0.90):
    if state == terminal_state:
        return 0
    action = policy[state]
    _, next_state, reward, _ = P[state][action][0]
    return reward + gamma * compute_state_value(next_state)
V = {state: compute_state_value(state) 
     for state in range(num_states)}