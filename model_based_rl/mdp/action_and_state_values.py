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
state_value = {state: compute_state_value(state) 
     for state in range(num_states)}

def compute_action_value(state, action, gamma=0.90, **kwargs):
    if state == terminal_state:
        return 0
    _, next_state, reward, _ = P[state][action][0]
    return reward + gamma * state_value[next_state]

Q_values = {(state, action): compute_action_value(state, action) 
     for state in range(num_states) 
     for action in range(num_actions)}

improved_policy = {}
for state in range(num_states -1):
    max_action = max(range(num_actions), key=lambda action: Q_values[(state, action)])
    improved_policy[state] = max_action

#policy iteration

policy = {0:1, 1:2, 2:1, 3:1, 4:3, 5:1, 6:2, 7:3}

def policy_evaluation(policy):
    V = {state: compute_state_value (state, policy) 
         for state in range(num_states)}
    return V

def policy_improvement(policy):
    improved_policy = {s: 0 for s in range(num_states - 1)}
    Q_values = {(state, action): compute_action_value(state, action, policy=policy) 
         for state in range(num_states) 
         for action in range(num_actions)}
    
    for state in range(num_states - 1):
        max_action = max(range(num_actions), key=lambda action: Q_values[(state, action)])
        improved_policy[state] = max_action
    return improved_policy

def policy_iteration(policy):
    policy = policy.copy()
    while True:
        V = policy_evaluation(policy)
        improved_policy = policy_improvement(policy)
        if improved_policy == policy:
            break
        policy = improved_policy
    return policy, V