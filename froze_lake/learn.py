import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human")

state = env.reset()[0]
num_actions = env.action_space.n
num_states = env.observation_space.n
P = env.unwrapped.P

def get_terminal_states():
    terminal_states = [s for s in P if all(done for _,_,_,done in sum(P[s].values(), []))]
    return terminal_states

terminal_states_list = get_terminal_states()

def compute_action_value(state, action, V, gamma=0.99):
    if state in terminal_states_list:
        return 0
    transitions = P[state][action]
    q_value = sum(prob * (reward + gamma * V[next_state]) 
                  for prob, next_state, reward, _ in transitions)
    return q_value

def value_iteration(max_iterations=1000, theta=1e-6, gamma=0.99):
    V = {s: 0.0 for s in range(num_states)}
    policy = {s: 0 for s in range(num_states)}
    
    for iteration in range(max_iterations):
        delta = 0
        
        for state in range(num_states):
            if state in terminal_states_list:
                continue
            
            old_value = V[state]
            Q_values = [compute_action_value(state, action, V, gamma) 
                       for action in range(num_actions)]
            V[state] = max(Q_values)
            policy[state] = np.argmax(Q_values)
            
            delta = max(delta, abs(old_value - V[state]))
        
        if delta < theta:
            print(f"Value iteration converged at iteration {iteration}")
            break
    
    return policy, V

def policy_iteration(max_iterations=100, gamma=0.99):
    V = {s: 0.0 for s in range(num_states)}
    policy = {s: 0 for s in range(num_states)}
    
    for iteration in range(max_iterations):
        # Policy Evaluation
        for _ in range(10):
            for state in range(num_states):
                if state in terminal_states_list:
                    continue
                
                action = policy[state]
                transitions = P[state][action]
                V[state] = sum(prob * (reward + gamma * V[next_state]) 
                              for prob, next_state, reward, _ in transitions)
        
        # Policy Improvement
        policy_stable = True
        for state in range(num_states):
            if state in terminal_states_list:
                continue
            
            old_action = policy[state]
            Q_values = [compute_action_value(state, action, V, gamma) 
                       for action in range(num_actions)]
            policy[state] = np.argmax(Q_values)
            
            if old_action != policy[state]:
                policy_stable = False
        
        if policy_stable:
            print(f"Policy iteration converged at iteration {iteration}")
            break
    
    return policy, V

def test_policy(policy, num_episodes=10, render=True):
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}, Steps = {steps}")
    
    return episode_rewards


policy_vi, V_vi = value_iteration()


rewards_vi = test_policy(policy_vi, num_episodes=5, render=True)
print(f"Average Reward (Value Iteration): {np.mean(rewards_vi):.2f}")



policy_pi, V_pi = policy_iteration()


rewards_pi = test_policy(policy_pi, num_episodes=5, render=True)
print(f"Average Reward (Policy Iteration): {np.mean(rewards_pi):.2f}")

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(rewards_vi, 'o-', label='Value Iteration', linewidth=2, markersize=8)
plt.plot(rewards_pi, 's-', label='Policy Iteration', linewidth=2, markersize=8)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.title('FrozenLake-8x8 Performance Comparison', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

env.close()