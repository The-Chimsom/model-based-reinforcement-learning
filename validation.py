import gymnasium as gym

env = gym.make("MountainCarContinuous-v0", render_mode="human")

state, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()


    state, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        state, info = env.reset()

env.close()