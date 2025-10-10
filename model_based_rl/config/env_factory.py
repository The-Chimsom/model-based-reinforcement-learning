import gymnasium as gym

def env(env_name: str, render_mode: str ="rgb_array", **kwargs):
    env_instance = gym.make(env_name, render_mode = render_mode,**kwargs)
    num_actions = env_instance.observation_space.n
    num_states = env_instance.observation_space.n
    return env_instance, num_actions, num_states