import gymnasium as gym
import random
import sys
import os

# Add the parent directory to sys.path to allow importing 'enviroment'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import enviroment.strategy_env # This line ensures the environment is registered

def test_environment():
    """
    Creates an instance of the StrategyProblem-v0 environment,
    runs it for a few steps with random actions, and prints the output.
    """
    try:
        env = gym.make("StrategyProblem-v0")
        print("Environment created successfully.")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    observation, info = env.reset()
    print("\n--- Initial State ---")
    print("Observation:", observation)
    print("Info:", info)
    env.render()

    for i in range(10): # Run for 10 steps
        print(f"\n--- Step {i+1} ---")
        # Sample random actions for both diplomacy and economy branches
        diplomacy_action = env.action_space["diplomacy"].sample()
        economy_action = env.action_space["economy"].sample()
        action = {"diplomacy": diplomacy_action, "economy": economy_action}

        print("Action taken:", action)

        observation, reward, terminated, truncated, info = env.step(action)
        print("Observation:", observation)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        print("Info:", info)
        env.render()

        if terminated or truncated:
            print("\nEpisode finished.")
            break
    
    env.close()
    print("\nEnvironment closed.")

if __name__ == "__main__":
    test_environment()
