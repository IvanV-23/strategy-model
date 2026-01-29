from gymnasium.envs.registration import register

register(
     id="StrategyProblem-v0",
     entry_point="enviroment.strategy_env:StrategyEnv",
     max_episode_steps=100,
     reward_threshold=10.0,
)
