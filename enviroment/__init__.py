from gymnasium.envs.registration import register

register(
     id="StrategyProblem-v0",
     entry_point="enviroment.strategy_env:StrategyEnv",
     max_episode_steps=1000,
     reward_threshold=10.0,
)

try:
    from enviroment.native_strategy_env import NativeStrategyEnv
    register(
        id="NativeStrategyProblem-v0",
        entry_point="enviroment.native_strategy_env:NativeStrategyEnv",
        max_episode_steps=1000,
        reward_threshold=10.0,
    )
    NATIVE_ENV_AVAILABLE = True
except ImportError:
    NATIVE_ENV_AVAILABLE = False
