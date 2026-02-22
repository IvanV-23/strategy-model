from  enviroment.enviroment_blocks import player_env
from  enviroment.enviroment_blocks import opponent_env
from  enviroment.enviroment_blocks import board_env

class EconomyEnv:
    def __init__(self, player_env: player_env.PlayerEnv, board_env: board_env.BoardEnv, opponent_env: opponent_env.OpponentEnv):
        self.player = player_env
        self.opponent = opponent_env
        self.board = board_env

    def _build_soldiers(self, new_soldiers:int):
        reward = 0
        eco_reward = self.player.process_economy(
            num_soldiers=new_soldiers, 
            num_mines=0
        )
        reward += eco_reward

        return reward
    
    def _build_mines(self, new_mines:int):

        reward = 0
        
        costs = [(0,0), (50,0)]
        gold_cost, wood_cost = costs[new_mines]

        # 2. Execute build if player has resources
        if self.player.resources[0] >= gold_cost and self.player.resources[1] >= wood_cost:
            success, msg = self.board.build_mine(1, new_mines)              
            if success and new_mines > 0:
                self.player.resources[0] -= gold_cost
                self.player.resources[1] -= wood_cost
                reward += 0.5 * new_mines  
                self.player.resources[3] += 1 * new_mines
                print(f"Built {new_mines} mines. Reward: {reward}")
            if not success:
                print(f"Failed to build mines: {msg}")
                reward -= 0.1

        return reward
    
    def _build_trade_routes(self,trade_route_action:int):
        reward = 0
        if trade_route_action == 1: # 1 means "Build Route"
            success, msg = self.board.create_trade_route(player_id=1)
            if success:
                print("Successfully built a trade route.")
                reward += 0.5
            else:
                print(f"Failed to build trade route: {msg}")
                reward -= 0.01
        return reward

    
    def execute_economy_action(self, new_soldiers:int, new_mines:int, trade_route_action:int):

        reward = self._build_soldiers(new_soldiers)

        reward = self._build_mines(new_mines)

        reward = self._build_trade_routes(trade_route_action)


        return {"reward":reward}

    

