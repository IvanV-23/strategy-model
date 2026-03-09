
from  enviroment.enviroment_blocks import player_env
from  enviroment.enviroment_blocks import opponent_env
from  enviroment.enviroment_blocks import board_env

class DiplomacyEnv:
    def __init__(self, player_env: player_env.PlayerEnv, board_env: board_env.BoardEnv, opponent_env: opponent_env.OpponentEnv):
        self.player = player_env
        self.opponent = opponent_env
        self.board = board_env
    
    def execute_diplomacy(self,diplomacy_action:int,target_row:int,target_col:int,current_turn=0):
        reward = 0

        if diplomacy_action == 0:
            action_result = self.trade()
        if diplomacy_action == 1:
            reward -= (1000 - current_turn) * 0.01
            action_result =    {"reward":reward,
                                "history":f"Pass",
                                "terminated":False,
                                "defeated_workers": 0
                                }
        if diplomacy_action == 2:
            action_result = self.attack(target_col=target_col,target_row=target_row)

        if "defeated_workers" not in action_result:
            action_result["defeated_workers"] = 0

        return action_result



    def trade(self): 
        """Performs a trade action to gain resources."""
        reward = 0.0
        trade_routes = len(self.board.p1_trade_manager.active_routes)
        if trade_routes == 0:
            reward -= 0.01
            return {"reward":reward,
                    "history":f"Fail trade",
                    "truncated":False,
                    "defeated_workers": 0
                    }
        if self.player._resources[1] >= trade_routes*50:
            self.player._resources[1] -= trade_routes*50
            self.player._resources[0] += trade_routes*25
            reward += 0.1*trade_routes*25
        else:
            reward -= 0.01
        return {"reward":reward,
                "history":f"Traded ({trade_routes*50}x{trade_routes*25})",
                "truncated":False,
                "defeated_workers": 0
                }

    def attack(self,target_row,target_col):
        terminated = False
        reward = 0
        # 1. Ask the board to resolve combat based on spatial worker distribution
        # Note: claim_adjacent_tile now handles 'Attack Power > Defense' internally
        victory, base_captured, prev_owner, reason, defeated_workers = self.board.claim_target_tile(1, (target_row, target_col))
        
        res_msg = "VICTORY" if victory else "FAILED"
        
        
        # 2. Update Player 1 resources based on result
        attack_reward = self.player.process_battle_consequences(victory, base_captured, prev_owner, reason, self.board.get_owned_tiles(owner_id=1))
        reward += attack_reward

        # 3. Update Opponent resources if they lost
        if victory:
            # If P1 won, P2 loses workers and resources
            self.opponent.resources[0] = max(0, self.opponent.resources[0] - 50)
            self.opponent.resources[1] = max(0, self.opponent.resources[1] - 25)
            
            # Deduct the actual workers lost on the tile from the opponent's pool
            self.opponent.resources[2] = int(max(0, self.opponent.resources[2] - defeated_workers))
            reward += defeated_workers * 0.01  

        # 4. Handle game termination
        if base_captured:
            terminated = True
            print(f"Opponent defeated! Total Reward: {reward}")

        return {"reward":reward,
                "history":f"Attack on ({target_row},{target_col}): {res_msg}",
                "terminated":terminated,
                "defeated_workers": defeated_workers
                }
