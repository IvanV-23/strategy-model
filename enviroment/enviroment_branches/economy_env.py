from  enviroment.enviroment_blocks import player_env
from  enviroment.enviroment_blocks import opponent_env
from  enviroment.enviroment_blocks import board_env

class EconomyEnv:
    def __init__(self, player_env: player_env.PlayerEnv, board_env: board_env.BoardEnv, opponent_env: opponent_env.OpponentEnv):
        self.player = player_env
        self.opponent = opponent_env
        self.board = board_env

    def _build_soldiers(self, new_soldiers: int):
        reward = 0
        if new_soldiers > 0:
            gold_cost = 500 * new_soldiers

            wood_cost = 250 * new_soldiers
            
            if self.player.resources[0] >= gold_cost and self.player.resources[1] >= wood_cost:
                self.player.resources[0] -= gold_cost
                self.player.resources[1] -= wood_cost
                self.board.spawn_soldiers(player_id=1, count=new_soldiers)
                # print(f"Recruited {new_soldiers} soldiers.")
                reward += 0.01 * new_soldiers
            else:
                reward -= 0.05
        return reward
    
    def _build_mines(self, mine_action: int):
        """
        mine_action: 0: Nothing, 1: Build L1, 2: Upgrade to L2
        """
        reward = 0
        if mine_action == 1: # Build Lvl 1
            gold_cost, wood_cost = 50, 0
            if self.player.resources[0] >= gold_cost and self.player.resources[1] >= wood_cost:
                success, msg = self.board.p1_buildings_manager.build_mine(1, 1)              
                if success:
                    self.player.resources[0] -= gold_cost
                    self.player.resources[1] -= wood_cost
                    reward += 0.5
                    print(f"Built mine. {msg}")
                else:
                    reward -= 0.1
            else:
                reward -= 0.1

        elif mine_action == 2: # Upgrade to Lvl 2
            gold_cost, wood_cost = 100, 50
            if self.player.resources[0] >= gold_cost and self.player.resources[1] >= wood_cost:
                success, msg = self.board.p1_buildings_manager.update_mine(player_id=1)
                if success:
                    self.player.resources[0] -= gold_cost
                    self.player.resources[1] -= wood_cost
                    reward += 1.0
                    print(f"Upgraded mine to Lvl 2. {msg}")
                else:
                    reward -= 0.1
            else:
                reward -= 0.1

        return reward
    
    def _build_trade_routes(self, trade_route_action: int):
        reward = 0
        if trade_route_action == 1: 
            success, msg = self.board.create_trade_route(player_id=1)
            if success:
                # print("Successfully built a trade route.")
                reward += 0.5
            else:
                reward -= 0.01
        return reward

    def _build_warehouse(self, warehouse_action: int):
            """
            warehouse_action: 0: Nothing, 1: Build L1, 2: Upgrade to L2
            """
            reward = 0
            if warehouse_action == 1: # Build Lvl 1
                gold_cost, wood_cost = 30, 20
                if self.player.resources[0] >= gold_cost and self.player.resources[1] >= wood_cost:
                    success, msg = self.board.p1_buildings_manager.build_warehouse(player_id=1)
                    if success:
                        self.player.resources[0] -= gold_cost
                        self.player.resources[1] -= wood_cost
                        self.player.capacity[1] += 500 
                        self.player.capacity[0] += 250
                        self.player.capacity[3] += 1000
                        print(f"Built warehouse. {msg}")
                        reward += 0.4
                    else:
                        reward -= 0.05
                else:
                    reward -= 0.1

            elif warehouse_action == 2: # Upgrade to Lvl 2
                gold_cost, wood_cost = 60, 40
                if self.player.resources[0] >= gold_cost and self.player.resources[1] >= wood_cost:
                    success, msg = self.board.p1_buildings_manager.update_warehouse(player_id=1)
                    if success:
                        self.player.resources[0] -= gold_cost
                        self.player.resources[1] -= wood_cost
                        # Double the capacity bonus of a warehouse (adding another set)
                        self.player.capacity[1] += 1000 
                        self.player.capacity[0] += 500
                        self.player.capacity[3] += 2000
                        print(f"Upgraded warehouse to Lvl 2. {msg}")
                        reward += 0.8
                    else:
                        reward -= 0.05
                else:
                    reward -= 0.1
                
            return reward

    def _build_crop_field(self, crop_field_action: int):
        """
        crop_field_action: 0: Nothing, 1: Build L1, 2: Upgrade to L2
        """
        reward = 0
        if crop_field_action == 1: # Build Lvl 1
            gold_cost, wood_cost = 20, 10
            if self.player.resources[0] >= gold_cost and self.player.resources[1] >= wood_cost:
                success, msg = self.board.p1_buildings_manager.build_crop_field(player_id=1)
                if success:
                    self.player.resources[0] -= gold_cost
                    self.player.resources[1] -= wood_cost
                    print(f"Built crop field. {msg}")
                    reward += 0.3
                else:
                    reward -= 0.05 
            else:
                reward -= 0.1

        elif crop_field_action == 2: # Upgrade to Lvl 2
            gold_cost, wood_cost = 40, 20
            if self.player.resources[0] >= gold_cost and self.player.resources[1] >= wood_cost:
                success, msg = self.board.p1_buildings_manager.update_crop_field(player_id=1)
                if success:
                    self.player.resources[0] -= gold_cost
                    self.player.resources[1] -= wood_cost
                    print(f"Upgraded crop field to Lvl 2. {msg}")
                    reward += 0.6
                else:
                    reward -= 0.05
            else:
                reward -= 0.1
            
        return reward
    
    def execute_economy_action(self,
                                new_soldiers: int, 
                                mine_action: int,
                                trade_route_action: int,
                                warehouse_action: int,
                                crop_field_action: int = 0
                                ):

        reward = self._build_soldiers(new_soldiers)
        reward += self._build_mines(mine_action)
        reward += self._build_trade_routes(trade_route_action)
        reward += self._build_warehouse(warehouse_action)
        reward += self._build_crop_field(crop_field_action)

        return {"reward": reward}
