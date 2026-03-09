import torch.nn as nn

class EconomyHead(nn.Module):
    def __init__(self, input_dim=256, worker_dim=10, mine_dim=2, trade_dim=6, warehouse_dim=2, crop_dim=2):
        super().__init__()

        
        # We create distinct pathways
        self.worker_logic = nn.Sequential(nn.Linear(input_dim, worker_dim), nn.LayerNorm(worker_dim))
        self.mine_logic    = nn.Sequential(nn.Linear(input_dim, mine_dim), nn.LayerNorm(mine_dim))
        self.trade_logic   = nn.Sequential(nn.Linear(input_dim, trade_dim), nn.LayerNorm(trade_dim))
        self.warehouse_logic = nn.Sequential(nn.Linear(input_dim, warehouse_dim), nn.LayerNorm(warehouse_dim))
        self.crop_logic = nn.Sequential(nn.Linear(input_dim, crop_dim), nn.LayerNorm(crop_dim))

    def forward(self, x):
        # Each head handles its own numerical scale
        return (
            self.worker_logic(x),
            self.mine_logic(x),
            self.trade_logic(x),
            self.warehouse_logic(x),
            self.crop_logic(x)
        )
