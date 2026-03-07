import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicBackbone(nn.Module):
    def __init__(self, in_channels=5, out_features=64):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_features, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # GAP layer: This turns (Batch, 64, H, W) -> (Batch, 64, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        features = self.conv_block(x) 
        pooled = self.gap(features)   
        return features, pooled.view(pooled.size(0), -1) 

class EconomicAgent(nn.Module):
    """Focuses on Workers, Mines, Trade, Warehouses, and Crop Fields."""
    def __init__(self, input_dim):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.workers_head = nn.Linear(128, 10)
        self.mines_head = nn.Linear(128, 6)
        self.trade_head = nn.Linear(128, 6)
        self.warehouse_head = nn.Linear(128, 2)
        self.crop_head = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.common(x)
        workers = self.workers_head(x)
        mines = self.mines_head(x)
        trade = self.trade_head(x)
        warehouse = self.warehouse_head(x)
        crop = self.crop_head(x)
        return workers, mines, trade, warehouse, crop

class MilitaryAgent(nn.Module):
    """Focuses on Spatial Targeting."""
    def __init__(self, spatial_channels, context_dim):
        super().__init__()
        self.target_head = nn.Conv2d(spatial_channels, 1, kernel_size=1)

    def forward(self, spatial_features, common_features):
        target_logits = self.target_head(spatial_features)
        return target_logits.view(target_logits.size(0), -1)

class MARL_Strategy(nn.Module):
    def __init__(self, in_channels=5, stats_dim=27):
        super().__init__()
        self.backbone = DynamicBackbone(in_channels, out_features=64)
        
        # Norm layers for stability
        self.spatial_norm = nn.LayerNorm(64)
        self.stats_norm = nn.LayerNorm(stats_dim)
        
        # Shared context: 64 (from Conv out_features) + stats
        self.context_input_dim = 64 + stats_dim
        self.fc_shared = nn.Sequential(
            nn.Linear(self.context_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # Specialized Agents
        self.eco_agent = EconomicAgent(256)
        self.mil_agent = MilitaryAgent(64, 256)
        self.dip_head = nn.Linear(256, 3) # Matches diplomacy_action_space (3)
        
        # Centralized Critic
        self.global_critic = nn.Sequential(
            nn.Linear(self.context_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, board_state, global_stats, target_mask=None, build_mask=None):
        # 1. Perception
        spatial_features, pooled_features = self.backbone(board_state)
        
        # Stability Norms
        pooled_features = self.spatial_norm(pooled_features)
        global_stats = self.stats_norm(global_stats)
        
        # 2. Shared Communication
        combined_context = torch.cat([pooled_features, global_stats], dim=1)
        shared_latent = self.fc_shared(combined_context)
        
        # 3. Agent Decisions
        workers_l, mines_l, trade_l, wh_l, crop_l = self.eco_agent(shared_latent)
        mil_target = self.mil_agent(spatial_features, shared_latent)
        dip_logits = self.dip_head(shared_latent)
        
        # 4. Masking
        if target_mask is not None:
            mil_target = mil_target.masked_fill(~target_mask.bool(), -1e4)
        
        if build_mask is not None:
            # Mines masking (indices 0-5 in build_mask map to mines categories)
            mines_l = mines_l.masked_fill(~build_mask[:, :6].bool(), -1e4)
            
            # Trade masking (index 6 in build_mask)
            trade_mask = torch.ones_like(trade_l).bool()
            trade_mask[:, 1] = build_mask[:, 6].bool()
            trade_l = trade_l.masked_fill(~trade_mask, -1e4)
            
            # Warehouse masking (index 7 in build_mask)
            wh_mask = torch.ones_like(wh_l).bool()
            wh_mask[:, 1] = build_mask[:, 7].bool()
            wh_l = wh_l.masked_fill(~wh_mask, -1e4)

            # Crop field masking (index 8 in build_mask)
            crop_mask = torch.ones_like(crop_l).bool()
            crop_mask[:, 1] = build_mask[:, 8].bool()
            crop_l = crop_l.masked_fill(~crop_mask, -1e4)
            
        # 5. Global Value Judgement
        value = self.global_critic(combined_context)
        
        return {
            "eco": (workers_l, mines_l, trade_l, wh_l, crop_l),
            "mil": mil_target,
            "dip": dip_logits,
            "value": value
        }
