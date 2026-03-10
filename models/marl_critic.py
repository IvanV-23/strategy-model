import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicBackbone(nn.Module):
    def __init__(self, in_channels=8, out_features=64):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_features, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        features = self.conv_block(x) 
        pooled = self.gap(features)   
        return features, pooled.view(pooled.size(0), -1) 

class EconomicAgent(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.soldiers_head = nn.Linear(128, 10)
        self.mines_head = nn.Linear(128, 3)
        self.trade_head = nn.Linear(128, 2)
        self.warehouse_head = nn.Linear(128, 3)
        self.crop_head = nn.Linear(128, 3)
        self.fortify_head = nn.Linear(128, 2) # NEW: 0 or 1
        
    def forward(self, x):
        x = self.common(x)
        return self.soldiers_head(x), self.mines_head(x), self.trade_head(x), \
               self.warehouse_head(x), self.crop_head(x), self.fortify_head(x)

class MilitaryAgent(nn.Module):
    def __init__(self, spatial_channels, context_dim):
        super().__init__()
        self.target_head = nn.Conv2d(spatial_channels, 1, kernel_size=1)

    def forward(self, spatial_features, common_features):
        target_logits = self.target_head(spatial_features)
        return target_logits.view(target_logits.size(0), -1)

class MARL_Strategy(nn.Module):
    def __init__(self, in_channels=8, stats_dim=27):
        super().__init__()
        self.backbone = DynamicBackbone(in_channels, out_features=64)
        self.spatial_norm = nn.LayerNorm(64)
        self.stats_norm = nn.LayerNorm(stats_dim)
        self.context_input_dim = 64 + stats_dim
        self.fc_shared = nn.Sequential(
            nn.Linear(self.context_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        self.eco_agent = EconomicAgent(256)
        self.mil_agent = MilitaryAgent(64, 256)
        self.dip_head = nn.Linear(256, 3)
        self.global_critic = nn.Sequential(
            nn.Linear(self.context_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, board_state, global_stats, target_mask=None, build_mask=None):
        spatial_features, pooled_features = self.backbone(board_state)
        combined_context = torch.cat([self.spatial_norm(pooled_features), self.stats_norm(global_stats)], dim=1)
        shared_latent = self.fc_shared(combined_context)
        
        workers_l, mines_l, trade_l, wh_l, crop_l, fort_l = self.eco_agent(shared_latent)
        mil_target = self.mil_agent(spatial_features, shared_latent)
        dip_logits = self.dip_head(shared_latent)
        
        if target_mask is not None:
            mil_target = mil_target.masked_fill(~target_mask.bool(), -1e4)
        
        if build_mask is not None:
            # Mask: [MineL1, MineL2, Trade, WH_L1, WH_L2, CropL1, CropL2, Fortify]
            m_mask = torch.ones_like(mines_l).bool()
            m_mask[:, 1], m_mask[:, 2] = build_mask[:, 0].bool(), build_mask[:, 1].bool()
            mines_l = mines_l.masked_fill(~m_mask, -1e4)
            
            t_mask = torch.ones_like(trade_l).bool()
            t_mask[:, 1] = build_mask[:, 2].bool()
            trade_l = trade_l.masked_fill(~t_mask, -1e4)
            
            w_mask = torch.ones_like(wh_l).bool()
            w_mask[:, 1], w_mask[:, 2] = build_mask[:, 3].bool(), build_mask[:, 4].bool()
            wh_l = wh_l.masked_fill(~w_mask, -1e4)

            c_mask = torch.ones_like(crop_l).bool()
            c_mask[:, 1], c_mask[:, 2] = build_mask[:, 5].bool(), build_mask[:, 6].bool()
            crop_l = crop_l.masked_fill(~c_mask, -1e4)

            f_mask = torch.ones_like(fort_l).bool()
            f_mask[:, 1] = build_mask[:, 7].bool()
            fort_l = fort_l.masked_fill(~f_mask, -1e4)
            
        value = self.global_critic(combined_context)
        return {
            "eco": (workers_l, mines_l, trade_l, wh_l, crop_l, fort_l),
            "mil": mil_target,
            "dip": dip_logits,
            "value": value
        }
