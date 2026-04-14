import abc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

# =========================================================================
# 0. 门控模块与通用基础组件 (Gating & Utils)
# =========================================================================

class AdvancedLearnableGate(nn.Module):
    def __init__(self, physics_feat_dim=14, hidden_dim=112, out_dim=1): 
        super().__init__()
        input_dim = physics_feat_dim + hidden_dim + 1
        self.critic_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim) # 动态输出维度 (1 或 pred_len)
        )
        
    def forward(self, ts_physics_features, t_emb, context_res_norm):
        gate_input = torch.cat([ts_physics_features, t_emb, context_res_norm], dim=-1)
        gate_logits = self.critic_net(gate_input) 
        return gate_logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, dim_feedforward=512, layer_norm_eps=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim), nn.Dropout(dropout)
        )

    def forward(self, tgt, memory_key, memory_value):
        tgt_norm = self.norm1(tgt)
        attn_out, weights = self.attn(query=tgt_norm, key=memory_key, value=memory_value)
        tgt = tgt + self.dropout1(attn_out)
        tgt = tgt + self.ffn(self.norm2(tgt))
        return tgt, weights

class StandardTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="gelu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, tgt, memory_k, memory_v):
        tgt_norm = self.norm1(tgt)
        attn_out, weights = self.cross_attn(query=tgt_norm, key=memory_k, value=memory_v)
        tgt = tgt + self.dropout1(attn_out)
        tgt_norm2 = self.norm2(tgt)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(tgt_norm2))))
        tgt = tgt + self.dropout2(ff_out)
        return tgt, weights

class IntraSequenceFusion(nn.Module):
    def __init__(self, embed_dim, d_model):
        super().__init__()
        self.seq_proj = nn.Linear(embed_dim, d_model)
        self.err_proj = nn.Linear(embed_dim, d_model)
        self.gate_proj = nn.Linear(embed_dim * 2, d_model)
        self.out_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        
    def forward(self, seq_emb, err_emb):
        s = self.seq_proj(seq_emb)
        e = self.err_proj(err_emb)
        concat = torch.cat([seq_emb, err_emb], dim=-1)
        gate = torch.sigmoid(self.gate_proj(concat))
        fused = (s + e) * gate
        return self.out_proj(fused)

class ConfidenceScorer(nn.Module):
    def __init__(self, embed_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid() 
        )
    def forward(self, seq_emb, err_emb):
        return self.net(torch.cat([seq_emb, err_emb], dim=-1))


# =========================================================================
# 🌟 1. 抽象基类 (Interface) - [核心加装：统一门控应用系统]
# =========================================================================

class BaseCorrector(nn.Module, abc.ABC):
    """
    [接口] 残差校正器基类
    统一在此处初始化并挂载所有门控机制 (Hard / Soft / Static)。
    所有子类的 forward 最后只需调用 self.apply_gating() 即可享用最新门控！
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.pred_len = config.get("pred_len", 96)
        self.embed_dim = config.get("embed_dim", 128)
        self.top_k = config.get("top_k", 5)

        # === 统一门控策略路由 (Gating Strategy) ===
        self.gating_strategy = config.get('gating_strategy', 'none') 
        if config.get('use_learnable_gate', False) and self.gating_strategy == 'none':
            self.gating_strategy = 'learnable'

        if self.gating_strategy in ['learnable', 'soft_scalar', 'soft_vector']:
            out_dim = self.pred_len if self.gating_strategy == 'soft_vector' else 1
            self.gate_module = AdvancedLearnableGate(physics_feat_dim=14, hidden_dim=self.embed_dim, out_dim=out_dim)
        elif self.gating_strategy == 'static':
            self.static_threshold = config.get('static_threshold', 0.15)

    @abc.abstractmethod
    def forward(self, 
                target_emb: torch.Tensor, 
                retrieved_embs: torch.Tensor, 
                retrieved_residuals: torch.Tensor, 
                history: torch.Tensor = None,
                target_residual: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pass

    def get_config(self):
        return self.config

    def apply_gating(self, safe_pred_res, info_dict, retrieved_residuals, target_emb, ts_physics_features=None, target_local_res=None):
        info_dict["raw_pred_res_normalized"] = safe_pred_res.clone()

        if self.gating_strategy in ['learnable', 'soft_scalar', 'soft_vector'] and ts_physics_features is not None:
            context_res_norm = torch.mean(torch.abs(retrieved_residuals), dim=(1, 2)).unsqueeze(-1)
            t_emb_sq = target_emb.squeeze(1) if target_emb.dim() == 3 else target_emb
            
            gate_logits = self.gate_module(ts_physics_features, t_emb_sq, context_res_norm)
            gate_prob = torch.sigmoid(gate_logits) 
            
            info_dict['gate_logits'] = gate_logits 
            info_dict['gate_prob'] = gate_prob     
            
            if self.gating_strategy == 'learnable':
                hard_mask = (gate_prob >= 0.5).float()
                mask = hard_mask.detach() - gate_prob.detach() + gate_prob if self.training else hard_mask
            else:
                mask = gate_prob 
                
            safe_pred_res = safe_pred_res * mask
            
        elif self.gating_strategy == 'static' and target_local_res is not None:
            if target_local_res.dim() == 3: target_local_res = target_local_res.squeeze(1)
            res_magnitude = torch.mean(torch.abs(target_local_res), dim=-1)
            mask = (res_magnitude >= getattr(self, 'static_threshold', 0.15)).float().unsqueeze(-1)
            safe_pred_res = safe_pred_res * mask
            info_dict['gate_prob'] = mask 

        info_dict["pred_res_normalized"] = safe_pred_res
        return safe_pred_res, info_dict

# =========================================================================
# 2. 高级架构与双源模型系列 (Advanced & Dual-Source Correctors)
# =========================================================================

class DualSourceFusionCorrector(BaseCorrector):
    def __init__(self, config):
        super().__init__(config)
        self.d_model = config.get('d_model', 128)
        n_heads = config.get('n_heads', 4)
        n_layers = config.get('n_layers', 2)
        dropout = config.get('dropout', 0.1)
        self.use_rank_encoding = config.get('use_rank_encoding', True)
        self.use_vibe = config.get('use_vibe_features', True)
        
        self.input_proj = nn.Linear(self.embed_dim, self.d_model)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.value_proj = nn.Linear(self.pred_len, self.d_model)
        if self.use_rank_encoding:
            self.val_pos_enc = PositionalEncoding(self.d_model, max_len=self.top_k + 5)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=n_heads, dim_feedforward=self.d_model * 4, 
            dropout=dropout, batch_first=True, norm_first=True 
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        if self.use_vibe:
            self.vibe_proj = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, self.d_model))
            self.output_head = nn.Linear(self.d_model * 2, self.pred_len)
        else:
            self.output_head = nn.Linear(self.d_model, self.pred_len)
        
        self.apply(self._init_weights)
        nn.init.xavier_uniform_(self.output_head.weight, gain=0.001)
        nn.init.constant_(self.output_head.bias, 0.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight[module.padding_idx], 0.0)

    def _fuse_dual_inputs(self, seq_emb, err_emb, batch_size, num_items):
        proj_seq = self.input_proj(seq_emb)
        proj_err = torch.zeros_like(proj_seq) if err_emb is None else self.input_proj(err_emb)
        return self.fusion_layer(torch.cat([proj_seq, proj_err], dim=-1))

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):           
        B, K, L = retrieved_residuals.shape
        t_err_emb = kwargs.get('t_err_emb', torch.zeros_like(target_emb))
        c_err_embs = kwargs.get('c_err_embs', torch.zeros_like(retrieved_embs))
        vibe_features = kwargs.get('vibe_features')

        fused_query = self._fuse_dual_inputs(target_emb, t_err_emb, B, 1)
        if fused_query.dim() == 2: fused_query = fused_query.unsqueeze(1)
        fused_key = self._fuse_dual_inputs(retrieved_embs, c_err_embs, B, K)
        value_feats = self.value_proj(retrieved_residuals)

        if self.use_rank_encoding:
            fused_key = self.val_pos_enc(fused_key)
            value_feats = self.val_pos_enc(value_feats)

        memory_combined = value_feats + fused_key
        attn_output = self.transformer_decoder(tgt=fused_query, memory=memory_combined, memory_key_padding_mask=None)
        hidden_state = attn_output.squeeze(1) 
        
        if self.use_vibe and vibe_features is not None:
            fused_hidden = torch.cat([hidden_state, self.vibe_proj(vibe_features)], dim=-1)
            safe_pred_res = torch.tanh(self.output_head(fused_hidden)) * (vibe_features[:, 0:1] * 3.0)
        else:
            safe_pred_res = self.output_head(hidden_state)

        info_dict = {"h_state": hidden_state}
        return self.apply_gating(safe_pred_res, info_dict, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class DualSourceSetMLPCorrector(BaseCorrector):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config.get('hidden_dim', 256)
        dropout = config.get('dropout', 0.2)
        self.use_vibe = config.get('use_vibe_features', True)
        
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + self.pred_len, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.target_encoder = nn.Linear(self.embed_dim * 2, self.hidden_dim)
        
        if self.use_vibe:
            self.vibe_proj = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, self.hidden_dim))
            predictor_input_dim = self.hidden_dim * 3
        else:
            predictor_input_dim = self.hidden_dim * 2
        
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.pred_len)
        )
        self.apply(self._init_weights)
        nn.init.xavier_uniform_(self.predictor[-1].weight, gain=0.001)
        nn.init.constant_(self.predictor[-1].bias, 0.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):           
        B, K, _ = retrieved_embs.shape
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        t_err_emb = kwargs.get('t_err_emb', torch.zeros_like(target_emb))
        c_err_embs = kwargs.get('c_err_embs', torch.zeros_like(retrieved_embs))
        if t_err_emb.dim() == 3: t_err_emb = t_err_emb.squeeze(1)
        vibe_features = kwargs.get('vibe_features')

        neighbor_feats = torch.cat([retrieved_embs, c_err_embs, retrieved_residuals], dim=-1)
        neighbor_encoded = self.neighbor_encoder(neighbor_feats.view(B * K, -1)).view(B, K, -1)
        context_vector = torch.mean(neighbor_encoded, dim=1)
        target_vec = self.target_encoder(torch.cat([target_emb, t_err_emb], dim=-1))
        
        if self.use_vibe and vibe_features is not None:
            vibe_emb = self.vibe_proj(vibe_features)
            combined = torch.cat([target_vec, context_vector, vibe_emb], dim=-1)
            raw_pred_res = self.predictor(combined)
            safe_pred_res = torch.tanh(raw_pred_res) * (vibe_features[:, 0:1] * 3.0)
        else:
            safe_pred_res = self.predictor(torch.cat([target_vec, context_vector], dim=-1))
            
        return self.apply_gating(safe_pred_res, {}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class RandomFrozenEncoderSetMLPCorrector(BaseCorrector):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config.get('hidden_dim', 256)
        dropout = config.get('dropout', 0.2)
        
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + self.pred_len, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.target_encoder = nn.Linear(self.embed_dim * 2, self.hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.pred_len)
        )
        self.apply(self._init_weights)
        for param in self.neighbor_encoder.parameters(): param.requires_grad = False
        for param in self.target_encoder.parameters(): param.requires_grad = False
        nn.init.xavier_uniform_(self.predictor[-1].weight, gain=0.001)
        nn.init.constant_(self.predictor[-1].bias, 0.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        B, K, _ = retrieved_embs.shape
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        t_err_emb = kwargs.get('t_err_emb', torch.zeros_like(target_emb))
        c_err_embs = kwargs.get('c_err_embs', torch.zeros_like(retrieved_embs))
        if t_err_emb.dim() == 3: t_err_emb = t_err_emb.squeeze(1)

        with torch.no_grad():
            neighbor_feats = torch.cat([retrieved_embs, c_err_embs, retrieved_residuals], dim=-1)
            neighbor_encoded = self.neighbor_encoder(neighbor_feats.view(B * K, -1)).view(B, K, -1)
            context_vector = torch.mean(neighbor_encoded, dim=1)
            target_vec = self.target_encoder(torch.cat([target_emb, t_err_emb], dim=-1))
        
        pred_res = self.predictor(torch.cat([target_vec, context_vector], dim=-1))
        return self.apply_gating(pred_res, {}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class DualSourceGatedMLPCorrector(BaseCorrector):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config.get('hidden_dim', 256)
        dropout = config.get('dropout', 0.2)
        self.zcr_threshold = config.get('zcr_threshold', 0.4)
        
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + self.pred_len, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.target_encoder = nn.Linear(self.embed_dim * 2, self.hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.hidden_dim, self.pred_len)
        )
        self.apply(self._init_weights)
        nn.init.xavier_uniform_(self.predictor[-1].weight, gain=0.001)
        nn.init.constant_(self.predictor[-1].bias, 0.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def _compute_stability_features(self, local_res_raw):
        if local_res_raw is None or local_res_raw.numel() == 0:
            return torch.zeros(1, 1, device=self.predictor[0].weight.device)
        return ((local_res_raw[:, :-1] * local_res_raw[:, 1:]) < 0).float().mean(dim=1, keepdim=True)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        B, K, _ = retrieved_embs.shape
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        t_err_emb = kwargs.get('t_err_emb', torch.zeros_like(target_emb))
        c_err_embs = kwargs.get('c_err_embs', torch.zeros_like(retrieved_embs))
        if t_err_emb.dim() == 3: t_err_emb = t_err_emb.squeeze(1)

        neighbor_feats = torch.cat([retrieved_embs, c_err_embs, retrieved_residuals], dim=-1)
        neighbor_encoded = self.neighbor_encoder(neighbor_feats.view(B * K, -1)).view(B, K, -1)
        
        target_vec = self.target_encoder(torch.cat([target_emb, t_err_emb], dim=-1))
        raw_pred = self.predictor(torch.cat([target_vec, torch.mean(neighbor_encoded, dim=1)], dim=-1))
        
        target_local_res = kwargs.get('target_local_res')
        if target_local_res is not None:
            zcr = self._compute_stability_features(target_local_res.squeeze(1) if target_local_res.dim() == 3 else target_local_res)
        else:
            zcr = torch.zeros(B, 1, device=target_emb.device)
            
        gate_state = torch.ones_like(zcr) if self.training else (zcr < self.zcr_threshold).float()
        final_pred = raw_pred * gate_state
        
        return self.apply_gating(final_pred, {"gate_value": gate_state.mean(), "raw_zcr": zcr.mean()}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), target_local_res)

class DualSourceResMLPCorrector(BaseCorrector):
    def __init__(self, config):
        super().__init__(config)
        self.context_len = config.get('context_len', 96) 
        self.hidden_dim = config.get('hidden_dim', 256)
        dropout = config.get('dropout', 0.2)
        
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + self.pred_len, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.target_emb_encoder = nn.Linear(self.embed_dim * 2, self.hidden_dim)
        self.raw_res_encoder = nn.Sequential(
            nn.Linear(self.context_len, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2), nn.GELU(), nn.Dropout(dropout)
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + (self.hidden_dim // 2), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.hidden_dim, self.pred_len)
        )
        self.apply(self._init_weights)
        nn.init.xavier_uniform_(self.predictor[-1].weight, gain=0.001)
        nn.init.constant_(self.predictor[-1].bias, 0.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        B, K, _ = retrieved_embs.shape
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        t_err_emb = kwargs.get('t_err_emb', torch.zeros_like(target_emb))
        c_err_embs = kwargs.get('c_err_embs', torch.zeros_like(retrieved_embs))
        if t_err_emb.dim() == 3: t_err_emb = t_err_emb.squeeze(1)

        neighbor_feats = torch.cat([retrieved_embs, c_err_embs, retrieved_residuals], dim=-1)
        context_vector = torch.mean(self.neighbor_encoder(neighbor_feats.view(B * K, -1)).view(B, K, -1), dim=1) 
        target_vec = self.target_emb_encoder(torch.cat([target_emb, t_err_emb], dim=-1)) 
        
        target_local_res = kwargs.get('target_local_res')
        if target_local_res is not None:
            l_res = target_local_res.squeeze(1) if target_local_res.dim() == 3 else target_local_res
            if l_res.shape[1] > self.context_len: l_res = l_res[:, -self.context_len:]
            elif l_res.shape[1] < self.context_len: l_res = F.pad(l_res, (self.context_len - l_res.shape[1], 0))
            raw_res_vec = self.raw_res_encoder(l_res)
        else:
            raw_res_vec = torch.zeros(B, self.hidden_dim // 2, device=target_emb.device)
            
        pred_res = self.predictor(torch.cat([target_vec, context_vector, raw_res_vec], dim=-1))
        return self.apply_gating(pred_res, {}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), target_local_res)

class SemanticRouterCorrector(BaseCorrector):
    def __init__(self, config):
        super().__init__(config)
        self.d_model = config.get('d_model', 128)
        dropout = config.get('dropout', 0.1)

        self.q_proj = nn.Sequential(nn.Linear(self.embed_dim * 2, self.d_model), nn.LayerNorm(self.d_model), nn.GELU())
        self.k_proj = nn.Sequential(nn.Linear(self.embed_dim * 2, self.d_model), nn.LayerNorm(self.d_model), nn.GELU())
        self.scale = self.d_model ** -0.5
        self.refinement = nn.Sequential(nn.Linear(self.pred_len, self.d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.d_model, self.pred_len))
        self.gate = nn.Parameter(torch.ones(1))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        if hasattr(self, 'refinement'):
            nn.init.constant_(self.refinement[-1].weight, 0.0)
            nn.init.constant_(self.refinement[-1].bias, 0.0)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        t_err_emb = kwargs.get('t_err_emb', torch.zeros_like(target_emb))
        c_err_embs = kwargs.get('c_err_embs', torch.zeros_like(retrieved_embs))
        if t_err_emb.dim() == 3: t_err_emb = t_err_emb.squeeze(1)

        query = self.q_proj(torch.cat([target_emb, t_err_emb], dim=-1)).unsqueeze(1)
        key = self.k_proj(torch.cat([retrieved_embs, c_err_embs], dim=-1))
        attn_weights = F.softmax(torch.bmm(query, key.transpose(1, 2)) * self.scale, dim=-1)

        routed_residual = torch.bmm(attn_weights, retrieved_residuals).squeeze(1)
        final_pred = (routed_residual + self.refinement(routed_residual)) * self.gate
        return self.apply_gating(final_pred, {"attn_weights": attn_weights.squeeze(1), "router_gate_value": self.gate}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class IntraInterRouterCorrector(BaseCorrector):
    def __init__(self, config):
        super().__init__(config)
        self.d_model = config.get('d_model', 128)
        dropout = config.get('dropout', 0.1)

        self.intra_fusion = IntraSequenceFusion(self.embed_dim, self.d_model)
        self.confidence_scorer = ConfidenceScorer(self.embed_dim, self.d_model // 2)
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.scale = self.d_model ** -0.5
        self.refinement = nn.Sequential(nn.Linear(self.pred_len, self.d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.d_model, self.pred_len))
        self.gate = nn.Parameter(torch.ones(1))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        if hasattr(self, 'refinement'):
            nn.init.constant_(self.refinement[-1].weight, 0.0)
            nn.init.constant_(self.refinement[-1].bias, 0.0)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        t_err_emb = kwargs.get('t_err_emb', torch.zeros_like(target_emb))
        c_err_embs = kwargs.get('c_err_embs', torch.zeros_like(retrieved_embs))
        if t_err_emb.dim() == 3: t_err_emb = t_err_emb.squeeze(1)

        t_enhanced = self.intra_fusion(target_emb, t_err_emb).unsqueeze(1) 
        c_enhanced = self.intra_fusion(retrieved_embs, c_err_embs) 
        c_confidence = self.confidence_scorer(retrieved_embs, c_err_embs) 

        scores = torch.bmm(self.q_proj(t_enhanced), self.k_proj(c_enhanced).transpose(1, 2)) * self.scale
        modulated_weights = F.softmax(scores, dim=-1) * c_confidence.transpose(1, 2) 
        modulated_weights = modulated_weights / (modulated_weights.sum(dim=-1, keepdim=True) + 1e-9)

        routed_residual = torch.bmm(modulated_weights, retrieved_residuals).squeeze(1) 
        final_pred = (routed_residual + self.refinement(routed_residual)) * self.gate
        return self.apply_gating(final_pred, {"attn_weights": modulated_weights.squeeze(1), "confidence_mean": c_confidence.mean()}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class DualLatentCrossAttnCorrector(BaseCorrector):
    def __init__(self, config):
        super().__init__(config)
        self.d_model = config.get('d_model', 256) 
        n_heads = config.get('n_heads', 8)
        dropout = config.get('dropout', 0.2)
        
        self.q_proj = nn.Sequential(nn.Linear(self.embed_dim * 2, self.d_model), nn.LayerNorm(self.d_model), nn.GELU())
        self.k_proj = nn.Sequential(nn.Linear(self.embed_dim * 2, self.d_model), nn.LayerNorm(self.d_model), nn.GELU())
        self.v_proj = nn.Sequential(nn.Linear(self.pred_len, self.d_model), nn.LayerNorm(self.d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.d_model, self.d_model))
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(self.d_model)
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model * 2), nn.LayerNorm(self.d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.pred_len)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        if hasattr(self, 'decoder'):
            nn.init.xavier_uniform_(self.decoder[-1].weight, gain=0.01)
            nn.init.constant_(self.decoder[-1].bias, 0.0)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        t_err_emb = kwargs.get('t_err_emb', torch.zeros_like(target_emb))
        c_err_embs = kwargs.get('c_err_embs', torch.zeros_like(retrieved_embs))
        if t_err_emb.dim() == 3: t_err_emb = t_err_emb.squeeze(1)

        q = self.q_proj(torch.cat([target_emb, t_err_emb], dim=-1)).unsqueeze(1) 
        k = self.k_proj(torch.cat([retrieved_embs, c_err_embs], dim=-1))             
        v = self.v_proj(retrieved_residuals)        

        attn_out, attn_weights = self.cross_attn(query=q, key=k, value=v)
        pred_res = self.decoder(torch.cat([q.squeeze(1), self.attn_norm(q + attn_out).squeeze(1)], dim=-1)) 
        return self.apply_gating(pred_res, {"attn_weights": attn_weights.squeeze(1)}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

# =========================================================================
# 3. 经典基线与传统模型 (Baselines & Traditional Models)
# =========================================================================

class DeepTransformerCorrector(BaseCorrector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        hidden_dim = config.get("hidden_dim", 256)
        num_heads = config.get("num_heads", 4)
        num_layers = config.get("num_layers", 3)
        dropout = config.get("dropout", 0.1)
        
        self.query_proj = nn.Linear(self.embed_dim, hidden_dim)
        self.key_proj = nn.Linear(self.embed_dim, hidden_dim)
        self.value_proj = nn.Sequential(nn.Linear(self.pred_len, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout))
        self.layers = nn.ModuleList([CrossAttentionBlock(hidden_dim, num_heads, dropout, config.get("dim_feedforward") or hidden_dim * 4) for _ in range(num_layers)])
        self.emb_dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, self.pred_len)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        nn.init.constant_(self.output_head.weight, 0.0)
        nn.init.constant_(self.output_head.bias, 0.0)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        if torch.isnan(target_emb).any() or torch.isnan(retrieved_residuals).any():
             return self.apply_gating(torch.zeros(target_emb.shape[0], self.pred_len, device=target_emb.device), {}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

        query = self.emb_dropout(self.query_proj(target_emb).unsqueeze(1))
        key = self.emb_dropout(self.key_proj(retrieved_embs))
        value = self.value_proj(retrieved_residuals)
        
        final_weights = None
        for layer in self.layers:
            query, weights = layer(query, key, value)
            final_weights = weights 
        
        pred_res = self.output_head(self.final_norm(query)).squeeze(1)
        return self.apply_gating(pred_res, {"attn_weights": final_weights}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class StandardTransformerCorrector(BaseCorrector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.d_model = config.get("hidden_dim", 256)
        nhead = config.get("num_heads", 4)
        num_layers = config.get("num_layers", 3)
        dropout = config.get("dropout", 0.1)
        
        self.query_proj = nn.Linear(self.embed_dim, self.d_model) 
        self.key_proj = nn.Linear(self.embed_dim, self.d_model)   
        self.value_proj = nn.Linear(self.pred_len, self.d_model)  
        self.emb_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([StandardTransformerLayer(self.d_model, nhead, config.get("dim_feedforward", self.d_model * 4), dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(self.d_model)
        self.output_head = nn.Linear(self.d_model, self.pred_len)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        nn.init.constant_(self.output_head.weight, 0.0)
        nn.init.constant_(self.output_head.bias, 0.0)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        if torch.isnan(target_emb).any() or torch.isnan(retrieved_residuals).any():
             return self.apply_gating(torch.zeros(target_emb.shape[0], self.pred_len, device=target_emb.device), {}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

        query = self.emb_dropout(self.query_proj(target_emb).unsqueeze(1))
        key = self.emb_dropout(self.key_proj(retrieved_embs))
        value = self.emb_dropout(self.value_proj(retrieved_residuals))

        final_weights = None
        for layer in self.layers:
            query, weights = layer(query, key, value)
            final_weights = weights
        
        pred_res = self.output_head(self.final_norm(query)).squeeze(1)
        return self.apply_gating(pred_res, {"attn_weights": final_weights}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class AttentionCorrector(BaseCorrector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        hidden_dim = config.get("hidden_dim", 128)
        num_heads = config.get("num_heads", 4)
        dropout = config.get("dropout", 0.1)
        
        self.query_proj = nn.Linear(self.embed_dim, hidden_dim)
        self.key_proj = nn.Linear(self.embed_dim, hidden_dim)
        self.value_proj = nn.Linear(self.pred_len, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.output_head = nn.Linear(hidden_dim, self.pred_len)
        
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        nn.init.constant_(self.output_head.weight, 0.0)
        nn.init.constant_(self.output_head.bias, 0.0)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        query = self.query_proj(target_emb).unsqueeze(1)
        key = self.key_proj(retrieved_embs)
        value = self.value_proj(retrieved_residuals)
        
        attn_out, weights = self.attention(query, key, value)
        x = self.norm(query + attn_out)
        pred_res = self.output_head(x + self.ffn(x)).squeeze(1)
        return self.apply_gating(pred_res, {"attn_weights": weights}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class MLPCorrector(BaseCorrector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        hidden_dim = config.get("hidden_dim", 128)
        input_dim = self.embed_dim + self.top_k * (self.embed_dim + self.pred_len)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.pred_len)
        )
        nn.init.constant_(self.net[-1].weight, 0.0)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        B = target_emb.shape[0]
        x = torch.cat([target_emb, retrieved_embs.view(B, -1), retrieved_residuals.view(B, -1)], dim=1)
        pred_res = self.net(x)
        return self.apply_gating(pred_res, {}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class LinearWeightedCorrector(BaseCorrector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.weights = nn.Parameter(torch.ones(self.top_k) / self.top_k)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        B, K, _ = retrieved_residuals.shape
        actual_k = min(K, self.top_k)
        attn_weights = F.softmax(self.weights[:actual_k], dim=0) 
        pred_res = torch.sum(retrieved_residuals[:, :actual_k, :] * attn_weights.unsqueeze(0).unsqueeze(-1), dim=1)
        return self.apply_gating(pred_res, {"attn_weights": attn_weights.unsqueeze(0).expand(B, -1)}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class SimilarityWeightedCorrector(BaseCorrector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.temperature = nn.Parameter(torch.tensor(config.get("temperature", 1.0)))
        self.scale = self.embed_dim ** -0.5

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        scores = torch.bmm(target_emb.unsqueeze(1), retrieved_embs.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(scores / self.temperature, dim=-1)
        pred_res = torch.bmm(attn_weights, retrieved_residuals).squeeze(1)
        return self.apply_gating(pred_res, {"attn_weights": attn_weights}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class WeightedBaselineCorrector(BaseCorrector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lambda_weight = config.get("lambda_weight", 0.5)
        self.temperature = config.get("temperature", 1.0)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        if torch.isnan(target_emb).any() or torch.isnan(retrieved_residuals).any():
             return self.apply_gating(torch.zeros(target_emb.shape[0], self.pred_len, device=target_emb.device), {}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

        scores = torch.bmm(F.normalize(target_emb, p=2, dim=1).unsqueeze(1), F.normalize(retrieved_embs, p=2, dim=2).transpose(1, 2))
        weights = F.softmax(scores / self.temperature, dim=-1)
        pred_res = self.lambda_weight * torch.bmm(weights, retrieved_residuals).squeeze(1)
        return self.apply_gating(pred_res, {"attn_weights": weights}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class LearnableWeightedCorrector(BaseCorrector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.temperature = nn.Parameter(torch.tensor(config.get("temperature_init", 1.0)))
        self.lambda_weight = nn.Parameter(torch.tensor(config.get("lambda_init", 0.5)))

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        scores = torch.bmm(F.normalize(target_emb, p=2, dim=1).unsqueeze(1), F.normalize(retrieved_embs, p=2, dim=2).transpose(1, 2))
        curr_temp = self.temperature.abs() + 1e-6
        weights = F.softmax(scores / curr_temp, dim=-1)
        pred_res = self.lambda_weight * torch.bmm(weights, retrieved_residuals).squeeze(1)
        return self.apply_gating(pred_res, {"attn_weights": weights, "lambda": self.lambda_weight, "temperature": curr_temp}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class MeanRetrievalCorrector(BaseCorrector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        pred_res = torch.mean(retrieved_residuals, dim=1)
        return self.apply_gating(pred_res, {}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class GlobalBiasCorrector(BaseCorrector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.global_bias = nn.Parameter(torch.zeros(self.pred_len))

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        pred_res = self.global_bias.unsqueeze(0).expand(target_emb.shape[0], -1)
        return self.apply_gating(pred_res, {}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class LocalResARCorrector(BaseCorrector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.context_len = config.get('pred_len', self.pred_len)
        self.linear = nn.Linear(self.context_len, self.pred_len)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.001)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        target_local_res = kwargs.get('target_local_res')
        if target_local_res is None:
            return self.apply_gating(torch.zeros(target_emb.shape[0], self.pred_len, device=target_emb.device), {}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), target_local_res)
        
        x = target_local_res.squeeze(1) if target_local_res.dim() == 3 else target_local_res
        if x.shape[1] > self.context_len: x = x[:, -self.context_len:]
        elif x.shape[1] < self.context_len: x = F.pad(x, (self.context_len - x.shape[1], 0))
            
        pred_res = self.linear(x)
        return self.apply_gating(pred_res, {}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), target_local_res)

class ZeroCorrector(BaseCorrector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        pred_res = torch.zeros((target_emb.shape[0], self.pred_len), device=target_emb.device, dtype=target_emb.dtype)
        return self.apply_gating(pred_res, {}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))

class LightWeightMetaCorrector(BaseCorrector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        input_dim = self.pred_len * 2 + 2 
        hidden_dim = config.get("hidden_dim", 64) 
        self.meta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.pred_len) 
        )
        self.temperature = config.get("temperature", 1.0)
        nn.init.constant_(self.meta_net[-1].weight, 0.0)
        nn.init.constant_(self.meta_net[-1].bias, 0.0)

    def _calc_trend(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        B, L = x.shape
        if L < 2: return torch.zeros(B, 1, device=x.device)
        t = torch.arange(L, device=x.device, dtype=x.dtype)
        return ((t - t.mean()) * (x - x.mean(dim=1, keepdim=True))).sum(dim=1, keepdim=True) / (((t - t.mean()) ** 2).sum() + 1e-6)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals, history=None, target_residual=None, **kwargs):
        if target_emb.dim() == 3: target_emb = target_emb.squeeze(1)
        B = target_emb.shape[0]
        scores = torch.bmm(F.normalize(target_emb, p=2, dim=1).unsqueeze(1), F.normalize(retrieved_embs, p=2, dim=2).transpose(1, 2))
        weights = F.softmax(scores / self.temperature, dim=-1) 
        
        weighted_res = torch.bmm(weights, retrieved_residuals).squeeze(1)
        consistency_std = retrieved_residuals.std(dim=1)
        
        if history is not None:
            if history.dim() == 3: history = history.squeeze(1)
            hist_std = history.std(dim=1, keepdim=True) + 1e-6
            hist_trend = self._calc_trend(history)
        else:
            hist_std, hist_trend = torch.zeros(B, 1, device=weighted_res.device), torch.zeros(B, 1, device=weighted_res.device)

        pred_res = self.meta_net(torch.cat([weighted_res, consistency_std, hist_std, hist_trend], dim=1))
        return self.apply_gating(pred_res, {"attn_weights": weights, "consistency_mean": consistency_std.mean(dim=1)}, retrieved_residuals, target_emb, kwargs.get('ts_physics_features'), kwargs.get('target_local_res'))