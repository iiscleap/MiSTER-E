import torch
import torch.nn as nn
import torch.nn.functional as F

# def expert_diversity_loss(expert_outputs):  # [B, T, 4, D]
#     B, T, E, D = expert_outputs.shape
#     loss = 0.0
#     num_pairs = 0

#     # Normalize along the feature dimension for cosine similarity
#     expert_outputs = F.normalize(expert_outputs, dim=-1)  # Still [B, T, 4, D]

#     # For every pair of experts, compute similarity and sum
#     for i in range(E):
#         for j in range(i + 1, E):
#             # Cosine similarity: [B, T]
#             sim = (expert_outputs[:, :, i] * expert_outputs[:, :, j]).sum(dim=-1)
#             loss += sim.mean()
#             num_pairs += 1

#     # Take average similarity over all pairs — higher similarity means less diversity
#     avg_similarity = loss / num_pairs

#     # Encourage *low similarity* → minimize this value
#     return avg_similarity

def compute_symmetric_kl(stacked_logits, temperature=1.0):
    """
    stacked_logits: [B, T, 3, D]
    Returns: scalar KL loss encouraging expert agreement
    """
    B, T, E, D = stacked_logits.shape  # E = number of experts (3)
    kl_loss = 0.0
    count = 0

    for i in range(E):
        for j in range(i + 1, E):
            p_logits = stacked_logits[:, :, i, :] / temperature
            q_logits = stacked_logits[:, :, j, :] / temperature

            p_log_probs = F.log_softmax(p_logits, dim=-1)
            q_probs = F.softmax(q_logits, dim=-1)

            q_log_probs = F.log_softmax(q_logits, dim=-1)
            p_probs = F.softmax(p_logits, dim=-1)

            # KL(p || q) + KL(q || p)
            kl_ij = F.kl_div(p_log_probs, q_probs, reduction='batchmean') + \
                    F.kl_div(q_log_probs, p_probs, reduction='batchmean')

            kl_loss += kl_ij
            count += 1

    return kl_loss / count  # Average over all expert pairs


# class MultiLayerSelfAttention(nn.Module):
#     def __init__(self, input_dim, num_layers=12, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=input_dim,
#                 nhead=num_heads,
#                 dropout=dropout,
#                 batch_first=True
#             ) for _ in range(num_layers)
#         ])
#         self.classifier = nn.Linear(input_dim, 6)

#     def forward(self, x, mask=None):
#         for layer in self.layers:
#             x = layer(x, src_key_padding_mask=mask)
#         return x, self.classifier(x)

# class GRUModel(nn.Module):

#     def __init__(self, input_dim, hidden_dim, output_dim, dropout, layers, bidirectional_flag):
#         super().__init__()

#         self.dropout = nn.Dropout(dropout)
#         self.num_layers = layers
#         self.hidden_dim = hidden_dim
#         self.units = nn.ModuleList()
#         self.rnn_1 = nn.GRU(input_dim, hidden_dim, num_layers=layers, bidirectional=bidirectional_flag, batch_first=True)
        
#         self.bidirectional_used = bidirectional_flag

#         self.fc_1 = nn.Linear(input_dim, hidden_dim*2)#+ input_dim)
#         self.conv = nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=1)

#         self.conv3 = nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=5, padding=2)
        
#         self.fc = nn.Linear(hidden_dim*2, output_dim)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         output_gru_1, _ = self.rnn_1(x)
#         output_gru_1 = output_gru_1.permute(1, 0, 2)
#         permuted_x = output_gru_1.clone()
#         output_con = permuted_x.permute(1, 2, 0)
#         output_gru_1 = output_gru_1.permute(1, 2, 0)
#         output_con_3 = self.relu(self.conv3(output_gru_1))
#         output_con_5 = self.relu(self.conv5(output_gru_1))
#         output_con = self.relu(self.conv(output_con))
#         output_con = output_con + output_con_3 + output_con_5
#         output_con = output_con.permute(0, 2, 1)
            

#         output = output_con

#         x_1 = self.fc_1(x)

#         output = output + x_1 
#         out = self.fc(output)
#         return output, out

class GRUModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, layers, bidirectional_flag):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.num_layers = layers
        self.hidden_dim = hidden_dim
        self.units = nn.ModuleList()
        self.rnn_1 = nn.GRU(hidden_dim*2, hidden_dim, num_layers=layers, bidirectional=bidirectional_flag, batch_first=True)
        
        self.bidirectional_used = bidirectional_flag

        self.fc_1 = nn.Linear(input_dim, hidden_dim*2)#+ input_dim)
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim*2, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim*2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim*2, kernel_size=5, padding=2)
        
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        permuted_x = x.permute(0, 2, 1)
        x_1 = self.relu(self.conv(permuted_x))
        x_3 = self.relu(self.conv3(permuted_x))
        x_5 = self.relu(self.conv5(permuted_x))
        output_conv = x_1 + x_3 + x_5
        output_conv = output_conv.permute(0, 2, 1)

        output_gru, _ = self.rnn_1(output_conv)
        x_skip = self.fc_1(x)
        output_gru = output_gru + x_skip

        out = self.fc(output_gru)
        return output_gru, out

# class TemporalInceptionBlock(nn.Module):
#     def __init__(self, in_dim, out_dim, kernel_sizes=(1, 3, 5), dropout=0.1):
#         super().__init__()
#         self.branches = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(in_dim, out_dim, kernel_size=k, padding=k//2),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(out_dim)
#             )
#             for k in kernel_sizes
#         ])
#         self.dropout = nn.Dropout(dropout)
#         self.proj = nn.Linear(len(kernel_sizes) * out_dim, in_dim)  # optional projection for residual

#     def forward(self, x):
#         # x: [B, T, D]
#         x = x.transpose(1, 2)  # [B, D, T]
#         out = torch.cat([branch(x) for branch in self.branches], dim=1)  # [B, len(K)*out_dim, T]
#         out = out.transpose(1, 2)  # [B, T, len(K)*out_dim]
#         out = self.proj(out)  # project back to original dimension
#         out = self.dropout(out)
#         return out

# class ContextAdditionModule(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_classes, kernel_sizes=(1, 3, 5)):
#         super().__init__()
#         self.inception = TemporalInceptionBlock(hidden_dim*2, hidden_dim*2, kernel_sizes)
#         self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
#         self.out_proj = nn.Linear(2 * hidden_dim, num_classes)
#         self.fc = nn.Linear(input_dim, hidden_dim*2)

#     def forward(self, x, lengths=None):
#         # x: [B, T, D]

#         x_gru, _ = self.gru(x)
#         x_incept = self.inception(x_gru) + self.fc(x)  # residual connection
#         # if lengths is not None:
#         #     packed = nn.utils.rnn.pack_padded_sequence(x_incept, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         #     packed_out, _ = self.gru(packed)
#         #     x_gru, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
#         # else:
#         #     x_gru, _ = self.gru(x_incept)
        
        
#         return x_incept, self.out_proj(x_incept)

class CrossAttentionModel_s(nn.Module):
    def __init__(self, hidden_dim, hidden_size=60, num_atten=3):
        super().__init__()
        self.inter_dim = hidden_size//num_atten
        self.num_heads = num_atten
        self.fc_audq = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_audk = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_audv = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)  
        
        self.fc_asrq = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_asrk = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_asrv = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)

        self.fc_audq_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_audk_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_audv_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_asrq_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_asrk_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_asrv_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)

        self.multihead_attn_asr = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.1,
                                                         bias = True)
        self.multihead_attn_aud = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.1,
                                                         bias = True)
        
        self.multihead_attn_selfaud = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.1,
                                                         bias = True)
        self.multihead_attn_selfasr = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.1,
                                                         bias = True)                      
        self.dropout = nn.Dropout(0.5)

        self.layer_norm_t = nn.LayerNorm(hidden_dim, eps = 1e-6)
        self.layer_norm_a = nn.LayerNorm(hidden_dim, eps = 1e-6)
        self.layer_norm_t_s = nn.LayerNorm(hidden_dim, eps = 1e-6)
        self.layer_norm_a_s = nn.LayerNorm(hidden_dim, eps = 1e-6)

        self.fc_text = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
        self.fc_audio = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_text_s = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
        self.fc_aud_s = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
    
    def forward(self, audio, text):

        text_q = self.fc_asrq(text)
        text_k = self.fc_asrk(text)
        text_v = self.fc_asrv(text)
        audio_q = self.fc_audq(audio)
        audio_k = self.fc_audk(audio)
        audio_v = self.fc_audv(audio)
        text_cross = self.multihead_attn_asr(text_q, audio_k, audio_v, need_weights = False)[0]
        audio_cross = self.multihead_attn_aud(audio_q, text_k, text_v, need_weights = False)[0]
        text_q = self.dropout(self.fc_text(text_cross))
        audio_q = self.dropout(self.fc_audio(audio_cross))
        text_q += text
        audio_q += audio
        text_q = self.layer_norm_t(text_q)
        audio_q = self.layer_norm_a(audio_q)
        text_q_s = self.fc_asrq_s(text_q)
        text_k_s = self.fc_asrk_s(text_q)
        text_v_s = self.fc_asrv_s(text_q)
        aud_q_s = self.fc_audq_s(audio_q)
        aud_k_s = self.fc_audk_s(audio_q)
        aud_v_s = self.fc_audv_s(audio_q)
        text_self = self.multihead_attn_selfasr(text_q_s, text_k_s, text_v_s, need_weights = False)[0]
        aud_self = self.multihead_attn_selfaud(aud_q_s, aud_k_s, aud_v_s, need_weights = False)[0]
        text_q_s = self.dropout(self.fc_text_s(text_self))
        aud_q_s = self.dropout(self.fc_aud_s(aud_self))
        text_q_s += text_q
        aud_q_s += audio_q
        text_q_fin = self.layer_norm_t_s(text_q_s)
        aud_q_fin = self.layer_norm_a_s(aud_q_s)
        return text_q_fin, aud_q_fin

class FusionModel(nn.Module):
    def __init__(self, hidden_dim, n_layers, num_classes):
        super().__init__()
        self.hid = 1024
        self.fc = nn.Linear(self.hid*2, 1024)
        self.fc_2 = nn.Linear(1024, num_classes)
        self.units = nn.ModuleList()
        self.fc_aud = nn.Linear(hidden_dim, self.hid)
        self.fc_text = nn.Linear(hidden_dim, self.hid)
        self.fc_aud_self = nn.Linear(hidden_dim, self.hid)
        self.fc_text_self = nn.Linear(hidden_dim, self.hid)
        self.bn = nn.LayerNorm(self.hid*2, eps = 1e-6)
        self.relu = nn.ReLU()
        for ind in range(n_layers):
            self.units.append(CrossAttentionModel_s(self.hid, 120, 3))
    
    def forward(self, audio_orig, text_orig):

        audio = self.fc_aud(audio_orig)
        text = self.fc_text(text_orig)
        audio = audio.permute(1, 0, 2)
        text = text.permute(1, 0, 2)
        for model_ca in self.units:
            text, audio = model_ca(audio, text)
        audio = audio.permute(1, 0, 2)
        text = text.permute(1, 0, 2)
        audio_self = self.fc_aud_self(audio_orig)
        text_self = self.fc_text_self(text_orig)
        audio += audio_self
        text += text_self
        concat = torch.cat((text, audio), -1)
        concat = self.bn(concat)
        concat = self.fc(concat)
        output = self.fc_2(self.relu(concat))

        return audio, text, output

class AttentionMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, return_weights = False):
        super().__init__()

        # Experts
        self.audio_self = GRUModel(2048, 512, num_classes, 0.2, 3, True)
        self.text_self = GRUModel(2048, 512, num_classes, 0.2, 3, True)
        # self.audio_self = ContextAdditionModule(1024, 512, num_classes)
        # self.text_self = ContextAdditionModule(1024, 512, num_classes)
        self.fusion = FusionModel(2048, 4, num_classes)
        self.return_weights = return_weights

        self.gate = nn.Sequential(
            nn.Linear(num_classes * 3, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
        self.temperature = 2
        self.proj = nn.Linear(1024, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, audio_feat, text_feat, mask=None, labels=None):
        # shape: [B, T, D]
        audio_self_out, audio_logits = self.audio_self(audio_feat)
        text_self_out, text_logits = self.text_self(text_feat)
        audio, text, fusion_logits = self.fusion(audio_feat, text_feat)
        concat_for_gate = torch.cat([
            audio_logits, text_logits, fusion_logits
        ], dim=-1)

        gate_logits = self.gate(concat_for_gate)  # (B, T, num_experts)
        weights = F.softmax(gate_logits / self.temperature, dim=-1)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
        # Stack expert outputs and mix
        stacked = torch.stack([
            audio_logits, text_logits, fusion_logits
        ], dim=-2)  # [B, T, 4, D]
        moe_output = torch.sum(weights.unsqueeze(-1) * stacked, dim=-2)  # [B, T, D]
        diversity_loss = compute_symmetric_kl(stacked)

        if self.return_weights : 
                return moe_output, audio, text, audio_logits, text_logits, fusion_logits, diversity_loss, audio_self_out, text_self_out, weights
        return moe_output, audio, text, audio_logits, text_logits, fusion_logits, diversity_loss, audio_self_out, text_self_out
