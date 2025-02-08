from constants import constants
import torch
import torch.nn as nn
import torch.nn.functional as F


class SD_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc3(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x


class RouteDistillNet(nn.Module):
    def __init__(self, embedding,
                 embed_size, hidden_size, 
                 num_layers,
                 route_num,
                 dataset):
        super().__init__()
        self.cnst = constants(dataset)

        # embedding layer
        self.embedding = embedding

        # encoder for SD
        self.route_num = route_num
        self.sd_mlps =nn.ModuleList([SD_MLP(embed_size * 2, hidden_size, hidden_size) for _ in range(self.route_num)])

        # encoder for trajectory
        self.src_encoder = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        # attention for route representation
        self.mh_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

        # decoder for trajectory
        self.src_decoder = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        # decoder for route
        self.route_decoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.num_layers = num_layers

    def loss(self, sd_reps):
        if len(sd_reps) == 1:
            return torch.tensor(0.0)
        total_distance = 0
        c = 0
        for i in range(len(sd_reps)):
            for j in range(len(sd_reps)):
                if i != j:
                    distance = F.cosine_similarity(sd_reps[i], sd_reps[j], dim=1)
                    total_distance += distance
                    c += 1
        total_distance = total_distance / c + 1
        total_distance = torch.mean(total_distance)
        return total_distance

    def src_encode(self, src, lens):
        # src: [batch_size, seq_len]
        # lens: [batch_size]
        lens = lens.tolist()
        src_embed = self.embedding(src)  # [batch_size, seq_len, embed_size]
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(src_embed, lens, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.src_encoder(packed_src_embed)  # h_n: [num_layers * 2, batch_size, hidden_size]
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # [batch_size, seq_len, hidden_size * 2]
        return output, h_n, c_n

    def src_decode(self, tgt, h_n, c_n, lens):
        # tgt: [batch_size, seq_len]
        # h_n: [num_layers * 2, batch_size, hidden_size]
        # lens: [batch_size]
        lens = lens.tolist()
        tgt_embed = self.embedding(tgt)  # [batch_size, seq_len, embed_size]
        packed_tgt_embed = nn.utils.rnn.pack_padded_sequence(tgt_embed, lens, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.src_decoder(packed_tgt_embed, (h_n, c_n))
        # output: [batch_size, seq_len, hidden_size * 2]
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return output


    def forward(self, src_data, src_lens, tgt_data):
        # src_data, tgt_data: [batch_size, seq_len]; src's seq_len = tgt's seq_len + 1
        # src_lens: [batch_size]

        # get sd representations
        s = src_data[:, 0] # [batch_size]
        d_index = src_lens - 1
        d = src_data[torch.arange(len(d_index)), d_index] # [batch_size]
        s_embed = self.embedding(s)  # [batch_size, embed_size]
        d_embed = self.embedding(d)  # [batch_size, embed_size]
        
        sd_embed = torch.cat([s_embed, d_embed], dim=-1)  # [batch_size, embed_size * 2]
        sd_reps = []
        for i in range(self.route_num):
            sd_rep_i = self.sd_mlps[i](sd_embed)  # [batch_size, hidden_size]
            sd_reps.append(sd_rep_i)
        rdn_loss = self.loss(sd_reps)
        sd_reps = torch.stack(sd_reps, dim=1)  # [batch_size, route_num, hidden_size]

        # get trajectory representation
        # encoder_output: [batch_size, seq_len, hidden_size * 2]
        # encoder_h_n: [num_layers * 2, batch_size, hidden_size]
        encoder_output, encoder_h_n, encoder_c_n = self.src_encode(src_data, src_lens)
        
        # attention
        encoder_h_n_query = encoder_h_n[-1].unsqueeze(1)  # [batch_size, 1, hidden_size]
        route_rep, weight = self.mh_attn(query=encoder_h_n_query, key=sd_reps, value=sd_reps, need_weights=True)  # [batch_size, 1, hidden_size]
        route_rep = route_rep.squeeze(dim=1)  # [batch_size, hidden_size]

        # src decoder
        tgt_lens = src_lens - 1
        decoder_output_src = self.src_decode(tgt_data, encoder_h_n, encoder_c_n, tgt_lens)  # [batch_size, seq_len, hidden_size * 2]

        # sd decoder
        tgt_embed = self.embedding(tgt_data)  # [batch_size, seq_len, embed_size]
        packed_tgt_embed = nn.utils.rnn.pack_padded_sequence(tgt_embed, tgt_lens.tolist(), batch_first=True, enforce_sorted=False)
        stack_route_rep = torch.stack([route_rep] * (self.num_layers * 2))  # [num_layers, batch_size, embed_size]
        default_c_0 = torch.zeros(self.num_layers * 2, stack_route_rep.size(1), stack_route_rep.size(2)).to(stack_route_rep.device)
        packed_decoder_output_route, _ = self.route_decoder(packed_tgt_embed, (stack_route_rep, default_c_0))
        decoder_output_route, _ = nn.utils.rnn.pad_packed_sequence(packed_decoder_output_route, batch_first=True)  # [batch_size, seq_len, hidden_size * 2]

        return route_rep, decoder_output_src, decoder_output_route, rdn_loss
