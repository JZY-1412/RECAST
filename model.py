import math
import torch
import torch.nn as nn
from constants import constants
from route_dist_net import RouteDistillNet


class Model(nn.Module):
    def __init__(self, 
                 vocab_size, embed_size, hidden_size,
                 num_layers, dropout,
                 cluster_num, route_num,
                 dataset, init_mu_c=None):
        super().__init__()
        self.cnst = constants(dataset)

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=self.cnst.PAD)

        # encoder
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)

        # route distillation
        self.rdn = RouteDistillNet(self.embedding,
                                   embed_size, hidden_size, 
                                   num_layers, 
                                   route_num,
                                   dataset)

        # latent space
        self.fc_mu_z = nn.Linear(hidden_size * 3, embed_size)
        self.fc_logvar_z = nn.Linear(hidden_size * 3, embed_size)
        if init_mu_c is None:
            self.mu_c = torch.nn.Parameter(torch.rand(cluster_num, embed_size))
        else:
            self.mu_c = torch.nn.Parameter(init_mu_c)
        self.logvar_c = torch.nn.Parameter(torch.zeros(cluster_num, embed_size), requires_grad=False)

        # fusion network for route_rep + z
        self.fc_fuse = nn.Linear(embed_size + hidden_size, embed_size)

        # decoder
        self.decoder = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)

        # functions for recontraction
        self.fc_output_recon = nn.Linear(hidden_size * 2, vocab_size)
        self.fc_src_recon = nn.Linear(hidden_size * 2, vocab_size)
        self.fc_route_recon = nn.Linear(hidden_size * 2, vocab_size)

    def encode(self, src, lens):
        # src: [batch_size, seq_len]
        # lens: [batch_size]
        lens = lens.tolist()
        src_embed = self.embedding(src)  # [batch_size, seq_len, embed_size]
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(src_embed, lens, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.encoder(packed_src_embed)  # h_n, c_n: [num_layers * 2, batch_size, hidden_size]
        encoder_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # [batch_size, seq_len, hidden_size * 2]
        return encoder_output, h_n, c_n
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def latent_space(self, encoder_output, route_rep):
        # encoder_output: [batch_size, seq_len, hidden_size * 2]
        # route_rep: [batch_size, hidden_size]
        stack_route_rep = torch.stack([route_rep] * encoder_output.size(1), dim=1)  # [batch_size, seq_len, hidden_size]
        encoder_output = torch.cat([encoder_output, stack_route_rep], dim=-1)  # [batch_size, seq_len, hidden_size * 3]
        mu_z = self.fc_mu_z(encoder_output)  # [batch_size, seq_len, embed_size]
        logvar_z = self.fc_logvar_z(encoder_output)  # [batch_size, seq_len, embed_size]
        z = self.reparameterize(mu_z, logvar_z)  # [batch_size, seq_len, embed_size]
        return mu_z, logvar_z, z
    
    def decode(self, z, h_n, c_n, lens):
        # z: [batch_size, seq_len, embed_size]
        # h_n, c_n: [num_layers * 2, batch_size, hidden_size]
        # lens: [batch_size]
        lens = lens.tolist()
        packed_z = nn.utils.rnn.pack_padded_sequence(z, lens, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.decoder(packed_z, (h_n, c_n))
        decoder_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # [batch_size, seq_len, hidden_size * 2]
        return decoder_output

    def forward(self, src_data, src_lens, tgt_data):
        # src_data, tgt_data: [batch_size, seq_len]; src's seq_len = tgt's seq_len + 1
        # src_lens: [batch_size]
        
        # encoder
        # h_n: [num_layers * 2, batch_size, hidden_size]
        # output: [batch_size, src_seq_len, hidden_size * 2]
        encoder_output, encoder_h_n, encoder_c_n = self.encode(src_data, src_lens)

        # route representation
        # route_rep: [batch_size, hidden_size]
        # decoder_output_src: [batch_size, tgt_seq_len, hidden_size * 2]
        # decoder_output_route: [batch_size, tgt_seq_len, hidden_size * 2]
        route_rep, decoder_output_src, decoder_output_route, rdn_loss = self.rdn(src_data, src_lens, tgt_data)

        # latent space
        mu_z, logvar_z, z = self.latent_space(encoder_output, route_rep)  # all: [batch_size, src_seq_len, embed_size]

        # fuse route_rep + z
        stack_route_rep = torch.stack([route_rep] * z.size(1), dim=1)  # [batch_size, src_seq_len, hidden_size]
        z_rd = torch.cat([z, stack_route_rep], dim=-1)  # [batch_size, src_seq_len, embed_size + hidden_size]
        z_rd = self.fc_fuse(z_rd)  # [batch_size, src_seq_len, embed_size]

        # decoder
        tgt_lens = src_lens - 1
        decoder_output = self.decode(z_rd, encoder_h_n, encoder_c_n, tgt_lens)#  [batch_size, tgt_seq_len, hidden_size * 2]

        return decoder_output, decoder_output_src, decoder_output_route, mu_z, logvar_z, z, self.mu_c, self.logvar_c, rdn_loss
    
    def evaluate(self, src_data, src_lens, tgt_data, tok_data):

        # encoder
        # h_n: [num_layers * 2, batch_size, hidden_size]
        # output: [batch_size, src_seq_len, hidden_size * 2]
        encoder_output, encoder_h_n, encoder_c_n = self.encode(src_data, src_lens)

        # route representation
        route_rep, _, _, _ = self.rdn(src_data, src_lens, tgt_data)  # [batch_size, hidden_size]

        # latent space
        _, _, z = self.latent_space(encoder_output, route_rep)  # all: [batch_size, src_seq_len, embed_size]

        # fuse route_rep + z
        stack_route_rep = torch.stack([route_rep] * z.size(1), dim=1)  # [batch_size, src_seq_len, hidden_size]
        z_rd = torch.cat([z, stack_route_rep], dim=-1)  # [batch_size, src_seq_len, embed_size + hidden_size]
        z_rd = self.fc_fuse(z_rd)  # [batch_size, src_seq_len, embed_size]

        # decoder
        tgt_lens = src_lens - 1
        decoder_output = self.decode(z_rd, encoder_h_n, encoder_c_n, tgt_lens) #  [batch_size, tgt_seq_len, hidden_size * 2]
        
        # anomaly score
        anomaly_score = self.anomaly_score(decoder_output, tok_data)

        return anomaly_score

    def anomaly_score(self, hidden_state, tok_data):
        recon_pred = self.fc_output_recon(hidden_state)  # [batch_size, tgt_seq_len, vocab_size]
        tok_data = tok_data.unsqueeze(-1)
        normal_score = torch.softmax(recon_pred, dim=-1)
        normal_score = torch.gather(normal_score, 2, tok_data).squeeze(-1)
        anomaly_score = 1 - normal_score
        
        return anomaly_score

    def evaluate_2(self, src_data, src_lens, tgt_data, tok_data):

        # encoder
        # h_n: [num_layers * 2, batch_size, hidden_size]
        # output: [batch_size, src_seq_len, hidden_size * 2]
        encoder_output, encoder_h_n, encoder_c_n = self.encode(src_data, src_lens)

        # route representation
        route_rep, _, _, _ = self.rdn(src_data, src_lens, tgt_data)  # [batch_size, hidden_size]

        # latent space
        # _, _, z = self.latent_space(encoder_output, route_rep)  # all: [batch_size, src_seq_len, embed_size]

        # fuse route_rep + z
        # z_rd = self.fc_fuse(z_rd)  # [batch_size, src_seq_len, embed_size]
        # use all cluster to generate, and find the maximum probability
        stack_route_rep = torch.stack([route_rep] * src_data.size(1), dim=1)  # [batch_size, src_seq_len, hidden_size]
        anomaly_score_list = []
        for i in range(len(self.mu_c)):
            z = self.mu_c[i]  # [hidden_size]
            z = torch.stack([z] * src_data.size(0), dim=0)  # [batch_size, hidden_size]
            z = torch.stack([z] * src_data.size(1), dim=1)  # [batch_size, src_seq_len, hidden_size]
            z_rd = torch.cat([z, stack_route_rep], dim=-1)  # [batch_size, src_seq_len, embed_size + hidden_size]
            z_rd = self.fc_fuse(z_rd)  # [batch_size, src_seq_len, embed_size]
            # decoder
            tgt_lens = src_lens - 1
            decoder_output = self.decode(z_rd, encoder_h_n, encoder_c_n, tgt_lens) #  [batch_size, tgt_seq_len, hidden_size * 2]
            # anomaly score
            anomaly_score = self.anomaly_score_2(decoder_output, tok_data)  # [batch_size, seq_len]
            anomaly_score_list.append(anomaly_score)
        anomaly_scores = torch.stack(anomaly_score_list)  # [cluster_number, batch_size, seq_len]
        anomaly_scores, _ = torch.max(anomaly_scores, dim=0)    # [batch_size, seq_len]
        # print(anomaly_scores.size())
        return anomaly_scores

    def anomaly_score_2(self, decoder_output, tok_data):
        log_sigmoid = nn.LogSigmoid()
        weight = torch.nn.functional.embedding(input=tok_data, weight=self.fc_output_recon.weight.data)  # [batch_size, seq_len, hidden_size]
        bias = torch.nn.functional.embedding(input=tok_data, weight=self.fc_output_recon.bias.data.reshape(-1, 1))  # [batch_size, seq_len, 1]
        normal_p = torch.exp(log_sigmoid(torch.sum(weight * decoder_output, dim=-1) + bias.squeeze()))  # [batch_size, seq_len]
        anomaly_p = 1 - normal_p
        return anomaly_p