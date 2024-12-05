from constants import constants
import pickle
import torch


class DataLoader:
    def __init__(self, batch_size, trajs_file_path, dataset, evaluation=False):
        self.cnst = constants(dataset)

        self.trajs_file_path = trajs_file_path
        self.batch_size = batch_size

        self.evaluation = evaluation

        self.sd_pair_dict = None
        self.src = []
        self.tok = []
        self.trg = []

        self.test_dataset = None
        self.test_labels = None
        self.test_trajs = None

    def load(self):
        if self.evaluation:
            self.test_dataset = pickle.load(open(self.trajs_file_path, "rb"))
            self.test_labels = self.test_dataset[0]
            self.test_trajs = self.test_dataset[1]
            for traj in self.test_trajs:
                traj = list(traj)
                self.tok.append(traj[1:])
                self.src.append(traj)
                self.trg.append(traj[:-1])  
            print("Total trajectories loaded: {}".format(len(self.src)))
        else:
            trajs = pickle.load(open(self.trajs_file_path, "rb"))
            for i in range(len(trajs)):
                traj = list(trajs[i])
                self.tok.append(traj[1:])
                self.src.append(traj)
                self.trg.append(traj[:-1])  
            print("Total trajectories loaded: {}".format(len(self.src)))

    def padding(self, src_data, tok_data, trg_data, label_data=None):
        index = [x for x, _ in sorted(enumerate(src_data), key=lambda x: len(x[1]), reverse=True)]
        src_data = [src_data[i] for i in index]
        tok_data = [tok_data[i] for i in index]
        trg_data = [trg_data[i] for i in index]
        src_length = torch.tensor([len(traj) for traj in src_data])
        tok_length = torch.tensor([len(traj) for traj in tok_data])
        trg_length = torch.tensor([len(traj) for traj in trg_data])
        src_max_len = max(len(x) for x in src_data)
        tok_max_len = max(len(x) for x in tok_data)
        trg_max_len = max(len(x) for x in trg_data)
        src_mask = [[1] * len(x) + [0] * (src_max_len - len(x)) for x in src_data]
        tok_mask = [[1] * len(x) + [0] * (tok_max_len - len(x)) for x in tok_data]
        trg_mask = [[1] * len(x) + [0] * (trg_max_len - len(x)) for x in trg_data]
        src_data = [x + [self.cnst.PAD] * (src_max_len - len(x)) for x in src_data]
        tok_data = [x + [self.cnst.PAD] * (tok_max_len - len(x)) for x in tok_data]
        trg_data = [x + [self.cnst.PAD] * (trg_max_len - len(x)) for x in trg_data]
        src_pack = (src_data, src_length, src_mask)
        tok_pack = (tok_data, tok_length, tok_mask)
        trg_pack = (trg_data, trg_length, trg_mask)
        if self.evaluation:
            label_data = [label_data[i] for i in index]
            return src_pack, tok_pack, trg_pack, label_data
        else:
            return src_pack, tok_pack, trg_pack

    def iterate_data(self):
        for sta_idx in range(0, len(self.src), self.batch_size):
            if self.evaluation:
                end_idx = min(sta_idx + self.batch_size, len(self.src))
                batch_src_data = self.src[sta_idx:end_idx]
                batch_tok_data = self.tok[sta_idx:end_idx]
                batch_trg_data = self.trg[sta_idx:end_idx]
                batch_label_data = self.test_labels[sta_idx:end_idx]
                batch_src_pack, batch_tok_pack, batch_trg_pack, batch_label_data = \
                    self.padding(batch_src_data, batch_tok_data, batch_trg_data, label_data=batch_label_data)
                batch_src_data = torch.LongTensor(batch_src_pack[0]).t()
                batch_tok_data = torch.LongTensor(batch_tok_pack[0]).t()
                batch_trg_data = torch.LongTensor(batch_trg_pack[0]).t()
                batch_src_mask = torch.LongTensor(batch_src_pack[2])
                batch_tok_mask = torch.LongTensor(batch_tok_pack[2])
                batch_trg_mask = torch.LongTensor(batch_trg_pack[2])
                batch_src = [batch_src_data.t(), batch_src_mask, batch_src_pack[1]]
                batch_tok = [batch_tok_data.t(), batch_tok_mask, batch_tok_pack[1]]
                batch_trg = [batch_trg_data.t(), batch_trg_mask, batch_trg_pack[1]]
                yield batch_src, batch_tok, batch_trg, batch_label_data
            else:
                end_idx = min(sta_idx + self.batch_size, len(self.src))
                batch_src_data = self.src[sta_idx:end_idx]
                batch_tok_data = self.tok[sta_idx:end_idx]
                batch_trg_data = self.trg[sta_idx:end_idx]
                batch_src_pack, batch_tok_pack, batch_trg_pack = \
                    self.padding(batch_src_data, batch_tok_data, batch_trg_data)
                batch_src_data = torch.LongTensor(batch_src_pack[0]).t()
                batch_tok_data = torch.LongTensor(batch_tok_pack[0]).t()
                batch_trg_data = torch.LongTensor(batch_trg_pack[0]).t()
                batch_src_mask = torch.LongTensor(batch_src_pack[2])
                batch_tok_mask = torch.LongTensor(batch_tok_pack[2])
                batch_trg_mask = torch.LongTensor(batch_trg_pack[2])
                batch_src = [batch_src_data.t(), batch_src_mask, batch_src_pack[1]]
                batch_tok = [batch_tok_data.t(), batch_tok_mask, batch_tok_pack[1]]
                batch_trg = [batch_trg_data.t(), batch_trg_mask, batch_trg_pack[1]]
                yield batch_src, batch_tok, batch_trg
                