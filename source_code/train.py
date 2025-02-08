import argparse
import logging
import numpy as np
import os
import random
import torch
import torch.nn as nn
from data_utils import DataLoader
from model import Model
from parameters import parameters
from sklearn import metrics
from torch.nn.utils import clip_grad_norm_


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", type=int, help="choose a cuda device")
    parser.add_argument("--cluster_num", type=int, help="set the number of clusters")
    parser.add_argument("--route_num", type=int, help="set the number of normal routes in one SD pair")
    parser.add_argument("--dataset", type=str, help="set the dataset name")
    parser.add_argument("--model_num", type=str, help="set the model name")
    params = parser.parse_args()
    return params


def setup_logger(name, log_file, formatter, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def log(logger, message, print_out=True):
    logger.info(message)
    if print_out:
        print(message)
        

def restruction_loss(criteria, model, hidden_state, tok_data, tok_mask):
    recon_pred = model.fc_output_recon(hidden_state)  # [batch_size, seq_len, vocab_size]

    recon_pred = recon_pred.view(-1, recon_pred.size(-1))  # [batch_size * seq_len, vocab_size]
    tok_data = tok_data.view(-1)  # [batch_size * seq_len]
    tok_mask = tok_mask.view(-1)  # [batch_size * seq_len]
    
    recon_pred = recon_pred[tok_mask.bool()]
    tok_data = tok_data[tok_mask.bool()]
    
    recon_loss = criteria(recon_pred, tok_data)
    recon_loss = torch.mean(recon_loss)
    
    return recon_loss


def src_restruction_loss(criteria, model, hidden_state, tok_data, tok_mask):
    recon_pred = model.fc_src_recon(hidden_state)  # [batch_size, seq_len, vocab_size]

    recon_pred = recon_pred.view(-1, recon_pred.size(-1))  # [batch_size * seq_len, vocab_size]
    tok_data = tok_data.view(-1)  # [batch_size * seq_len]
    tok_mask = tok_mask.view(-1)  # [batch_size * seq_len]
    
    recon_pred = recon_pred[tok_mask.bool()]
    tok_data = tok_data[tok_mask.bool()]

    recon_loss = criteria(recon_pred, tok_data)
    recon_loss = torch.mean(recon_loss)
    
    return recon_loss


def route_restruction_loss(criteria, model, hidden_state, tok_data, tok_mask):
    recon_pred = model.fc_route_recon(hidden_state)  # [batch_size, seq_len, vocab_size]

    recon_pred = recon_pred.view(-1, recon_pred.size(-1))  # [batch_size * seq_len, vocab_size]
    tok_data = tok_data.view(-1)  # [batch_size * seq_len]
    tok_mask = tok_mask.view(-1)  # [batch_size * seq_len]
    
    recon_pred = recon_pred[tok_mask.bool()]
    tok_data = tok_data[tok_mask.bool()]
    
    recon_loss = criteria(recon_pred, tok_data)
    recon_loss = torch.mean(recon_loss)
    
    return recon_loss


def latent_loss(mu_z, logvar_z, z, mu_c, logvar_c, mask):
    cluster_num = mu_c.size(0)
    batch_size = mu_z.size(0)
    mu_z = mu_z.permute(1, 0, 2)  # [seq_len, batch_size, embed_size]
    logvar_z = logvar_z.permute(1, 0, 2)  # [seq_len, batch_size, embed_size]
    z = z.permute(1, 0, 2)  # [seq_len, batch_size, embed_size]
    mask = mask.permute(1, 0)  # [seq_len, batch_size]
    batch_gaus_loss = []
    batch_cate_loss = []
    for i in range(len(z)):
        mask_i = mask[i]  # [batch_size]
        mu_z_i = mu_z[i]  # [batch_size, hidden_size]
        logvar_z_i = logvar_z[i]  # [batch_size, hidden_size]
        z_i = z[i]  # [batch_size, hidden_size]
        stack_z_i = torch.stack([z_i] * cluster_num, dim=1)  # [batch_size, num_cluster, hidden_size]
        stack_mu_zri = torch.stack([mu_z_i] * cluster_num, dim=1)  # [batch_size, num_cluster, hidden_size]
        stack_logvar_zri = torch.stack([logvar_z_i] * cluster_num, dim=1)  # [batch_size, num_cluster, hidden_size]
        stack_mu_c = torch.stack([mu_c] * batch_size)  # [batch_size, num_cluster, hidden_size]
        stack_logvar_c = torch.stack([logvar_c] * batch_size)  # [batch_size, num_cluster, hidden_size]
        # posterior p(z|c)
        pzc_logit = - torch.sum(torch.square(stack_z_i - stack_mu_c) / torch.exp(stack_logvar_c), dim=-1)  # [batch_size, num_cluster]
        pzc = torch.add(torch.nn.functional.softmax(pzc_logit, dim=1), 1e-10)  # [batch_size, num_cluster]
        # gaussian loss
        batch_gaus_loss_i = 0.5 * torch.sum(
            pzc * torch.mean(stack_logvar_c
                       + torch.exp(stack_logvar_zri) / torch.exp(stack_logvar_c)
                       + torch.square(stack_mu_zri - stack_mu_c) / torch.exp(stack_logvar_c), dim=-1)
            , dim=-1) - 0.5 * torch.mean(1 + logvar_z_i, dim=-1)
        batch_gaus_loss_i = batch_gaus_loss_i * mask_i
        batch_gaus_loss.append(batch_gaus_loss_i)
        # categorical loss
        uniform = torch.full_like(pzc, 1.0 / pzc.size(-1))
        batch_cate_loss_i = pzc * (torch.log(pzc) - torch.log(uniform))  # [batch_size, num_cluster]
        batch_cate_loss_i = torch.sum(batch_cate_loss_i, dim=-1)  # [batch_size]
        batch_cate_loss_i = batch_cate_loss_i * mask_i  # [batch_size]
        batch_cate_loss.append(batch_cate_loss_i)
    valid_lens = torch.sum(mask, dim=0)
    batch_gaus_loss = torch.stack(batch_gaus_loss)  # [seq_len, batch_size]
    batch_gaus_loss = torch.sum(batch_gaus_loss, dim=0) / valid_lens  # [batch_size]
    batch_cate_loss = torch.stack(batch_cate_loss)  # [seq_len, batch_size]
    batch_cate_loss = torch.sum(batch_cate_loss, dim=0) / valid_lens  # [batch_size]
    return torch.mean(batch_gaus_loss), torch.mean(batch_cate_loss)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(params, args):
    # # set seed for repeatable result
    # set_seed(int(args.model_num))

    # logger
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger_train = setup_logger("training_log", "log_files/{}/log_training_cluster{}_routeNum{}_{}.log"
                                .format(args.dataset, args.cluster_num, args.route_num, args.model_num), formatter)
    logger_train.info("TRAINING START")

    # data
    trajs_file_path_train = params.data_folder + params.trajs_file_train
    data_loader_train = DataLoader(params.batch_size, trajs_file_path_train, args.dataset)
    data_loader_train.load()
    log(logger_train, "Total trajectories loaded for training: {}".format(len(data_loader_train.src)))
    trajs_file_path_val = params.data_folder + params.trajs_file_val
    data_loader_val = DataLoader(params.eval_batch_size, trajs_file_path_val, args.dataset, evaluation=True)
    data_loader_val.load()
    log(logger_train, "Total trajectories loaded for validation: {}".format(len(data_loader_val.src)))

    # model
    model = Model(params.vocab_size, params.embed_size, params.hidden_size,
                  params.num_layers, params.dropout,
                  args.cluster_num, args.route_num,
                  args.dataset)

    if args.cuda_device != -1 and torch.cuda.is_available():
        log(logger_train, "Training with GPU")
        torch.cuda.set_device(args.cuda_device)
        model.cuda()
    else:
        log(logger_train, "Training with CPU")
    
    # optimier
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    
    # load model
    if os.path.isfile(params.checkpoint_file):
        log(logger_train, "Loading checkpoint: {}".format(params.checkpoint_file))
        checkpoint = torch.load(params.checkpoint_file)
        start_epoch = checkpoint["epoch"] + 1
        best_val_score = checkpoint["best_val_score"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        early_stop_count = checkpoint["early_stop_count"]
        log(logger_train, "Restore training from epoch {}".format(start_epoch))
    else:
        log(logger_train, "No check point at: {}".format(params.checkpoint_file))
        start_epoch = 1
        best_val_score= 0
        early_stop_count = 0

    # training
    for epoch in range(start_epoch, params.epoch_num + 1):
        if early_stop_count >= 10:
                break
        model.train()
        epoch_recon_loss, epoch_src_recon_loss, epoch_route_recon_loss, epoch_gaus_loss, epoch_cate_loss, epoch_rdn_loss = 0, 0, 0, 0, 0, 0
        batch_num = 0
        for data_batch in data_loader_train.iterate_data():
            batch_num += 1
            
            optimizer.zero_grad()

            batch_src, batch_tok, batch_tgt = data_batch
            if args.cuda_device != -1 and torch.cuda.is_available():
                torch.cuda.set_device(args.cuda_device)
                src_data = batch_src[0].cuda()
                src_mask = batch_src[1].cuda()
                src_lens = batch_src[2].cuda()
                tgt_data = batch_tgt[0].cuda()
                tgt_mask = batch_tgt[1].cuda()
                tgt_lens = batch_tgt[2].cuda()
                tok_data = batch_tok[0].cuda()
                tok_mask = batch_tok[1].cuda()
                tok_lens = batch_tok[2].cuda()

            decoder_output, decoder_output_src, decoder_output_route, mu_z, logvar_z, z, mu_c, logvar_c, rdn_loss = model(src_data, src_lens, tgt_data)

            # reconstruction loss
            criteria = nn.CrossEntropyLoss(reduction="none")
            recon_loss = restruction_loss(criteria, model, decoder_output, tok_data, tok_mask) * params.recon_loss_weight
            src_recon_loss = src_restruction_loss(criteria, model, decoder_output_src, tok_data, tok_mask) * params.src_recon_loss_weight
            route_recon_loss = route_restruction_loss(criteria, model, decoder_output_route, tok_data, tok_mask) * params.src_recon_loss_weight

            # latent_loss
            gaus_loss, cate_loss = latent_loss(mu_z, logvar_z, z, mu_c, logvar_c, src_mask)
            gaus_loss = gaus_loss * params.gaussian_loss_weight
            cate_loss = cate_loss * params.cate_loss_weight

            # RDN loss
            rdn_loss = rdn_loss * params.rdn_loss_weight

            # total loss
            total_loss = recon_loss + src_recon_loss + route_recon_loss + gaus_loss + cate_loss + rdn_loss
            total_loss.backward()

            clip_grad_norm_(model.parameters(), params.max_grad_norm)

            optimizer.step()

            epoch_recon_loss += recon_loss.item()
            epoch_src_recon_loss += src_recon_loss.item()
            epoch_route_recon_loss += route_recon_loss.item()
            epoch_gaus_loss += gaus_loss.item()
            epoch_cate_loss += cate_loss.item()
            epoch_rdn_loss += rdn_loss.item()

        epoch_recon_loss = epoch_recon_loss / batch_num
        epoch_src_recon_loss = epoch_src_recon_loss / batch_num
        epoch_route_recon_loss = epoch_route_recon_loss / batch_num
        epoch_gaus_loss = epoch_gaus_loss / batch_num
        epoch_cate_loss = epoch_cate_loss / batch_num
        epoch_rdn_loss = epoch_rdn_loss / batch_num

        epoch_total_loss = epoch_recon_loss + epoch_src_recon_loss + epoch_route_recon_loss + epoch_gaus_loss + epoch_cate_loss + epoch_rdn_loss

        log(logger_train,
            "Epoch {}: recon_loss: {}, src_recon_loss: {}, route_recon_loss: {}, gaus_loss: {}, cate_loss: {}, rdn_loss: {}, total loss: {}".
            format(epoch, epoch_recon_loss, epoch_src_recon_loss, epoch_route_recon_loss, epoch_gaus_loss, epoch_cate_loss, epoch_rdn_loss, epoch_total_loss))

        # validation
        label_true = []
        anomaly_score_vals = []
        model.eval()
        for data_batch in data_loader_val.iterate_data():
            batch_src, batch_tok, batch_tgt, batch_label_true = data_batch
            if args.cuda_device != -1 and torch.cuda.is_available():
                torch.cuda.set_device(args.cuda_device)
                src_data = batch_src[0].cuda()
                src_mask = batch_src[1].cuda()
                src_lens = batch_src[2].cuda()
                tgt_data = batch_tgt[0].cuda()
                tgt_mask = batch_tgt[1].cuda()
                tgt_lens = batch_tgt[2].cuda()
                tok_data = batch_tok[0].cuda()
            
            anomaly_scores = model.evaluate(src_data, src_lens, tgt_data, tok_data)
            lengths = batch_tgt[2]
            batch_score_pred = []
            for i in range(len(lengths)):
                score_list = anomaly_scores[i][:lengths[i]].tolist()
                score_list = [0.0] + score_list[:-1] + [0.0]
                batch_score_pred.append(score_list)
            label_true += batch_label_true
            anomaly_score_vals += batch_score_pred
        flattened_label_true = [x for sublist in label_true for x in sublist]
        flattened_anomaly_scores = [x for sublist in anomaly_score_vals for x in sublist]
        val_score = metrics.average_precision_score(flattened_label_true, flattened_anomaly_scores, pos_label=1)
        log(logger_train, "Epoch {}: validation score: {}".format(epoch, val_score))
        model.train()
        
        # save best model
        if val_score > best_val_score:
            early_stop_count = 0
            best_val_score = val_score
            log(logger_train,
                "Epoch {}: best model at epoch {}, validation score: {}".
                format(epoch, epoch, best_val_score))
            state = {
                "epoch": epoch,
                "best_val_score": best_val_score,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "early_stop_count": early_stop_count
            }
            torch.save(state, params.best_checkpoint_file)
        else:
            early_stop_count += 1
        
        # save model
        state = {
            "epoch": epoch,
            "best_val_score": best_val_score,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "early_stop_count": early_stop_count
        }
        torch.save(state, params.checkpoint_file)


if __name__ == "__main__":
    args = define_args()
    paras = parameters(args.dataset)
    paras.data_folder = "../data/{}/".format(args.dataset)
    paras.trajs_file_train = "training_data/training_data.pkl"
    paras.trajs_file_val = "training_data/val_data.pkl"
    paras.best_checkpoint_file = "saved_models/{}/bestModel_cluster{}_routeNum{}_{}.pt".format(args.dataset, args.cluster_num, args.route_num, args.model_num)
    paras.checkpoint_file = "saved_models/{}/model_cluster{}_routeNum{}_{}.pt".format(args.dataset, args.cluster_num, args.route_num, args.model_num)
    train(paras, args)
