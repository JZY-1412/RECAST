import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from parameters import parameters
import torch
from data_utils import DataLoader
from model import Model
from sklearn import metrics
import time


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", type=int, help="choose a cuda device")
    parser.add_argument("--cluster_num", type=int, help="set the number of clusters")
    parser.add_argument("--route_num", type=int, help="set the number of normal routes in one SD pair")
    parser.add_argument("--dataset", type=str, help="set the dataset name")
    parser.add_argument("--anomaly_type", type=str, help="choose an anomaly type")
    parser.add_argument("--detour_percent", type=float)
    parser.add_argument("--offset", type=int)
    parser.add_argument("--model_num", type=str, default="", help="choose a model number")
    params = parser.parse_args()
    return params


def load_model(params, args):
    model = Model(params.vocab_size, params.embed_size, params.hidden_size,
                  params.num_layers, params.dropout,
                  args.cluster_num, args.route_num,
                  args.dataset)
    best_model = torch.load(params.best_model_file, map_location=lambda storage, loc: storage.cuda(int(args.cuda_device)))
    model.load_state_dict(best_model["model"])
    epoch_num = best_model["epoch"]
    print("Load model with epoch_num:", epoch_num)
    print("Model loaded: {}".format(params.best_model_file))
    if args.cuda_device != -1 and torch.cuda.is_available():
        print("Evaluation with GPU")
        torch.cuda.set_device(args.cuda_device)
        model.cuda()
    else:
        print("Evaluation with CPU")
    return model


def evaluation(model, data_loader, params, args):
    trajs = []
    label_true = []
    anomaly_score_vals = []
    start_time = time.time()
    model.eval()
    for data_batch in data_loader.iterate_data():
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
            tok_mask = batch_tok[1].cuda()
        anomaly_scores = model.evaluate(src_data, src_lens, tgt_data, tok_data)
        lengths = batch_tgt[2]
        batch_score_pred = []
        for i in range(len(lengths)):
            score_list = anomaly_scores[i][:lengths[i]].tolist()
            score_list = [0.0] + score_list[:-1] + [0.0]
            batch_score_pred.append(score_list)
        label_true += batch_label_true
        anomaly_score_vals += batch_score_pred
        for i in range(len(src_data)):
                traj = src_data[i].tolist()
                trajs.append(traj[0:lengths[i] + 1])
    model.train()
    end_time = time.time()
    eval_time = round(end_time - start_time, 2)
    print("Evaluation time:", eval_time, "seconds")

    flattened_label_true = [x for sublist in label_true for x in sublist]
    flattened_anomaly_scores = [x for sublist in anomaly_score_vals for x in sublist]

    print("Road Segment Number:", len(flattened_label_true))
    negative_number = sum(flattened_label_true)
    positive_number = len(flattened_label_true) - negative_number
    print("Positive label number:", positive_number)
    print("Negative label number:", negative_number)

    print("Anomaly Type:", str(args.anomaly_type))
    if args.anomaly_type == "detour":
        print("Detour Percent:", str(args.detour_percent), "- Offset:", str(args.offset))
    else:
        print("Switch Point:", str(args.detour_percent))

    # precision_recall_curve
    precision, recall, thresholds = metrics.precision_recall_curve(flattened_label_true, flattened_anomaly_scores, pos_label=params.positive_label)
    auc_score = metrics.auc(recall, precision)
    print("Precision-Recall Curve - AUC:", auc_score)
    # average precision score
    ap = metrics.average_precision_score(flattened_label_true, flattened_anomaly_scores, pos_label=params.positive_label)
    print("Average Precision score:", ap)
    # f1 score
    f1_scores = 2 * precision * recall / (precision + recall)
    best_f1_score = f1_scores[np.argmax(f1_scores)]
    print("F1-score:", best_f1_score)
    # best threshold
    best_threshold = thresholds[np.argmax(f1_scores)]
    print("Best threshold:", best_threshold)

    # write metric result
    if not  os.path.exists("evaluation_results/{}/result_cluster{}_routeNum{}_{}/".format(args.dataset, args.cluster_num, args.route_num, args.model_num)):
        os.makedirs("evaluation_results/{}/result_cluster{}_routeNum{}_{}/".format(args.dataset, args.cluster_num, args.route_num, args.model_num))
    if args.anomaly_type == "detour":
        file = open("evaluation_results/{}/result_cluster{}_routeNum{}_{}/model_metric_result_{}_dp{}_o{}_c{}.txt".format(args.dataset, args.cluster_num, args.route_num, args.model_num, args.anomaly_type, args.detour_percent, args.offset), "w")
    else:
        file = open("evaluation_results/{}/result_cluster{}_routeNum{}_{}/model_metric_result_{}_sp{}_c{}.txt".format(args.dataset, args.cluster_num, args.route_num, args.model_num, args.anomaly_type, args.detour_percent), "w")
    file.write("Model loaded: {}".format(params.best_model_file) + "\n")
    file.write("Evaluation Dataset: " + params.trajs_file_test + "\n")
    file.write("Evaluation time: " + str(eval_time) + "seconds" + "\n")
    file.write("Road Segment Number: {}".format(len(flattened_label_true)) + "\n")
    file.write("Positive label number: {}".format(positive_number) + "\n")
    file.write("Negative label number: {}".format(negative_number) + "\n")
    file.write("Cluster Number: " + str(args.cluster_num) + " - Positive Label: " + str(params.positive_label) + "\n")
    file.write("Anomaly Type: " + str(args.anomaly_type) + "\n")
    if args.anomaly_type == "detour":
        file.write("Detour Percent: " + str(args.detour_percent) + " - Offset: " + str(args.offset) + "\n")
    else:
        file.write("Switch Point: " + str(args.detour_percent) + "\n")
    file.write("Precision-Recall Curve - AUC: " + str(auc_score) + "\n")
    file.write("Average Precision: " + str(ap) + "\n")
    file.write("Best F1-score:" + str(best_f1_score) + "\n")
    file.write("Best threshold: " + str(best_threshold) + "\n")
    file.write("\n")
    file.close()
    # write anomaly detection result
    if args.anomaly_type == "detour":
        file = open("evaluation_results/{}/result_cluster{}_routeNum{}_{}/model_detection_result_{}_dp{}_o{}_c{}.csv".format(args.dataset, args.cluster_num, args.route_num, args.model_num, args.anomaly_type, args.detour_percent, args.offset), "w")
    else:
        file = open("evaluation_results/{}/result_cluster{}_routeNum{}_{}/model_detection_result_{}_sp{}_c{}.csv".format(args.dataset, args.cluster_num, args.route_num, args.model_num, args.anomaly_type, args.detour_percent), "w")
    for i in range(len(trajs)):
        file.write(str(trajs[i]).strip("]").strip("[") + "\n")
        file.write(str(label_true[i]).strip("]").strip("[") + "\n")
        label_pred = []
        for anomaly_score in anomaly_score_vals[i]:
            if anomaly_score >= best_threshold:
                label_pred.append(1)
            else:
                label_pred.append(0)
        file.write(str(label_pred).strip("]").strip("[") + "\n")
        file.write(str(anomaly_score_vals[i]).strip("]").strip("[") + "\n")
        file.write("\n")
    file.close()


if __name__ == "__main__":
    args = define_args()

    paras = parameters(args.dataset)

    if args.anomaly_type == "detour":
        paras.trajs_file_test = "test_data.pkl_dp{}_o{}".format(args.detour_percent, args.offset)
    else:
        paras.trajs_file_test = "test_data.pkl_sp{}".format(args.detour_percent)
    trajs_file_path_eval = "../data/{}/".format(args.dataset) + "test_data/" + paras.trajs_file_test
    data_loader = DataLoader(paras.eval_batch_size, trajs_file_path_eval, args.dataset, evaluation=True)
    data_loader.load()
    
    paras.best_model_file = "saved_models/{}/bestModel_cluster{}_routeNum{}_{}.pt".format(args.dataset, args.cluster_num, args.route_num, args.model_num)
    model = load_model(paras, args)

    evaluation(model, data_loader, paras, args)
    print()
