import argparse
import geopandas as gpd
import networkx as nx
import os
import osmnx as ox
import pickle as pkl
import random
import sys
import time
# from tqdm.notebook import tqdm
from tqdm import tqdm


def define_args():
    """
    定义参数
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--city", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--adjlist_path", type=str)
    parser.add_argument("--detour_dictList_output_path", type=str)
    parser.add_argument("--test_dictList_output_path", type=str)
    parser.add_argument("--val_dictList_output_path", type=str)
    parser.add_argument("--train_dictList_output_path", type=str)
    parser.add_argument("--test_output_path", type=str)
    parser.add_argument("--val_output_path", type=str)
    parser.add_argument("--train_output_path", type=str)
    
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--test_percent", type=float)
    parser.add_argument("--val_percent", type=float)
    parser.add_argument("--train_percent", type=float)

    parser.add_argument("--switch_percent", type=float)
    parser.add_argument("--offset", type=int)
    parser.add_argument("--connect", type=int)

    return parser.parse_args()


def load_data(data_path):
    """
    读取数据
    """
    dataset = pkl.load(open(data_path, "rb"))
    return dataset


def generate_graph(adjlist_path):
    """
    根据邻接列表创建图
    """
    graph = nx.read_adjlist(adjlist_path, create_using=nx.DiGraph, nodetype=int)
    return graph


def plot_road_network(edges_shp_path, nodes_shp_path, plot_path, trajs=None, edge_color="blue"):
    """
    绘制地图
    """ 
    edges_gdf = gpd.read_file(edges_shp_path)
    edges_gdf = edges_gdf.set_index(['u','v','key'])
    nodes_gdf = gpd.read_file(nodes_shp_path)

    edges_color = ["black"] * len(edges_gdf)
    edges_gdf["color"] = edges_color

    if trajs is not None:

        # print()
        # check = 51
        # trajs = [trajs[1][0:check]]
        # print(trajs[0])
        
        for traj in trajs:
            for index, row in edges_gdf.iterrows():
                if row["fid"] in traj:
                    edges_gdf.at[index, "color"] = edge_color

    G = ox.graph_from_gdfs(gdf_nodes=nodes_gdf, gdf_edges=edges_gdf)
    fig, ax = ox.plot_graph(G, node_size=0, edge_linewidth=1.5, edge_color=edges_gdf["color"], edge_alpha=0.5, bgcolor="white", filepath=plot_path, save=True, dpi=600)
    return G


def create_SD_traj_count_dict(dataset):
    """
    创建字典, {sd: {traj: count, ...}, ...}
    """
    sd_traj_count_dict = {}  # {sd_pair: {traj: count}}
    for traj in dataset:
        s = traj[0]
        d = traj[-1]
        sd_pair = (s, d)
        traj = tuple(traj)
        if sd_pair in sd_traj_count_dict:
            if traj in sd_traj_count_dict[sd_pair]:
                sd_traj_count_dict[sd_pair][traj] += 1
            else:
                sd_traj_count_dict[sd_pair][traj] = 1
        else:
            sd_traj_count_dict[sd_pair] = {traj: 1}
    return sd_traj_count_dict


def create_SD_traj_dict(dataset):
    """
    创建字典, {sd: {traj, ...}, ...}
    """
    sd_traj_dict = {}  # {sd_pair: {traj}}
    for traj in dataset:
        s = traj[0]
        d = traj[-1]
        sd_pair = (s, d)
        traj = tuple(traj)
        if sd_pair in sd_traj_dict:
            sd_traj_dict[sd_pair].add(traj)
        else:
            sd_traj_dict[sd_pair] = {traj}
    return sd_traj_dict


def anomaly_injection_switch(random_seed, sd_traj_count_dict, test_percent, G, switch_percent, connect):
    random.seed(random_seed)

    # 选择一部分 SD
    sd_keys = list(sd_traj_count_dict.keys())
    sd_number = round(len(sd_keys) * test_percent)
    selected_sd_ksys = random.sample(sd_keys, sd_number)

    # switch main part
    switch_anomaly_dict = []
    for sd in tqdm(selected_sd_ksys):
        traj_count_list = list(sd_traj_count_dict[sd].items())  # [(traj, count)]
        random.shuffle(traj_count_list)
        if len(traj_count_list) < 2:
            continue
        
        for traj_count_1 in traj_count_list:
            for traj_count_2 in traj_count_list:
                if traj_count_1 == traj_count_2:
                    continue
                # traj1 and traj2
                traj1 = traj_count_1[0]
                traj2 = traj_count_2[0]
                # switch length
                traj1_switch_length = round(len(traj1) * switch_percent)
                traj2_switch_length = round(len(traj2) * switch_percent)
                # random switch point
                traj1_sp1 = random.randrange(1, len(traj1) - traj1_switch_length)
                traj1_sp2 = traj1_sp1 + traj1_switch_length
                traj2_sp1 = random.randrange(1, len(traj2) - traj2_switch_length)
                traj2_sp2 = traj2_sp1 + traj2_switch_length
                # switch
                traj1_part1 = traj1[:traj1_sp1 + 1]
                traj1_part2 = traj1[traj1_sp2:]
                traj2_switch_part = traj2[traj2_sp1:traj2_sp2 + 1]
                switch_traj = None
                if not connect:
                    switch_traj = traj1_part1 + traj2_switch_part + traj1_part2
                else:
                    r1 = traj1_part1[-1]
                    r2 = traj2_switch_part[0]
                    r3 = traj2_switch_part[-1]
                    r4 = traj1_part2[0]
                    if nx.has_path(G, r1, r2) and nx.has_path(G, r3, r4):
                        paths1 = list(nx.all_shortest_paths(G, r1, r2))
                        paths2 = list(nx.all_shortest_paths(G, r3, r4))
                        path1 = random.sample(paths1, 1)[0]
                        path2 = random.sample(paths2, 1)[0]
                        switch_traj = traj1_part1 + tuple(path1[1:-1]) + traj2_switch_part + tuple(path2[1:-1]) + traj1_part2
                if switch_traj is not None:
                    tmp_dict = {
                        "traj_1": traj1,
                        "traj_2": traj2,
                        "switch_traj": switch_traj,
                    }
                    switch_anomaly_dict.append(tmp_dict)       

    return switch_anomaly_dict


def generate_labels_switch(switch_anomaly_dict, sd_traj_dict, switch_percent, random_seed=1):
    random.seed(random_seed)

    for tmp_dict in switch_anomaly_dict:
        traj_1 = tmp_dict["traj_1"]
        traj_2 = tmp_dict["traj_2"]
        switch_traj = tmp_dict["switch_traj"]
        sd = (switch_traj[0], switch_traj[-1])
        sd_trajs = sd_traj_dict[sd]

        # 标记头部和尾部的正常部分
        simi_traj_dict = {}
        traj_label_dict = {}
        for traj in sd_trajs:
            label = [1] * len(switch_traj)
            for i in range(len(traj)):
                if i >= len(switch_traj):
                    break
                normal_road = traj[i]
                anomaly_road = switch_traj[i]
                if normal_road != anomaly_road:
                    break
                label[i] = 0
            for i in range(-1, -len(traj)-1, -1):
                if i < -len(switch_traj):
                    break
                normal_road = traj[i]
                anomaly_road = switch_traj[i]
                if normal_road != anomaly_road:
                    break
                label[i] = 0

            if sum(label) == 0:
                continue

            if sum(label) in simi_traj_dict:
                simi_traj_dict[sum(label)].append(traj)
            else:
                simi_traj_dict[sum(label)] = [traj]

            traj_label_dict[traj] = label

        if list(simi_traj_dict.keys()) == []:
            continue

        max_simi = min(list(simi_traj_dict.keys()))
        candidate_orj_trajs = simi_traj_dict[max_simi]  # 找到最相似的轨迹（最少的 1）

        candi_org_traj = []
        for current_org_traj in candidate_orj_trajs:
            current_switch_percent = traj_label_dict[current_org_traj].index(1)
            if (switch_percent - 0.1) <= (current_switch_percent / len(current_org_traj)) <= (switch_percent + 0.1):
                candi_org_traj.append(current_org_traj)
        

        if candi_org_traj == []:
            continue
        current_org_traj = random.sample(candi_org_traj, 1)[0]
        current_label = traj_label_dict[current_org_traj]

        tmp_dict["traj_1"] = current_org_traj
        tmp_dict["label"] = current_label

    return switch_anomaly_dict


def generate_test_val_train(switch_anomaly_dict, val_percent, train_percent, random_seed):
    random.seed(random_seed)

    clean_dict = []
    for tmp_dict in switch_anomaly_dict:
        if "label" in tmp_dict and 1 in tmp_dict["label"]:
            clean_dict.append(tmp_dict)
    test_dict = []
    val_dict = []
    train_dict = []
    test_list = [[], [], [], []]
    val_list = [[], [], [], []]
    train_list = [[], [], [], []]
    total_percent = val_percent + train_percent
    val_percent = val_percent / total_percent
    train_percent = train_percent / total_percent
    select_indices = random.sample(range(0, len(clean_dict)), round(len(clean_dict) * total_percent))
    val_indices = random.sample(select_indices, round(len(select_indices) * val_percent))
    for i in range(len(clean_dict)):
        if i in val_indices:
            val_dict.append(clean_dict[i])
            val_list[0].append(clean_dict[i]["label"])
            val_list[1].append(clean_dict[i]["switch_traj"])
            val_list[2].append(clean_dict[i]["traj_1"])
            val_list[3].append(clean_dict[i]["traj_2"])
        elif i not in val_indices and i in select_indices:
            train_dict.append(clean_dict[i])
            train_list[0].append(clean_dict[i]["label"])
            train_list[1].append(clean_dict[i]["switch_traj"])
            train_list[2].append(clean_dict[i]["traj_1"])
            train_list[3].append(clean_dict[i]["traj_2"])
        elif i not in select_indices:
            test_dict.append(clean_dict[i])
            test_list[0].append(clean_dict[i]["label"])
            test_list[1].append(clean_dict[i]["switch_traj"])
            test_list[2].append(clean_dict[i]["traj_1"])
            test_list[3].append(clean_dict[i]["traj_2"])
    
    return clean_dict, test_dict, val_dict, train_dict, test_list, val_list, train_list
    
        


def main():
    args = define_args()
    if args.connect == 1:
        args.connect = True
    elif args.connect == 0:
        args.connect = False

    # count time
    start_time = time.time()

    # load data
    dataset = load_data(args.data_path)
    print("Number of trajectory:", len(dataset))
    road_number = 0
    for traj in dataset:
        road_number += len(traj)
    print("Number of road:", road_number)

    # generate graph and check id
    graph = generate_graph(args.adjlist_path)
    road_set = set(list(graph.nodes))
    print("Minimum road id:", min(road_set))
    print("Maximum road id:", max(road_set))

    # SD traj count dict
    sd_traj_count_dict = create_SD_traj_count_dict(dataset)  # {sd_pair: {traj: count}}
    # SD traj dict
    sd_traj_dict = create_SD_traj_dict(dataset)  # {sd_pair: {traj}}
    print("Number of SD-pair:", len(sd_traj_count_dict))
    print("Average trajectories in SD:", len(dataset) / len(sd_traj_dict))

    # switch anomaly
    # dictionary list: [{"traj_1", "traj_2", "switch_traj"}]
    switch_anomaly_dict = anomaly_injection_switch(args.random_seed, sd_traj_count_dict, args.test_percent, graph, args.switch_percent, args.connect)
    print("Number of switch:", len(switch_anomaly_dict))

    # save anomaly
    pkl.dump(switch_anomaly_dict, open("./{}/test_data/unlabeled_test_data/switch_anomaly_dict_sp{}_c{}.pkl".format(args.city, args.switch_percent, args.connect), "wb"))
    print("Switch trajectory writen: ./{}/test_data/unlabeled_test_data/switch_anomaly_dict_sp{}_c{}.pkl".format(args.city, args.switch_percent, args.connect))

    # load anomaly
    switch_anomaly_dict = pkl.load(open("./{}/test_data/unlabeled_test_data/switch_anomaly_dict_sp{}_c{}.pkl".format(args.city, args.switch_percent, args.connect), "rb"))
    print("Switch trajectory load: ./{}/test_data/unlabeled_test_data/switch_anomaly_dict_sp{}_c{}.pkl".format(args.city, args.switch_percent, args.connect))
    print("Number of switch trajectory:", len(switch_anomaly_dict))

    # label_anomaly
    # dictionary list: [{"traj_1", "traj_2", "switch_traj", ""label"}]
    switch_anomaly_dict = generate_labels_switch(switch_anomaly_dict, sd_traj_dict, args.switch_percent)
    
    # test data and validation data
    clean_dict, test_dict, val_dict, train_dict, test_list, val_list, train_list = generate_test_val_train(switch_anomaly_dict, args.val_percent, args.train_percent, args.random_seed)
    print("Number of valid switch:", len(clean_dict))
    print("Number of test dataset:", len(test_dict))
    print("Number of validation dataset:", len(val_dict))
    print("Number of training dataset:", len(train_dict))

    # store test data and validation data
    parameter_str = "_sp" + str(args.switch_percent) + "_c" + str(args.connect)
    # pkl.dump(clean_dict, open(args.switch_dictList_output_path + parameter_str, "wb"))
    # pkl.dump(test_dict, open(args.test_dictList_output_path + parameter_str, "wb"))
    # pkl.dump(val_dict, open(args.val_dictList_output_path + parameter_str, "wb"))
    # pkl.dump(train_dict, open(args.train_dictList_output_path + parameter_str, "wb"))
    pkl.dump(test_list, open(args.test_output_path + parameter_str, "wb"))
    pkl.dump(val_list, open(args.val_output_path + parameter_str, "wb"))
    # pkl.dump(train_list, open(args.train_output_path + parameter_str, "wb"))

    # count time
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    print("Total time:", total_time, "seconds")
    print()


if __name__ == "__main__":
    main()
