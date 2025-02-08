import argparse
import geopandas as gpd
import networkx as nx
import os
import osmnx as ox
import pickle as pkl
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
# from tqdm.notebook import tqdm
from tqdm import tqdm


def define_args():
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
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--test_percent", type=float)
    parser.add_argument("--val_percent", type=float)
    parser.add_argument("--train_percent", type=float)

    parser.add_argument("--detour_percent", type=float)
    parser.add_argument("--offset", type=int)
    parser.add_argument("--connect", type=int)

    return parser.parse_args()


def load_data(data_path):
    dataset = pkl.load(open(data_path, "rb"))
    return dataset


def generate_graph(adjlist_path):
    graph = nx.read_adjlist(adjlist_path, create_using=nx.DiGraph, nodetype=int)
    return graph


def plot_road_network(edges_shp_path, nodes_shp_path, plot_path, trajs=None, edge_color="blue"):
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


def find_all_shortest_paths(G, source, target):
    try:
        paths = list(nx.all_shortest_paths(G, source, target))
        return (source, target, paths)
    except nx.NetworkXNoPath:
        return (source, target, [])


def run_find_all_shortest_paths(args):
    G, source, target = args
    return find_all_shortest_paths(G, source, target)


def create_SD_traj_count_dict(dataset):
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


def create_SD_normalRoad_dict(sd_traj_count_dict, threshold):
    sd_normalRoad_dict = {}  # {sd_pair: {normal_road}}
    for sd in sd_traj_count_dict:
        traj_count_dict = sd_traj_count_dict[sd]
        traj_total_num = sum(list(traj_count_dict.values()))
        normal_road_set = set()
        for traj in traj_count_dict:
            if traj_count_dict[traj] / traj_total_num > threshold:
                for road in traj:
                    normal_road_set.add(road)
        sd_normalRoad_dict[sd] = normal_road_set
    return sd_normalRoad_dict


def anomaly_injection_detour(random_seed, sd_traj_count_dict, test_percent, G, detour_percent, offset, connect):
    random.seed(random_seed)
    
    sd_keys = list(sd_traj_count_dict.keys())
    sd_number = round(len(sd_keys) * test_percent)
    selected_sd_ksys = random.sample(sd_keys, sd_number)

    detour_anomaly_dict = []
    for sd in tqdm(selected_sd_ksys):
        traj_count_list = list(sd_traj_count_dict[sd].items())  # [(traj, count)]
        traj_count = random.sample(traj_count_list, 1)[0]
        traj = traj_count[0]
        
        detour_length = round(len(traj) * detour_percent)
        tolerance_lengths = [round(len(traj) * (detour_percent - 0.05)), round(len(traj) * (detour_percent + 0.05))]

        p1 = random.randrange(1, len(traj) - detour_length)
        p2 = p1 + detour_length
        r1 = traj[p1]
        r2 = traj[p2]

        k_hop_neighbors_1 = []
        for n, length in nx.single_source_shortest_path_length(G, r1, cutoff=offset).items():
            if length == offset:
                k_hop_neighbors_1.append(n)
        k_hop_neighbors_2 = []
        G_r = G.reverse()
        for n, length in nx.single_source_shortest_path_length(G_r, r2, cutoff=offset).items():
            if length == offset:
                k_hop_neighbors_2.append(n)
        if len(k_hop_neighbors_1) == 0 or len(k_hop_neighbors_2) == 0:
            continue

        detour_path = None
        connect_set = []
        for dr1 in k_hop_neighbors_1:
            for dr2 in k_hop_neighbors_2:
                if nx.has_path(G, dr1, dr2):
                    connect_set.append((G, dr1, dr2))
        if not connect:
            dr1_dr2_paths = []

            for dr_tuple in connect_set:
                dr1 = dr_tuple[1]
                dr2 = dr_tuple[2]
                paths = list(nx.all_shortest_paths(G, dr1, dr2))
                for path in paths:
                    if tolerance_lengths[0] <= len(path) <= tolerance_lengths[1]:
                        dr1_dr2_paths.append(path)
                if len(dr1_dr2_paths) != 0:
                    detour_path = random.sample(dr1_dr2_paths, 1)[0]
                    break
        else:
            # new_connect_set = []
            # for dr_tuple in connect_set:
            #     new_connect_set.append((G, dr_tuple[0], dr_tuple[1]))
            # with ProcessPoolExecutor() as executor:
            #     results = list(executor.map(lambda p: find_all_shortest_paths(*p), new_connect_set))
            dr1_dr2_paths = []
            for dr_tuple in connect_set:
                r1_dr1_paths = [d for d in nx.all_shortest_paths(G, r1, dr1)]
                dr2_r2_paths = [d for d in nx.all_shortest_paths(G, dr2, r2)]
                r1_dr1_path = random.sample(r1_dr1_paths, 1)[0]
                dr2_r2_path = random.sample(dr2_r2_paths, 1)[0]
                tolerance_lengths_1 = tolerance_lengths[0] - len(r1_dr1_path[1:-1]) - len(dr2_r2_path[1:-1])
                tolerance_lengths_2 = tolerance_lengths[1] - len(r1_dr1_path[1:-1]) - len(dr2_r2_path[1:-1])
                dr1 = dr_tuple[0]
                dr2 = dr_tuple[1]
                paths = list(nx.all_shortest_paths(G, dr1, dr2))
                for path in paths:
                    if tolerance_lengths_1 <= len(path) <= tolerance_lengths_2:
                        dr1_dr2_paths.append(path)
                if len(dr1_dr2_paths) != 0:
                    dr1_dr2_path = random.sample(dr1_dr2_paths, 1)[0]
                    detour_path = r1_dr1_path[1:-1] + dr1_dr2_path + dr2_r2_path[1:-1]
                    break

        # combine original trajectory with the detour path
        if detour_path is not None:
            traj = list(traj)
            detour_traj = traj[:p1 + 1] + detour_path + traj[p2:]
            # source = traj[0]
            # dest = traj[-1]
            # source_index = len(detour_traj) - 1 - detour_traj[::-1].index(source)
            # dest_index = detour_traj.index(dest)
            # detour_traj = detour_traj[source_index:dest_index + 1]
            tmp_dict = {
                "org_traj": traj,
                "break_points": (p1, p2),
                "detour_path": detour_path,
                "detour_traj": detour_traj,
            }
            detour_anomaly_dict.append(tmp_dict)
        
    return detour_anomaly_dict


def generate_labels(detour_anomaly_dict, sd_traj_dict, detour_percent, random_seed=1):
    random.seed(random_seed)
    for tmp_dict in detour_anomaly_dict:
        traj = tmp_dict["org_traj"]
        detour_traj = tmp_dict["detour_traj"]
        sd = (traj[0], traj[-1])
        sd_trajs = sd_traj_dict[sd]

        simi_traj_dict = {}
        traj_label_dict = {}
        for traj in sd_trajs:
            label = [1] * len(detour_traj)
            for i in range(len(traj)):
                if i >= len(detour_traj):
                    break
                normal_road = traj[i]
                anomaly_road = detour_traj[i]
                if normal_road != anomaly_road:
                    break
                label[i] = 0
            for i in range(-1, -len(traj)-1, -1):
                if i < -len(detour_traj):
                    break
                normal_road = traj[i]
                anomaly_road = detour_traj[i]
                if normal_road != anomaly_road:
                    break
                label[i] = 0
            if sum(label) in simi_traj_dict:
                if (detour_percent) - 0.1 <= (sum(label) / len(detour_traj)) <= (detour_percent + 0.1):
                    simi_traj_dict[sum(label)].append(traj)
            else:
                if (detour_percent - 0.1) <= (sum(label) / len(detour_traj)) <= (detour_percent + 0.1):
                    simi_traj_dict[sum(label)] = [traj]
            
            traj_label_dict[traj] = label
        
        if list(simi_traj_dict.keys()) == []:
            continue
        max_simi = min(list(simi_traj_dict.keys()))
        candidate_orj_trajs = simi_traj_dict[max_simi]  # 找到最相似的轨迹（最少的 1）
        current_org_traj = random.sample(candidate_orj_trajs, 1)[0]
        current_label = traj_label_dict[current_org_traj]

        tmp_dict["org_traj"] = current_org_traj
        tmp_dict["label"] = current_label
    return detour_anomaly_dict


def generate_test_val_train(detour_anomaly_dict, val_percent, train_percent, random_seed):
    random.seed(random_seed)

    clean_dict = []
    for tmp_dict in detour_anomaly_dict:
        if "label" in tmp_dict and 1 in tmp_dict["label"]:
            clean_dict.append(tmp_dict)
    
    test_dict = []
    val_dict = []
    train_dict = []
    test_list = [[], [], []]
    val_list = [[], [], []]
    train_list = [[], [], []]
    total_percent = val_percent + train_percent
    val_percent = val_percent / total_percent
    train_percent = train_percent / total_percent
    select_indices = random.sample(range(0, len(clean_dict)), round(len(clean_dict) * total_percent))
    val_indices = random.sample(select_indices, round(len(select_indices) * val_percent))
    for i in range(len(clean_dict)):
        if i in val_indices:
            val_dict.append(clean_dict[i])
            val_list[0].append(clean_dict[i]["label"])
            val_list[1].append(clean_dict[i]["detour_traj"])
            val_list[2].append(clean_dict[i]["org_traj"])
        elif i not in val_indices and i in select_indices:
            train_dict.append(clean_dict[i])
            train_list[0].append(clean_dict[i]["label"])
            train_list[1].append(clean_dict[i]["detour_traj"])
            train_list[2].append(clean_dict[i]["org_traj"])
        elif i not in select_indices:
            test_dict.append(clean_dict[i])
            test_list[0].append(clean_dict[i]["label"])
            test_list[1].append(clean_dict[i]["detour_traj"])
            test_list[2].append(clean_dict[i]["org_traj"])
    
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

    # detour anomaly
    # dictionary list: [{"orj_traj", "detour_traj", "detour_path", "break_points", "label"}]
    detour_anomaly_dict = anomaly_injection_detour(args.random_seed, sd_traj_count_dict, args.test_percent, graph, args.detour_percent, args.offset, args.connect)

    # save anomaly
    pkl.dump(detour_anomaly_dict, open("./{}/test_data/unlabeled_test_data/detour_anomaly_dict_dp{}_o{}_c{}.pkl".format(args.city, args.detour_percent, args.offset, args.connect), "wb"))
    print("Detoured trajectory writen: ./{}/test_data/unlabeled_test_data/detour_anomaly_dict_dp{}_o{}_c{}.pkl".format(args.city, args.detour_percent, args.offset, args.connect))

    # load anomaly
    detour_anomaly_dict = pkl.load(open("./{}/test_data/unlabeled_test_data/detour_anomaly_dict_dp{}_o{}_c{}.pkl".format(args.city, args.detour_percent, args.offset, args.connect), "rb"))
    print("Detoured trajectory load: ./{}/test_data/unlabeled_test_data/detour_anomaly_dict_dp{}_o{}_c{}.pkl".format(args.city, args.detour_percent, args.offset, args.connect))
    print("Number of detour trajectory:", len(detour_anomaly_dict))
    
    # label anomaly
    detour_anomaly_dict = generate_labels(detour_anomaly_dict, sd_traj_dict, args.detour_percent)
    
    # test data and validation data
    clean_dict, test_dict, val_dict, train_dict, test_list, val_list, train_list = generate_test_val_train(detour_anomaly_dict, args.val_percent, args.train_percent, args.random_seed)
    print("Number of valid detour:", len(clean_dict))
    print("Number of test dataset:", len(test_dict))
    print("Number of validation dataset:", len(val_dict))
    # print("Number of training dataset:", len(train_dict))

    # store test data and validation data
    parameter_str = "_dp" + str(args.detour_percent) + "_o" + str(args.offset) + "_c" + str(args.connect)
    # pkl.dump(clean_dict, open(args.detour_dictList_output_path + parameter_str, "wb"))
    # pkl.dump(test_dict, open(args.test_dictList_output_path + parameter_str, "wb"))
    # pkl.dump(val_dict, open(args.val_dictList_output_path + parameter_str, "wb"))
    # pkl.dump(train_dict, open(args.train_dictList_output_path + parameter_str, "wb"))
    pkl.dump(test_list, open(args.test_output_path + parameter_str, "wb"))
    pkl.dump(val_list, open(args.val_output_path + parameter_str, "wb"))
    print("test data writen:", args.test_output_path + parameter_str)
    print("val data writen:", args.test_output_path + parameter_str)
    # pkl.dump(train_list, open(args.train_output_path + parameter_str, "wb"))

    # count time
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    print("Total time:", total_time, "seconds")
    print()


if __name__ == "__main__":
    main()
