import argparse
import pickle as pkl


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, help="set the city")
    return parser.parse_args()


def preprocessing(args):
    # raw_data_path = "./" + args.city + "/raw_data/" + args.city + "_mm_cpath.pkl"
    raw_data_path = "./" + args.city + "/raw_data/" + args.city + "_id_tms_cpath_dict.pkl"
    dataset = pkl.load(open(raw_data_path, "rb"))
    
    print("Orginal trajectory number: {}".format(len(dataset)))

    trajs = []
    traj_count_dict = {}
    for traj in dataset:
        traj = dataset[traj]["cpath"]
        if type(traj) != int:
            source = traj[0]
            dest = traj[-1]
            source_index = len(traj) - 1 - traj[::-1].index(source)
            dest_index = traj.index(dest)
            traj = traj[source_index:dest_index + 1]
            if len(traj) < 5:
                continue
            if traj in traj_count_dict:
                traj_count_dict[traj] += 1
            else:
                traj_count_dict[traj] = 1
            trajs.append(traj)
    clean_trajs = []
    lens_list = []
    
    for traj in trajs:
        if traj_count_dict[traj] >= 3:
            clean_trajs.append(traj)
            lens_list.append(len(traj))
    trajs = clean_trajs

    print("Trajectory number: {}".format(len(trajs)))
    average_length = sum(lens_list) / len(trajs)
    min_length = min(lens_list)
    max_length = max(lens_list)
    print("Trajectory average length: {}".format(average_length))
    print("Trajectory minimum length: {}".format(min_length))
    print("Trajectory maximum length: {}".format(max_length))

    output_data_path = "./" + args.city + "/training_data/training_data.pkl"
    pkl.dump(trajs, open(output_data_path, "wb"))


def main():
    args = define_args()
    preprocessing(args)


if __name__ == "__main__":
    main()
