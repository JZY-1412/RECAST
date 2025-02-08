import os
import pickle as pkl
import random


def main():
    # city = "porto"
    city = "beijing"
    folder = "./{}/training_data/val_datasets".format(city)
    val_data = [[], [], []]
    train_data = [[], [], []]
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if "val_data.pkl_" in file_path:
            val_data_list = pkl.load(open(file_path, "rb"))
            val_data[0] += val_data_list[0]
            val_data[1] += val_data_list[1]
            val_data[2] += val_data_list[2]
        elif "train_label_data.pkl_" in file_path:
            train_data_list = pkl.load(open(file_path, "rb"))
            train_data[0] += train_data_list[0]
            train_data[1] += train_data_list[1]
            train_data[2] += train_data_list[2]
    print("Number of validation data:", len(val_data[0]))
    print("Number of training data:", len(train_data[0]))
    pkl.dump(val_data, open("./{}/training_data/val_data.pkl".format(city), "wb"))
    # pkl.dump(train_data, open(folder + "train_data_label.pkl", "wb"))

main()
