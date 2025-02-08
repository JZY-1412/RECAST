# RECAST

This is the implementation code of RECAST. The implementation is based on Python 3.10 and PyTorch 2.1.1. The source_code folder contains the implementation of the RECAST. The data_preprocessing folder contains the implementation of data preprocessing and anomalous sub-trajectory injection.

# Data

The $data.zip$ file contains the preprocessed training data, validation data, and a sample of test data for the Porto dataset. The original Porto dataset is available at: https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data.

# Training

train.py contains the code for the training process. To train the model, please follow the command:
```
train.py --cuda_device=${cuda_device} --cluster_num=${cluster_num} --route_num=${route_num} --dataset=${dataset} --model_num=${model_num}
```

# Evaluation

evaluation.py contains the code for the evaluation process. To evaluate the model, please follow the command:
```
python evaluation.py --cuda_device=${cuda_device} \
                     --cluster_num=${cluster_num} --route_num=${route_num} \
                     --dataset=${dataset} --anomaly_type=${anomaly_type} \
                     --detour_percent=${switch_percent} --offset=${offset} --connect=${connect} \
                     --model_num=${model_num}
```
