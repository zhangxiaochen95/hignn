# hignn
Code implementation for the paper ``Scalable Power Control/Beamforming in Heterogeneous Wireless Networks with Graph Neural Networks''.

***Note:*** *This paper has been submitted to IEEE. Currently, no license is added to permit copy, distribution, or modification in any form.*

- `envs.py` defines classes of heterogeneous wireless channels and provides an implementation of the closed-form FP algorithm in heterogeneous settings.
- `gen_data.py` generates datasets for training/test.
- `utils.py` includes functions shared by both `train_hignn.py` and `train_dnn.py`.
- `nn_modules.py` defines the neural network (NN) modules.
- `train_hignn.py` is the main file carrying out the training-loop of heterogeneous interference graph neural network (HIGNN).
- `train_dnn.py` is the main file carrying out the training-loop of deep neural networks (DNNs) as comparison.
