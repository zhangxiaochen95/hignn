# hignn
Code implementation of our paper [Scalable Power Control/Beamforming in Heterogeneous Wireless Networks with Graph Neural Networks](https://ieeexplore.ieee.org/document/9685457).

- `envs.py` defines classes of heterogeneous wireless channels and provides an implementation of the closed-form FP algorithm in heterogeneous settings.
- `gen_data.py` generates datasets for training/test.
- `utils.py` includes functions shared by both `train_hignn.py` and `train_dnn.py`.
- `nn_modules.py` defines the neural network (NN) modules.
- `train_hignn.py` is the main file carrying out the training-loop of *heterogeneous interference graph neural networks (HIGNNs)*.
- `train_dnn.py` is the main file carrying out the training-loop of deep neural networks (DNNs) as comparison.

If you use this code, please cite our work:

```tex
@INPROCEEDINGS{9685457,  
  author={Zhang, Xiaochen and Zhao, Haitao and Xiong, Jun and Liu, Xiaoran and Zhou, Li and Wei, Jibo},  
  booktitle={2021 IEEE Global Communications Conference (GLOBECOM)},   
  title={Scalable Power Control/Beamforming in Heterogeneous Wireless Networks with Graph Neural Networks},   
  year={2021},  
  volume={},  
  number={},  
  pages={01-06},  
  doi={10.1109/GLOBECOM46510.2021.9685457}
}
```

If you have any questions, please contact zhangxiaochen14@nudt.edu.cn.
