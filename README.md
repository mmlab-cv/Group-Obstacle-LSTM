# Group-Obstacle-LSTM
Code for the papers Group-LSTM (ECCV2018) and Group-Obstacle-LSTM (CVIU2020).

This repo contains a Tensorflow implementation for our [ECCV paper](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11131/Bisagno_Group_LSTM_Group_Trajectory_Prediction_in_Crowded_Scenarios_ECCVW_2018_paper.pdf) and our [CVIU paper](https://www.sciencedirect.com/science/article/pii/S1077314220301454?casa_token=mArBjZiDVYoAAAAA:BkQvs7yJcF8YhXfHpYtmxlfQOUvyyRz94symyUV8jm90D8sS3rgLgwd9DConXMbzqFWgB5xD5w).  If you find this code useful in your research, please consider citing:


    @inproceedings{bisagno2018group,
    title={Group lstm: Group trajectory prediction in crowded scenarios},
    author={Bisagno, Niccol{\'o} and Zhang, Bo and Conci, Nicola},
    booktitle={Proceedings of the European conference on computer vision (ECCV)},
    pages={0--0},
    year={2018}
    }
    
    @article{bisagno2020embedding,
    title={Embedding group and obstacle information in LSTM networks for human trajectory prediction in crowded scenes},
    author={Bisagno, Niccol{\'o} and Saltori, Cristiano and Zhang, Bo and De Natale, Francesco GB and Conci, Nicola},
    journal={Computer Vision and Image Understanding},
    pages={103126},
    year={2020},
    publisher={Elsevier}
    }


This code was tested on an Ubuntu 14.04 system using Pytorch 0.3.1

The data files are organised as:
-frame_id
-Pedestrian_id
-x_position
-y_position
-group_id

If group_id=0, the pedestrian of interest does not belong to any group.

