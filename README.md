<p align="right">English | <a href="./README_CN.md">简体中文</a></p>  


<p align="center">
  <img src="images/pi3det.gif" width="12.5%" align="center">

  <h1 align="center">
    <strong>Perspective-Invariant 3D Object Detection</strong>
  </h1>

  <p align="center">
    <a href="https://alanliang.vercel.app/" target="_blank">Ao Liang</a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://ldkong.com/" target="_blank">Lingdong Kong</a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://dylanorange.github.io/" target="_blank">Dongyue Lu</a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="" target="_blank">Youquan Liu</a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="" target="_blank">Jian Fang</a><br>
    <a href="" target="_blank">Huaici Zhao</a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://www.comp.nus.edu.sg/~ooiwt/" target="_blank">Wei Tsang Ooi</a>
  </p>

  <p align="center">
    <a href="https://arxiv.org/abs/2507.17665" target='_blank'>
      <img src="https://img.shields.io/badge/Paper-%F0%9F%93%96-darkred">
    </a>&nbsp;
    <a href="http://pi3det.github.io/" target='_blank'>
      <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-blue">
    </a>&nbsp;
    <a href="https://huggingface.co/datasets/Pi3DET/data" target='_blank'>
      <img src="https://img.shields.io/badge/Dataset-%F0%9F%94%97-green">
    </a>&nbsp;
    <a href="" target='_blank'>
      <img src="https://visitor-badge.laobi.icu/badge?page_id=pi3det.Pi3EDT">
    </a>
  </p>

| <img src="./images/teaser.png" alt="Teaser" width="100%"> |
| :-: |

This work focuses on the practical yet challenging task of 3D object detection from heterogeneous robot platforms: **Vehicle**, **Drone**, and **Quadruped**. To achieve strong generalization ability, we contribute: 
- The **first** dataset for **multi-platform 3D object detection**, comprising more than **51,000+** LiDAR frames with over **250,000+** meticulously annotated 3D bounding boxes.
- A **cross-platform 3D domain adaptation** framework, effectively transferring capabilities from vehicles to other platforms by integrating geometric and feature-level representations.
- A comprehensive benchmark study of **state-of-the-art 3D object detectors** on cross-platform scenarios.


### :books: Citation
If you find this work helpful for your research, please kindly consider citing our papers:

```bibtex
@inproceedings{liang2025pi3det,
    title     = {Perspective-Invariant 3D Object Detection},
    author    = {Ao Liang and Lingdong Kong and Dongyue Lu and Youquan Liu and Jian Fang and Huaici Zhao and Wei Tsang Ooi},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages     = {27725-27738},
    year      = {2025},
}
```
```bibtex
@misc{robosense_challenge_2025,
    title     = {The {RoboSense} Challenge: Sense Anything, Navigate Anywhere, Adapt Across Platforms},
    author    = {Kong, Lingdong and Xie, Shaoyuan and Gong, Zeying and Li, Ye and Chu, Meng and Liang, Ao and Dong, Yuhao and Hu, Tianshuai and Qiu, Ronghe and Li, Rong and Hu, Hanjiang and Lu, Dongyue and Yin, Wei and Ding, Wenhao and Li, Linfeng and Song, Hang and Zhang, Wenwei and Ma, Yuexin and Liang, Junwei and Zheng, Zhedong and Ng, Lai Xing and Cottereau, Benoit R. and Ooi, Wei Tsang and Liu, Ziwei and Zhang, Zhanpeng and Qiu, Weichao and Zhang, Wei and Ao, Ji and Zheng, Jiangpeng and Wang, Siyu and Yang, Guang and Zhang, Zihao and Zhong, Yu and Gao, Enzhu and Zheng, Xinhan and Wang, Xueting and Li, Shouming and Gao, Yunkai and Lan, Siming and Han, Mingfei and Hu, Xing and Malic, Dusan and Fruhwirth-Reisinger, Christian and Prutsch, Alexander and Lin, Wei and Schulter, Samuel and Possegger, Horst and Li, Linfeng and Zhao, Jian and Yang, Zepeng and Song, Yuhang and Lin, Bojun and Zhang, Tianle and Yuan, Yuchen and Zhang, Chi and Li, Xuelong and Kim, Youngseok and Hwang, Sihwan and Jeong, Hyeonjun and Wu, Aodi and Luo, Xubo and Xiao, Erjia and Zhang, Lingfeng and Tang, Yingbo and Cheng, Hao and Xu, Renjing and Ding, Wenbo and Zhou, Lei and Chen, Long and Ye, Hangjun and Hao, Xiaoshuai and Li, Shuangzhi and Shen, Junlong and Li, Xingyu and Ruan, Hao and Lin, Jinliang and Luo, Zhiming and Zang, Yu and Wang, Cheng and Wang, Hanshi and Gong, Xijie and Yang, Yixiang and Ma, Qianli and Zhang, Zhipeng and Shi, Wenxiang and Zhou, Jingmeng and Zeng, Weijun and Xu, Kexin and Zhang, Yuchen and Fu, Haoxiang and Hu, Ruibin and Ma, Yanbiao and Feng, Xiyan and Zhang, Wenbo and Zhang, Lu and Zhuge, Yunzhi and Lu, Huchuan and He, You and Yu, Seungjun and Park, Junsung and Lim, Youngsun and Shim, Hyunjung and Liang, Faduo and Wang, Zihang and Peng, Yiming and Zong, Guanyu and Li, Xu and Wang, Binghao and Wei, Hao and Ma, Yongxin and Shi, Yunke and Liu, Shuaipeng and Kong, Dong and Lin, Yongchun and Yang, Huitong and Lei, Liang and Li, Haoang and Zhang, Xinliang and Wang, Zhiyong and Wang, Xiaofeng and Fu, Yuxia and Luo, Yadan and Etchegaray, Djamahl and Li, Yang and Li, Congfei and Sun, Yuxiang and Zhu, Wenkai and Xu, Wang and Li, Linru and Liao, Longjie and Yan, Jun and Wang, Benwu and Ren, Xueliang and Yue, Xiaoyu and Zheng, Jixian and Wu, Jinfeng and Qin, Shurui and Cong, Wei and He, Yao},
    howpublished = {\url{https://robosense2025.github.io}},
    year      = {2025}
}
```


## Updates
- **[10/2025]** - We have published the baseline models and our key methods for data augmentation. See <a href="https://github.com/robosense2025/track5" target="_blank" rel="noopener noreferrer">GitHub repo</a> for more details on data preparation and installation.
- **[07/2025]** - The **Pi3DET** dataset has been extended to <strong>Track 5: Cross-Platform 3D Object Detection</strong> of the <a href="https://robosense2025.github.io/" target="_blank">RoboSense Challenge</a> at <a href="https://www.iros25.org/" target="_blank">IROS 2025</a>. See the <a href="https://robosense2025.github.io/track5" target="_blank" rel="noopener noreferrer">track homepage</a> and <a href="https://github.com/robosense2025/track5" target="_blank" rel="noopener noreferrer">GitHub repo</a> for more details.
- **[07/2025]** - The [project page](https://pi3det.github.io/) is online. :rocket:
- **[07/2025]** - This work has been accepted to [ICCV 2025](https://iccv.thecvf.com/Conferences/2025). See you in Honolulu! 🌸



## Outline
- [Installation](#gear-installation)
- [Data Preparation](#hotsprings-data-preparation)
- [Getting Started](#rocket-getting-started)
- [Model Zoo](#snake-model-zoo)
- [Pi3DET Benchmark](#triangular_ruler-pi3det-benchmark)
  - [Statistical Analysis](#statistical-analysis)
  - [Methodology](#methodology)
- [Pi3DET Dataset](#pi3det-dataset)
  - [Detailed statistical information](#detailed-statistical-information)
  - [Dataset Examples](#dataset-examples)
- [TODO List](#memo-todo-list)
- [License](#license)
- [Acknowledgements](#acknowledgements)



## :gear: Installation
For details related to installation and environment setups, kindly refer to [INSTALL.md](docs/INSTALL.md).



## :hotsprings: Data Preparation
Kindly refer to our **HuggingFace Dataset** :hugs: page from [here](https://huggingface.co/datasets/Pi3DET/data) for more details.



## :rocket: Getting Started
To learn more usage of this codebase, kindly refer to [GET_STARTED.md](docs/GET_STARTED.md).



## :snake: Model Zoo
To be updated.


## :triangular_ruler: Pi3DET Benchmark
### Statistical Analysis

| <img src="./images/distributions.png" alt="Distribution" width="100%">|
| :-: |

We observe significant cross-platform geometric discrepancies in ego‑motion jitter, point‑cloud elevation distributions, and target pitch‑angle distributions across vehicle, quadruped, and drone platforms, which hinder single‑platform model generalization.


### Methodology

| <img src="./images/framework.png" alt="Framework" width="100%"> |
| :-: |

Pi3DET‑Net employs a two‑stage adaptation pipeline—Pre‑Adaptation uses random jitter and virtual poses to learn and align global geometric transformations, while Knowledge Adaptation leverages geometry‑aware descriptors and KL‑based probabilistic feature alignment to synchronize feature distributions across platforms.  


## Pi3DET Dataset
### Detailed statistical information
| Platform                    | Condition      | Sequence               | # of Frames | # of Points (M) | # of Vehicles | # of Pedestrians |
|-----------------------------|----------------|------------------------|------------:|----------------:|--------------:|-----------------:|
| **Vehicle (8)**             | **Daytime (4)**| city_hall              |      2,982  |           26.61 |       19,489  |          12,199 |
|                             |                | penno_big_loop         |      3,151  |           33.29 |       17,240  |           1,886 |
|                             |                | rittenhouse            |      3,899  |           49.36 |       11,056  |          12,003 |
|                             |                | ucity_small_loop       |      6,746  |           67.49 |       34,049  |          34,346 |
|                             | **Nighttime (4)**| city_hall            |      2,856  |           26.16 |       12,655  |           5,492 |
|                             |                | penno_big_loop         |      3,291  |           38.04 |        8,068  |             106 |
|                             |                | rittenhouse            |      4,135  |           52.68 |       11,103  |          14,315 |
|                             |                | ucity_small_loop       |      5,133  |           53.32 |       18,251  |           8,639 |
|                             |                | **Summary (Vehicle)**  | **32,193**  |      **346.95** |  **131,911**  |      **88,986** |
| **Drone (7)**               | **Daytime (4)**| penno_parking_1        |      1,125  |            8.69 |        6,075  |             115 |
|                             |                | penno_parking_2        |      1,086  |            8.55 |        5,896  |             340 |
|                             |                | penno_plaza            |        678  |            5.60 |          721  |              65 |
|                             |                | penno_trees            |      1,319  |           11.58 |          657  |             160 |
|                             | **Nighttime (3)**| high_beams           |        674  |            5.51 |          578  |             211 |
|                             |                | penno_parking_1        |      1,030  |            9.42 |          524  |             151 |
|                             |                | penno_parking_2        |      1,140  |           10.12 |          83   |             230 |
|                             |                | **Summary (Drone)**    |  **7,052**  |       **59.47** |   **14,534**  |       **1,272** |
| **Quadruped (10)**          | **Daytime (8)**| art_plaza_loop         |      1,446  |           14.90 |           0   |           3,579 |
|                             |                | penno_short_loop       |      1,176  |           14.68 |        3,532  |              89 |
|                             |                | rocky_steps            |      1,535  |           14.42 |           0   |           5,739 |
|                             |                | skatepark_1            |        661  |           12.21 |           0   |             893 |
|                             |                | skatepark_2            |        921  |            8.47 |           0   |             916 |
|                             |                | srt_green_loop         |        639  |            9.23 |        1,349  |             285 |
|                             |                | srt_under_bridge_1     |      2,033  |           28.95 |           0   |           1,432 |
|                             |                | srt_under_bridge_2     |      1,813  |           25.85 |           0   |           1,463 |
|                             | **Nighttime (2)**| penno_plaza_lights   |        755  |           11.25 |          197  |              52 |
|                             |                | penno_short_loop       |      1,321  |           16.79 |          904  |             103 |
|                             |                | **Summary (Quadruped)**| **12,300**  |      **156.75** |    **5,982**  |      **14,551** |
| **All Three Platforms (25)**|                | **Summary (All)**      | **51,545**  |      **563.17** |  **152,427**  |     **104,809** |



### Dataset Examples

| <img src="./images/example1.png" alt="Examples" width="100%"> |
| :-: |

| <img src="./images/example2.png" alt="Examples" width="100%"> |
| :-: |



## :memo: TODO List
- [x] Initial release. 🚀
- [x] Release the dataset for the RoboSense Challenge 2025.
- [x] Release the code for the RoboSense Challenge 2025.
- [ ] Release the whole Pi3DET dataset.
- [ ] Release the code for the Pi3DET-Net method.



## License
This work is under the <a rel="license" href="https://www.apache.org/licenses/LICENSE-2.0">Apache License Version 2.0</a>, while some specific implementations in this codebase might be under other licenses. Kindly refer to [LICENSE.md](docs/LICENSE.md) for a more careful check, if you are using our code for commercial matters.



## Acknowledgements
This work is developed based on the [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) codebase.

> <img src="https://github.com/open-mmlab/mmdetection3d/blob/main/resources/mmdet3d-logo.png" width="31%"/><br>
> MMDetection3D is an open-source toolbox based on PyTorch, towards the next-generation platform for general 3D perception. It is a part of the OpenMMLab project developed by MMLab.

Part of the benchmarked models are from the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [3DTrans](https://github.com/PJLab-ADG/3DTrans) projects.


## Related Projects

| :sunglasses: Awesome | Projects |
|:-:|:-|
| |
| <img width="95px" src="https://github.com/ldkong1205/ldkong1205/blob/master/Images/worldbench_survey.webp"> | **3D and 4D World Modeling: A Survey**<br>[[GitHub Repo](https://github.com/worldbench/survey)] - [[Project Page](https://worldbench.github.io/survey)] - [[Paper](https://worldbench.github.io/assets_common/papers/survey.pdf)] |
| <img width="95px" src="https://github.com/ldkong1205/ldkong1205/blob/master/Images/worldlens.png"> | **WorldLens: Full-Spectrum Evaluations of Driving World Models in Real World**<br>[[GitHub Repo](https://github.com/worldbench/WorldLens)] - [[Project Page](https://worldbench.github.io/worldlens)] - [[Paper](https://worldbench.github.io/assets_common/papers/worldlens.pdf)] |
| <img width="95px" src="https://github.com/ldkong1205/ldkong1205/blob/master/Images/lidarcrafter.png"> | **LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences**<br>[[GitHub Repo](https://github.com/lidarcrafter/toolkit)] - [[Project Page]](https://lidarcrafter.github.io/) - [[Paper](https://arxiv.org/abs/2508.03692)] |
| <img width="95px" src="https://github.com/ldkong1205/ldkong1205/blob/master/Images/3eed.png"> | **3EED: Ground Everything Everywhere in 3D**<br>[[GitHub Repo](https://github.com/worldbench/3EED)] - [[Project Page]](https://project-3eed.github.io/) - [[Paper](https://arxiv.org/abs/2511.01755)] |
| <img width="95px" src="https://github.com/ldkong1205/ldkong1205/blob/master/Images/drivebench.png"> | **Are VLMs Ready for Autonomous Driving? A Study from Reliability, Data & Metric Perspectives**<br>[[GitHub Repo](https://github.com/drive-bench/toolkit)] - [[Project Page]](https://drive-bench.github.io/) - [[Paper](https://arxiv.org/abs/2501.04003)] |
| <img width="95px" src="https://github.com/ldkong1205/ldkong1205/blob/master/Images/dynamiccity.webp"> | **DynamicCity: Large-Scale 4D Occupancy Generation from Dynamic Scenes**<br>[[GitHub Repo](https://github.com/3DTopia/DynamicCity)] - [[Project Page]](https://dynamic-city.github.io/) - [[Paper](https://arxiv.org/abs/2410.18084)] |
| |
