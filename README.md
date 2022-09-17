# An Adaptive Multi-Modal Cross-Entropy Loss for Stereo Matching

## Abstract

Despite the great success of deep learning in stereo matching, recovering accurate and clearly-contoured disparity map is still challenging. In this paper, we propose a simple yet effective adaptive multi-modal cross-entropy loss for training the networks. The proposed loss is able to encourage the models to separately learn the intuitive bimodal and regular unimodal distributions on the edge and non-edge area. Under the supervision of this clear pattern, models are prone to yielding bimodal distributions with distinct primary and secondary peaks in ambiguous regions, facilitating the post-processing to achieve a higher disparity accuracy. In this way, the classical stereo matching model is able to regain competitiveness with the state-of-the-art method. We validate our method with experiments on three popular stereo matching datasets SceneFlow, KITTI 2012 and 2015. Various backbones are further employed to demonstrate the generality of our method.

## Environment

- python == 3.9.12

- pytorch == 1.11.0

- numpy == 1.21.5

- Nvidia RTX 2080Ti

## Datasets

- [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

- [KITTI 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

- [KITTI 2012](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)

Download the three datasets, and change the `datapath` args in `./scripts/sceneflow.sh` or `./scripts.kitti.sh`.

## Training

We use the distributed data parallel (DDP) to train the model.

Please exec the bash shell in `./scripts/`.

```bash
/bin/bash ./scripts/sceneflow.sh
/bin/bash ./scripts/kitti.sh
```

Training logs are saved in ```./log/```.

## Evaluation

please change the ```train_DDP.py``` in shell scripts to `val.py` and uncomment the `loadmodel` args.

EPE, 1px, 3px, D1 metrics are reported.
