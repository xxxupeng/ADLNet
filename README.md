## An Adaptive Multi-Modal Cross-Entropy Loss for Stereo Matching

#### Abstract

Despite the great success of deep learning in stereo matching, recovering accurate and clearly-contoured disparity map is still challenging. In this paper, we propose a simple yet effective adaptive multi-modal cross-entropy loss for training the networks. The proposed loss is able to encourage the models to separately learn the intuitive bimodal and regular unimodal distributions on the edge and non-edge area. Under the supervision of this clear pattern, models are prone to yielding bimodal distributions with distinct primary and secondary peaks in ambiguous regions, facilitating the post-processing to achieve a higher disparity accuracy. In this way, the classical stereo matching model is able to regain competitiveness with the state-of-the-art method. We validate our method with experiments on three popular stereo matching datasets SceneFlow, KITTI 2012 and 2015. Various backbones are further employed to demonstrate the generality of our method.


