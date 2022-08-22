from .sceneflow_dataset import SceneFlowDatset
from .sceneflow_dataset_mm import SceneFlowDatset_mm
from .kitti_dataset import KITTIDataset



__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "sceneflow_mm": SceneFlowDatset_mm,
    "kitti": KITTIDataset,
}
