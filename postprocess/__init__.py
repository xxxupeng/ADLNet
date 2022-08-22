from postprocess.disparity_regression import *

__disparity_regression__ = {
    "mean": mean_disparityregression,
    "hard": hard_unimodal_disparityregression,
    "monotony": unimodal_disparityregression,
}
