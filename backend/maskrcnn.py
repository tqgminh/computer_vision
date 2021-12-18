# You may need to restart your runtime prior to this, to let your installation take effect
from detectron2.utils.visualizer import ColorMode
import logging
from detectron2.evaluation import COCOEvaluator
import os
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2
import numpy as np
import detectron2
from PIL import Image
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries


class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def mask_rcnn(img_path, save_path, device="cpu"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.MODEL.WEIGHTS = "model/model_final.pth"
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

    predictor = DefaultPredictor(cfg)

    im = cv2.imread(img_path)
    outputs = predictor(im)

    board_metadata = MetadataCatalog.get(
        "pubdal6_train").set(thing_classes=["Cat", "Dog"])

    v = Visualizer(im[:, :, ::-1],
                   metadata=board_metadata,
                   scale=1,
                   instance_mode=ColorMode.IMAGE
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    data = v.get_image()[:, :, ::-1]
    img = Image.fromarray(data, 'RGB')
    img.save(save_path)


if __name__ == '__main__':
    mask_rcnn("dog.jpg", "mask-rcnn/res.jpg")
