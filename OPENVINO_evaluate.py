import nncf
from openvino.tools import mo
from openvino.runtime import serialize
import torch
from pathlib import Path
import logging
import os
from zipfile import ZipFile
from multiprocessing.pool import ThreadPool
import yaml
from itertools import repeat
import time
import platform

from tqdm.notebook import tqdm
from ultralytics.yolo.utils.metrics import ConfusionMatrix
import torch
import numpy as np


def my_test(model, core, data_loader:torch.utils.data.DataLoader, validator, num_samples:int = None):
    """
    OpenVINO YOLOv8 model accuracy validation function. Runs model validation on dataset and returns metrics
    Parameters:
        model (Model): OpenVINO model
        data_loader (torch.utils.data.DataLoader): dataset loader
        validato: instalce of validator class
        num_samples (int, *optional*, None): validate model only on specified number samples, if provided
    Returns:
        stats: (Dict[str, float]) - dictionary with aggregated accuracy metrics statistics, key is metric name, value is metric value
    """
    validator.seen = 0
    validator.jdict = []
    validator.stats = []
    validator.batch_i = 1
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
    model.reshape({0: [1, 3, -1, -1]})
    num_outputs = len(model.outputs)
    compiled_model = core.compile_model(model)
    for batch_i, batch in enumerate(tqdm(data_loader, total=num_samples)):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)
        results = compiled_model(batch["img"])
        if num_outputs == 1:
            preds = torch.from_numpy(results[compiled_model.output(0)])
        else:
            preds = [torch.from_numpy(results[compiled_model.output(0)]), torch.from_numpy(results[compiled_model.output(1)])]
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
    stats = validator.get_stats()
    return stats


def print_stats(stats:np.ndarray, total_images:int, total_objects:int):
    """
    Helper function for printing accuracy statistic
    Parameters:
        stats: (Dict[str, float]) - dictionary with aggregated accuracy metrics statistics, key is metric name, value is metric value
        total_images (int) -  number of evaluated images
        total objects (int)
    Returns:
        None
    """
    # print("Boxes:")
    mp, mr, map50, mean_ap = stats['metrics/precision(B)'], stats['metrics/recall(B)'], stats['metrics/mAP50(B)'], stats['metrics/mAP50-95(B)']
    # Print results
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95')
    print('Boxes:',s)
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('\t\t\tall', total_images, total_objects, mp, mr, map50, mean_ap))
    if 'metrics/precision(M)' in stats:
        # print("Masks:")
        s_mp, s_mr, s_map50, s_mean_ap = stats['metrics/precision(M)'], stats['metrics/recall(M)'], stats['metrics/mAP50(M)'], stats['metrics/mAP50-95(M)']
        # Print results
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95')
        print("Masks:",s)
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        print(pf % ('\t\t\tall', total_images, total_objects, s_mp, s_mr, s_map50, s_mean_ap))

if __name__ == "__main__":
    from ultralytics.yolo.utils import DEFAULT_CFG
    from ultralytics.yolo.cfg import get_cfg
    from ultralytics.yolo.data.utils import check_det_dataset
    from ultralytics import YOLO
    from ultralytics.yolo.utils import ops
    from openvino.runtime import Core
    MODEL_PATH = f"runs/segment/train5/weights"
    IR_MODEL_NAME = "v8_seg"
    core = Core()

    pt_path = f"{MODEL_PATH}/best.pt"
    #将需要量化的onnx模型路径串起来
    # onnx_path = f"{MODEL_PATH}/{MODEL_NAME}"

    #定义量化后的模型的路径名称
    FP32_path = f"{MODEL_PATH}/FP32_openvino_model/{IR_MODEL_NAME}_FP32.xml"
    FP16_path = f"{MODEL_PATH}/FP16_openvino_model/{IR_MODEL_NAME}_FP16.xml"
    Int8_path = f"{MODEL_PATH}/Int8_openvino_model/{IR_MODEL_NAME}_Int8.xml"

    CFG_PATH = 'ultralytics/yolo/cfg/default.yaml'
    NUM_TEST_SAMPLES = 300

    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = str(CFG_PATH)

    fp32_model = core.read_model(FP32_path)
    fp16_model = core.read_model(FP16_path)
    quantized_model = core.read_model(Int8_path)
    seg_model = YOLO(pt_path)

    seg_validator = seg_model.ValidatorClass(args=args)
    seg_validator.data = check_det_dataset('ultralytics/datasets/coco128-seg.yaml')
    seg_data_loader = seg_validator.get_dataloader("datasets/coco128-seg", 1)
    seg_validator.is_coco = True
    seg_validator.class_map = [1,2,3,4,5]
    seg_validator.names = seg_model.model.names
    seg_validator.metrics.names = seg_validator.names
    seg_validator.nc = seg_model.model.model[-1].nc
    seg_validator.nm = 32
    seg_validator.process = ops.process_mask
    seg_validator.plot_masks = []
    # seg_data_loader = create_data_source()

    fp32_seg_stats = my_test(fp32_model, core, seg_data_loader, seg_validator, num_samples=None)
    fp16_seg_stats = my_test(fp16_model, core, seg_data_loader, seg_validator, num_samples=None)
    int8_seg_stats = my_test(quantized_model, core, seg_data_loader, seg_validator, num_samples=None)

    print("FP32 model accuracy")
    print_stats(fp32_seg_stats, seg_validator.seen, seg_validator.nt_per_class.sum())

    print("FP16 model accuracy")
    print_stats(fp16_seg_stats, seg_validator.seen, seg_validator.nt_per_class.sum())

    print("INT8 model accuracy")
    print_stats(int8_seg_stats, seg_validator.seen, seg_validator.nt_per_class.sum())