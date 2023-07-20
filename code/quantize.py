import amct_pytorch as amct
import torchvision
import torch
from amct_pytorch.common.auto_calibration import AutoCalibrationEvaluatorBase

evaluator = amct.ModelEvaluator(
    data_dir="/mnt/cephfs/home/lyy/amct/data/imagenet_train/",
    input_shape="input:1,3,224,224", 
    data_types="float32")

model = torchvision.models.resnet101(pretrained=False)
model.load_state_dict(torch.load("/mnt/cephfs/home/lyy/amct/amct_torch_samples/resnet-101/model/resnet101-5d3b4d8f.pth"))
input_data = tuple([torch.randn((1,3,224,224))])
amc_config = './amc.cfg'
save_dir = 'output_mix/output_mix1'
amct.auto_mixed_precision_search(model=model, input_data=input_data, config=amc_config, save_dir=save_dir, evaluator=evaluator, sensitivity='MseSimilarity')