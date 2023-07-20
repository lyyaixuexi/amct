import os
import json
import argparse
import copy

def run(mix_json_path,quant_json_path,target_json_path):
    with open(mix_json_path) as f:
        mix_quant_datas=json.load(f)
    with open(quant_json_path) as f:
        quant_datas=json.load(f)
    output_datas=copy.deepcopy(quant_datas)
    for layer_name in quant_datas:
        try:
            output_datas[layer_name]['activation_quant_params']['num_bits']=mix_quant_datas[layer_name]['num_bits']
            output_datas[layer_name]['weight_quant_params']['num_bits']=mix_quant_datas[layer_name]['num_bits']
        except:
            print(f"pass {layer_name}")
            continue

    with open(target_json_path,"w") as f:
        json.dump(output_datas,f)
    print(f"Finish. Save json to {target_json_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--mix_json_path', default='/mnt/cephfs/home/lyy/amct/code/output_mix/output_mix1_qat_mixed_precision.json', type=str)
    parser.add_argument('--quant_json_path', default='/mnt/cephfs/home/lyy/amct/amct_torch_samples/resnet-101/outputs/calibration/tmp/config.json', type=str)
    parser.add_argument('--target_json_path', type=str, default='/mnt/cephfs/home/lyy/amct/code/mix_quant.json')
    args = parser.parse_args()
    run(mix_json_path=args.mix_json_path,quant_json_path=args.quant_json_path,target_json_path=args.target_json_path)