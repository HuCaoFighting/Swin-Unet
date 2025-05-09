# Swin-Unet
[ECCVW2022] The codes for the work "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation"(https://arxiv.org/abs/2105.05537). Our paper has been accepted by ECCV 2022 MEDICAL COMPUTER VISION WORKSHOP (https://mcv-workshop.github.io/). We updated the Reproducibility. I hope this will help you to reproduce the results.

## 1. Download pre-trained swin transformer model (Swin-T)
* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"

## 2. Prepare data

- The datasets we used are provided by TransUnet's authors. [Get processed data in this link] (Synapse/BTCV: https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd and ACDC: https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4).

## 3. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 4. Train/Test

- Run the train script on synapse dataset. The batch size we used is 24. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

- Train

```bash
sh train.sh 
# or 
python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
sh test.sh 
# or 
python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

## Reproducibility


- Codes

Our trained model is stored on the Huawei cloud. The interns do not have the right to send any files out from the internal system, so I can't share our trained model weights. Regarding how to reproduce the segmentation results presented in the paper, we discovered that different GPU types would generate different results. In our code, we carefully set the random seed, so the results should be consistent when trained multiple times on the same type of GPU. If the training does not give the same segmentation results as in the paper, it is recommended to adjust the learning rate. And, the type of GPU we used in this work is Tesla v100. Finaly, pre-training is very important for pure transformer models. In our experiments, both the encoder and decoder are initialized with pretrained weights rather than initializing the encoder with pretrained weights only.

## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)

## Citation

```bibtex
@InProceedings{swinunet,
author = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
title = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
booktitle = {Proceedings of the European Conference on Computer Vision Workshops(ECCVW)},
year = {2022}
}

@misc{cao2021swinunet,
      title={Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation}, 
      author={Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
      year={2021},
      eprint={2105.05537},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
