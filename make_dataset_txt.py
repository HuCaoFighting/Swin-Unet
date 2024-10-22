from collections import defaultdict
from itertools import chain
from os.path import join, split, exists
import numpy as np
import os

import pandas as pd
from deep_utils import DirUtils
from argparse import ArgumentParser
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--split", action="store_true")
parser.add_argument("--name", default="datasets", type=str)
parser.add_argument("--n_jobs", default=10, type=int)
parser.add_argument("--data", default=".npz", type=str)
parser.add_argument("--train", action="store_true")
parser.add_argument("--nnunet",
                    default="/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/")

args = parser.parse_args()

seed = 1234


def chain(lst: list[list]):
    out = []
    for l in lst:
        out.extend(l)
    return out


def npz_csv():
    datasets_config = {
        # 'CT_CORONARY': {
        #     'data_dir': f'{args.nnunet}/Dataset002_china_narco/nnUNetPlans_2d',
        #     'num_classes': 3 + 1,  # plus background
        #     'predict_head': 1
        # },
        'MRI_MM': {
            'data_dir': f'{args.nnunet}/Dataset001_mm/nnUNetPlans_2d',
            'num_classes': 3 + 1,  # plus background
            'predict_head': 0
        },
    }

    samples = []
    columns = ["data_dir", "predict_head", "n_classes"]

    for dataset_name, config in datasets_config.items():
        data_files = DirUtils.list_dir_full_path(config['data_dir'], interest_extensions=args.data)
        split_path = config['data_dir'] + "_split"
        if exists(split_path):
            data = DirUtils.list_dir_full_path(split_path, return_dict=True, interest_extensions=".npz")
            seg_img_samples = dict()
            for key, val in tqdm(data.items(), desc="getting data"):
                item = key.replace("_seg", "").replace("_img", "")
                seg_img_samples[item] = val

            file_samples = defaultdict(list)
            for key, val in tqdm(seg_img_samples.items(), desc="Getting final data"):
                item = "_".join(k for k in key.split("_")[:-1])
                file_samples[item].append(val)
        else:
            file_samples = []
        if args.split:
            split_path = DirUtils.split_extension(config['data_dir'], suffix="_split")
            os.makedirs(split_path, exist_ok=True)
        else:
            split_path = None
        print("Getting ready for the data splitting!")
        samples_ = Parallel(n_jobs=args.n_jobs)(
            delayed(process_file)(config, split_path, filepath, file_samples) for filepath in tqdm(data_files))
        samples.extend(samples_)

    train, val = train_test_split(samples)
    csv_file_path = f'./lists/{args.name}/'

    train = chain(train)
    val = chain(val)
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    pd.DataFrame(train, columns=columns).to_csv(csv_file_path + "/train.txt", index=False)
    pd.DataFrame(val, columns=columns).to_csv(csv_file_path + "/val.txt", index=False)


def process_file(config, split_path, filepath, file_samples):
    filename = split(filepath)[-1].replace(".npz", "")
    if split_path and filename not in file_samples:
        # print(filename)
        samples = []
        file_data = np.load(filepath)
        img = file_data['data']
        seg = file_data['seg']
        for z_index in range(img.shape[1]):
            img_ = img[:, z_index, ...]
            seg_ = seg[:, z_index, ...]
            img_path = join(split_path,
                            f"{DirUtils.split_extension(split(filepath)[-1], suffix=f'_{z_index:04}')}")
            # seg_path = join(split_path,
            #                 f"{DirUtils.split_extension(split(filepath)[-1], suffix=f'_{z_index:04}_seg')}")
            if not exists(img_path):
                seg_ = seg_.squeeze(0)
                seg_[seg_ < 0] = 0
                np.savez(img_path, image=img_.squeeze(0), label=seg_)
            samples.append(
                [img_path,
                 config['predict_head'],
                 config['num_classes'],
                 ]
            )
            # np.savez(seg_path, seg_)
    else:
        samples = [[
            filepath,
            config['predict_head'],
            config['num_classes'],
        ]]

    return samples


if __name__ == '__main__':
    npz_csv()
