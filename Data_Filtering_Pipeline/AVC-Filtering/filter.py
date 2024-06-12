from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

import os
import csv
from tqdm import tqdm
import torch
import numpy as np
from torch.nn import functional as F
import copy
import sys
import matplotlib.pyplot as plt


def get_filelist(csv_path):
    audio_paths = []
    frame_paths = []
    text_list = []
    f_con = []
    with open(csv_path) as f:
        for item in tqdm(csv.reader(f)):
            audio_path = item[0]
            frame_dir = item[1]
            audio_paths.append(audio_path)
            frame_paths=[os.path.join(frame_dir, file) for file in frame_dir]
            f_con.append(item[2])
    return audio_paths, frame_paths, f_con


if __name__ == '__main__':

    # Instantiate model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    feature_mode = 'imagebind'
    rank = int(sys.argv[1])
    print(rank)
    con = 0
    for csv_file in os.listdir('./data/audioset'):
        audio_feats, frame_feats = [], [], []
        audio_paths, frame_paths, f_con = get_filelist(os.path.join('./data/audioset', csv_file))

        a, v, c, t = [], [], [], []
        for audio_path, frame_path, f_con in tqdm(zip(audio_paths, frame_paths, f_con)):
            try:
                audio_feats.append(F.normalize(torch.tensor(np.load(audio_path)), dim=-1))

                all_feats = []

                for frame_path in frame_paths:
                    feat = np.load(frame_path)
                    feat_tensor = torch.tensor(feat)
                    normalized_feat = F.normalize(feat_tensor, dim=-1)
                    all_feats.append(normalized_feat)

                all_feats_tensor = torch.stack(all_feats)

                mean_feats = all_feats_tensor.mean(dim=0)

                frame_feats.append(mean_feats.numpy())
                a.append(audio_path)
                v.append(frame_path)
                c.append(f_con)

            except Exception as e:
                print(e)
        audio_feats = torch.stack(audio_feats)
        frame_feats = torch.stack(frame_feats)

        n = audio_feats.size(0)
        sim = audio_feats @ frame_feats.T
        with open('./data/audioset_filter/' + csv_file, 'w') as f:
            for i in range(n):
                f.write(f'{a[i]},{v[i]},{c[i]},{t[i]},{sim[i][i]}\n')
