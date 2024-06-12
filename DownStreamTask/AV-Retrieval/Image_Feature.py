import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pathlib
import sys


def get_model(model_name="imagebind", device='cpu'):
    from imagebind.models import imagebind_model
    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    return model


def get_feature(model, obj_list, modality='text', device='cpu', model_name='imagebind'):
    from imagebind.models.imagebind_model import ModalityType
    from imagebind import data
    if modality == 'image':
        key = ModalityType.VISION
        value = data.load_and_transform_vision_data(obj_list, device)
    elif modality == 'audio':
        key = ModalityType.AUDIO
        value = data.load_and_transform_audio_data(obj_list, device)
    inputs = {key: value}
    with torch.no_grad():
        embeddings = model(inputs)[key]
    return embeddings


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    root = '{root_dir}'
    model_name = "imagebind"
    batch_size = 10

    audio_files = []
    total_files = 0
    model = get_model(model_name, device)
    rank = 0

    # # # check feature exists
    # # # audio
    total_files = 0
    text_list = []
    for file in os.listdir(root):
        csv_path = os.path.join(root, file)
        with open(csv_path) as f:
            for item in tqdm(csv.reader(f)):
                if os.path.exists(item[0].replace('audio', f'{model_name}/audio').replace('.wav', '.npy')):
                    continue
                text_list.append(item[-1])
                audio_files.append(item[0])
                total_files += 1

    # # audio 提取特征
    for idx in tqdm(range(0, total_files, batch_size)):
        file_paths = audio_files[idx:min(total_files, idx + batch_size)]
        audio_embeddings = get_feature(model, file_paths, modality='audio', device=device, model_name=model_name)
        for audio_embedding, file_path in zip(audio_embeddings, file_paths):
            os.makedirs(os.path.dirname(file_path.replace('audio', f'{model_name}/audio').replace('.wav', '.npy')),
                        exist_ok=True)
            np.save(file_path.replace('audio', f'{model_name}/audio').replace('.wav', '.npy'),
                    audio_embedding.cpu().numpy())


    # image
    total_files = 0
    frames_files = []
    for file in os.listdir(root):
        csv_path = os.path.join(root, file)
        with open(csv_path) as f:
            for item in tqdm(csv.reader(f)):
                for frame_path in os.listdir(item[1]):
                    frames_files.append(os.path.join(item[1], frame_path))
                    total_files += 1

    # # image 提取特征
    for idx in tqdm(range(0, total_files, batch_size)):
        if (idx / batch_size) % 1 != rank:
            continue
        file_paths = frames_files[idx:min(total_files, idx + batch_size)]
        frame_embeddings = get_feature(model, file_paths, modality='image', device=device, model_name=model_name)
        for frame_embedding, file_path in zip(frame_embeddings, file_paths):
            os.makedirs(os.path.dirname(file_path.replace('frames', f'{model_name}/frames').replace('.jpg', '.npy')),
                        exist_ok=True)
            np.save(file_path.replace('frames', f'{model_name}/frames').replace('.jpg', '.npy'),
                    frame_embedding.cpu().numpy())
