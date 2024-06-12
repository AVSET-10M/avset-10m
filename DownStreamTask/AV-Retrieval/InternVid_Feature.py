import numpy as np
import os
import cv2
import torch
from tqdm import tqdm
import csv
import sys

try:
    from viclip import get_viclip, retrieve_text, _frame_from_video
except:
    from .viclip import get_viclip, retrieve_text, _frame_from_video

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def normalize(data):
    return (data / 255.0 - v_mean) / v_std


def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert (len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).float()
    # vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube


def load_batch(video_list, model):
    videos = []
    for video_path in video_list:
        video = cv2.VideoCapture(video_path)
        frames = [x for x in _frame_from_video(video)]
        frames_tensor = frames2tensor(frames, device=torch.device('cuda'))
        videos.append(frames_tensor[0])
    videos = torch.stack(videos).to("cuda")
    vid_feat = get_vid_feat(videos, clip)
    print(vid_feat.size())
    return vid_feat


def get_vid_feat(frames, clip):
    return clip.get_vid_features(frames)


# modify xxx to the path of the pretrained model
model_cfgs = {
    'viclip-l-internvid-10m-flt': {
        'size': 'l',
        'pretrained': 'xxx/ViCLIP-L_InternVid-FLT-10M.pth',
    },
    'viclip-l-internvid-200m': {
        'size': 'l',
        'pretrained': '/nfs/chengxize.cxz/projects/InternVideo-main/checkpoints/ViCLIP-L_InternVid-200M.pth',
    },
    'viclip-b-internvid-10m-flt': {
        'size': 'b',
        'pretrained': 'xxx/ViCLIP-B_InternVid-FLT-10M.pth',
    },
    'viclip-b-internvid-200m': {
        'size': 'b',
        'pretrained': 'xxx/ViCLIP-B_InternVid-200M.pth',
    },
}

cfg = model_cfgs['viclip-l-internvid-200m']
model_l = get_viclip(cfg['size'], cfg['pretrained'])
clip = model_l['viclip']
clip = clip.to("cuda")

model_name = 'InternVid'
batch_size = 10

root = '{root_dir}'
total_files = 0
video_files = []
for file in os.listdir(root):
    csv_path = os.path.join(root, file)
    with open(csv_path) as f:
        for item in tqdm(csv.reader(f)):
            video_path = item[0].replace('audio', 'video').replace('.wav', '.mp4')
            video_files.append(video_path)
            total_files += 1

rank = int(sys.argv[1])

for idx in tqdm(range(0, total_files, batch_size)):
    if idx % 1 != rank:
        continue
    file_paths = video_files[idx:min(total_files, idx + batch_size)]
    video_embeddings = load_batch(file_paths, clip)
    for video_embedding, file_path in zip(video_embeddings, file_paths):
        os.makedirs(os.path.dirname(file_path.replace('video', f'{model_name}/video').replace('.mp4', '.npy')),
                    exist_ok=True)
        np.save(file_path.replace('video', f'{model_name}/video').replace('.mp4', '.npy'),video_embedding.cpu().numpy())