import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
# import h5py as h5
import numpy as np
import os, glob
from imagebind import data
import pandas
from pathlib import Path

# def load_audiocaps_text_data(csv_path):
#     data = pandas.read_csv(csv_path)
#     data = data.values.tolist()
#     data.sort(key=lambda student: student[1])
#     names = []
#     text_data = []
#     for row in data:
#         name = row[1]
#         file = Path(os.path.join(''))
#         if file.exists():
#             text_data.append(row[3])
#             names.append(name)
#     return names, text_data


class Audioset_raw(Dataset):
    def __init__(self):
        # self.names = torch.load('')['names']
        # self.image_embs = torch.load('')['text_embs'].cpu()
        pass
        
    def __getitem__(self, index):
        # inputs = {
        #     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path, torch.device('cpu'))
        # }
        # audio_path = os.path.join('', self.names[index] + '.wav')
        # audio_input = data.load_and_transform_audio_data(audio_path, torch.device('cpu'))
        # image_embs = self.image_embs[index]
        # return audio_input, image_embs
        return torch.randn([1,3,1,128,204]), torch.randn(768)
    def __len__(self):
        return 10000

class Audiocaps_raw(Dataset):
    def __init__(self):
        self.names = torch.load('')['names']
        self.text_embs = torch.load('')['text_embs'].cpu()
        
    def __getitem__(self, index):
        # inputs = {
        #     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path, torch.device('cpu'))
        # }
        audio_path = os.path.join('', self.names[index] + '.wav')
        audio_input = data.load_and_transform_audio_data(audio_path, torch.device('cpu'))
        text_embs = self.text_embs[index]
        return audio_input, text_embs
        # return torch.randn([1,3,1,128,204]), torch.randn(768)
    def __len__(self):
        return 10000

# VGGSS_PATH = ''
VGGSS_PATH = ''
VGGSS_AUDIO_PATH = VGGSS_PATH + 'audio/train/'
VGGSS_VIDEO_PATH = VGGSS_PATH + 'InternVid/video/train/'

VGGSS_AUDIO_TEST = VGGSS_PATH + 'audio/test/'
VGGSS_VIDEO_TEST = VGGSS_PATH + 'InternVid/video/test/'
VGGSS_NAMES_TEST = VGGSS_PATH + 'vggss_test_names.pt'

AVE_PATH = ''
AVE_AUDIO_TEST = AVE_PATH + 'audios/'
AVE_VIDEO_TEST = AVE_PATH + 'InternVid/video/'
AVE_IMAGE_TEST = AVE_PATH + 'AVE_Dataset/images/'


# AS_PATH = ''
AS_PATH = ''
AS_AUDIO_PATH = AS_PATH + 'audio/'
AS_VIDEO_PATH = AS_PATH + 'InternVid/video/'
AS_NAMES_PATH = AS_PATH + 'as_names_check_exists.pt'

class Audio_Visual_train(Dataset):
    def __init__(self, mode = ['AS', 'VGG']):
        # self.names = torch.load('')
        # self.text_embs = torch.load('')['text_embs'].cpu()
        
        # as_names = []
        # for i in range(2):
        #     tmp_names = glob.glob(os.path.join(AS_VIDEO_PATH, 'unbalanced_train_segments_part{0:02}'.format(i),"*.npy"))
        #     tmp_names.sort()
        #     tmp_names = ['unbalanced_train_segments_part{0:02}/'.format(i) + os.path.basename(name)[:-4] for name in tmp_names]
        #     as_names += tmp_names

        # as_names_check = [n if os.path.join(AS_AUDIO_PATH, n + '.wav') for n in as_names]
        as_names = torch.load(AS_NAMES_PATH)

        vgg_names = []
        vgg_sub_dirs = os.listdir(VGGSS_VIDEO_PATH)
        vgg_sub_dirs.sort()
        print(len(vgg_sub_dirs))
        for vgg_sub in vgg_sub_dirs:
            tmp_names = glob.glob(os.path.join(VGGSS_VIDEO_PATH, vgg_sub,"*.npy"))
            tmp_names.sort()
            tmp_names = [vgg_sub+ '/' + os.path.basename(name)[:-4] for name in tmp_names]
            vgg_names += tmp_names

        self.as_audios = [os.path.join(AS_AUDIO_PATH, n + '.wav') for n in as_names]
        self.as_videos = [os.path.join(AS_VIDEO_PATH, n + '.npy') for n in as_names]
        
        self.vgg_audios = [os.path.join(VGGSS_AUDIO_PATH, n + '.wav') for n in vgg_names]
        self.vgg_videos = [os.path.join(VGGSS_VIDEO_PATH, n + '.npy') for n in vgg_names]

        self.audios = []
        self.videos = []

        if 'AS' in mode:
            self.audios += self.as_audios
            self.videos += self.as_videos
        
        if 'VGG' in mode:
            self.audios += self.vgg_audios
            self.videos += self.vgg_videos

        # self.audios = self.as_audios + self.vgg_audios
        # self.videos = self.as_videos + self.vgg_videos

        print(len(self.audios), len(self.videos))

    def __getitem__(self, index):
        # inputs = {
        #     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path, torch.device('cpu'))
        # }
        audio_path = self.audios[index]
        video_path = self.videos[index]
        audio_input = data.load_and_transform_audio_data([audio_path], torch.device('cpu'))
        video_embs = torch.from_numpy(np.load(video_path))
        assert audio_input.shape == torch.Size([1, 3, 1, 128, 204])
        assert video_embs.shape == torch.Size([768])
        return audio_input, video_embs
        # return torch.randn([1,3,1,128,204]), torch.randn(768)
    def __len__(self):
        return len(self.videos)
        # return 10000

class Audio_Visual_test(Dataset):
    def __init__(self, mode = 'VGG'):

        # vgg_names = []
        # vgg_sub_dirs = os.listdir(VGGSS_VIDEO_TEST)
        # vgg_sub_dirs.sort()
        # print(len(vgg_sub_dirs))
        # for vgg_sub in vgg_sub_dirs:
        #     tmp_names = glob.glob(os.path.join(VGGSS_VIDEO_TEST, vgg_sub,"*.npy"))
        #     tmp_names.sort()
        #     tmp_names = [vgg_sub+ '/' + os.path.basename(name)[:-4] for name in tmp_names]
        #     vgg_names += tmp_names
        
        if mode == 'VGG':
            vgg_names = torch.load(VGGSS_NAMES_TEST)

            self.vgg_audios = [os.path.join(VGGSS_AUDIO_TEST, n + '.wav') for n in vgg_names]
            self.vgg_videos = [os.path.join(VGGSS_VIDEO_TEST, n + '.npy') for n in vgg_names]

            self.audios = self.vgg_audios
            self.videos = self.vgg_videos
        if mode == 'AVE':
            self.ave_audios = glob.glob(os.path.join(AVE_AUDIO_TEST,"*.wav"))
            self.ave_audios.sort()
            ave_names = [os.path.basename(n)[:11] for n in self.ave_audios]

            # self.ave_audios = [os.path.join(AVE_AUDIO_TEST, n + '.wav') for n in ave_names]
            self.ave_videos = [os.path.join(AVE_VIDEO_TEST, n + '.npy') for n in ave_names]

            self.audios = self.ave_audios
            self.videos = self.ave_videos

        print(len(self.audios), len(self.videos))
        

    def __getitem__(self, index):
        # inputs = {
        #     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path, torch.device('cpu'))
        # }
        audio_path = self.audios[index]
        video_path = self.videos[index]
        audio_input = data.load_and_transform_audio_data([audio_path], torch.device('cpu'))
        video_embs = torch.from_numpy(np.load(video_path))
        assert audio_input.shape == torch.Size([1, 3, 1, 128, 204])
        assert video_embs.shape == torch.Size([768])
        return audio_input, video_embs
        # return torch.randn([1,3,1,128,204]), torch.randn(768)
    def __len__(self):
        return len(self.videos)

class IB_VGGSound_FT(Dataset):
    def __init__(self):
        vgg = torch.load('')

        vgg_names = vgg['names']
        
        self.vgg_audios = [os.path.join(VGGSS_AUDIO_PATH, n + '.wav') for n in vgg_names]
        self.video_embs = vgg['video_embs']

        self.audios = self.vgg_audios

        print(len(self.audios), len(self.video_embs))

    def __getitem__(self, index):
        # inputs = {
        #     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path, torch.device('cpu'))
        # }
        audio_path = self.audios[index]
        audio_input = data.load_and_transform_audio_data([audio_path], torch.device('cpu'))
        video_embs = self.video_embs[index]
        assert audio_input.shape == torch.Size([1, 3, 1, 128, 204])
        assert video_embs.shape == torch.Size([1024])
        return audio_input, video_embs
        # return torch.randn([1,3,1,128,204]), torch.randn(768)
    def __len__(self):
        return len(self.video_embs)
        # return 10000

class IB_AS_FT(Dataset):
    def __init__(self):
        vgg = torch.load('')

        vgg_names = vgg['names']
        
        self.vgg_audios = [os.path.join(AS_AUDIO_PATH, n + '.wav') for n in vgg_names]
        self.video_embs = vgg['video_embs']

        self.audios = self.vgg_audios

        print(len(self.audios), len(self.video_embs))

    def __getitem__(self, index):
        # inputs = {
        #     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path, torch.device('cpu'))
        # }
        audio_path = self.audios[index]
        audio_input = data.load_and_transform_audio_data([audio_path], torch.device('cpu'))
        video_embs = self.video_embs[index]
        assert audio_input.shape == torch.Size([1, 3, 1, 128, 204])
        assert video_embs.shape == torch.Size([1024])
        return audio_input, video_embs
        # return torch.randn([1,3,1,128,204]), torch.randn(768)
    def __len__(self):
        return len(self.video_embs)
        # return 10000

class IB_AS_VSS_FT(Dataset):
    def __init__(self):
        as_file = torch.load('')
        as_names = as_file['names']
        vgg = torch.load('')
        vgg_names = vgg['names']
        
        self.as_audios  = [os.path.join(AS_AUDIO_PATH, n + '.wav') for n in as_names]
        self.as_video   = as_file['video_embs']
        self.vgg_audios = [os.path.join(VGGSS_AUDIO_PATH, n + '.wav') for n in vgg_names]
        self.vgg_video  = vgg['video_embs']

        self.audios = self.as_audios + self.vgg_audios
        self.video_embs = torch.cat([self.as_video, self.vgg_video])

        print(len(self.audios), len(self.video_embs))

    def __getitem__(self, index):
        # inputs = {
        #     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path, torch.device('cpu'))
        # }
        audio_path = self.audios[index]
        audio_input = data.load_and_transform_audio_data([audio_path], torch.device('cpu'))
        video_embs = self.video_embs[index]
        assert audio_input.shape == torch.Size([1, 3, 1, 128, 204])
        assert video_embs.shape == torch.Size([1024])
        return audio_input, video_embs
        # return torch.randn([1,3,1,128,204]), torch.randn(768)
    def __len__(self):
        return len(self.video_embs)
        # return 10000


class IB_FT_test(Dataset):
    def __init__(self, mode = 'VGG'):
        
        if mode == 'VGG':
            vgg_dict = torch.load('')
            vgg_names = vgg_dict['names']
            self.vgg_audios = [os.path.join(VGGSS_AUDIO_TEST, n + '.wav') for n in vgg_names]
            self.vgg_videos = vgg_dict['video_embs']

            self.audios = self.vgg_audios
            self.videos = self.vgg_videos
        if mode == 'AVE':
            ave_dict = torch.load('')
            ave_names = ave_dict['names']
            self.ave_audios = [os.path.join(AVE_AUDIO_TEST, n + '.wav') for n in ave_names]
            self.ave_videos = ave_dict['video_embs']

            self.audios = self.ave_audios
            self.videos = self.ave_videos

        print(len(self.audios), len(self.videos))
        

    def __getitem__(self, index):
        # inputs = {
        #     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path, torch.device('cpu'))
        # }
        audio_path = self.audios[index]
        # video_path = self.videos[index]
        audio_input = data.load_and_transform_audio_data([audio_path], torch.device('cpu'))
        # video_embs = torch.from_numpy(np.load(video_path))
        video_embs = self.videos[index]
        assert audio_input.shape == torch.Size([1, 3, 1, 128, 204])
        assert video_embs.shape == torch.Size([1024])
        return audio_input, video_embs
        # return torch.randn([1,3,1,128,204]), torch.randn(768)
    def __len__(self):
        return len(self.videos)

class IB_10M_FT(Dataset):
    def __init__(self, mode):
        
        # pd_dict = torch.load('')
        # # pd_names = pd_dict['names']
        # self.pd_audios = pd_dict['audio_embs']
        # self.pd_videos = pd_dict['video_embs']

        # as_dict = torch.load('')
        # # as_names = as_dict['names']
        # self.as_audios = as_dict['audio_embs']
        # self.as_videos = as_dict['video_embs']

        # self.audios = torch.cat([self.as_audios, self.pd_audios])
        # self.videos = torch.cat([self.as_videos, self.pd_videos])
        self.audios = np.load('/path/to/train_emb_gather/ib_audio.npy', mmap_mode='r')
        self.videos = np.load('/path/to/train_emb_gather/ib_image.npy', mmap_mode='r')

        print(len(self.audios), len(self.videos))
        if mode == 'AS':
            self.num = 1039268
        elif mode == 'PD+AS':
            self.num = len(self.videos)   
        

    def __getitem__(self, index):
        # inputs = {
        #     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path, torch.device('cpu'))
        # }
        audio_embs = torch.from_numpy(self.audios[index])
        # video_path = self.videos[index]
        video_embs = torch.from_numpy(self.videos[index])
        assert audio_embs.shape == torch.Size([1024])
        assert video_embs.shape == torch.Size([1024])
        return audio_embs, video_embs
        # return torch.randn([1,3,1,128,204]), torch.randn(768)
    def __len__(self):
        # return len(self.videos)
        return self.num

class Vid_10M_FT(Dataset):
    def __init__(self, mode):
        
        # pd_dict = torch.load('/path/to/pd_train_emb_gather.pt')
        # # pd_names = pd_dict['names']
        # self.pd_audios = pd_dict['audio_embs']
        # self.pd_videos = pd_dict['video_embs']

        # as_dict = torch.load('')
        # # as_names = as_dict['names']
        # self.as_audios = as_dict['audio_embs']
        # self.as_videos = as_dict['video_embs']

        # self.audios = torch.cat([self.as_audios, self.pd_audios])
        # self.videos = torch.cat([self.as_videos, self.pd_videos])
        self.audios = np.load('/path/to/train_emb_gather/ib_audio.npy', mmap_mode='r')
        self.videos = np.load('/path/to/train_emb_gather/vid_video.npy', mmap_mode='r')

        print(len(self.audios), len(self.videos))
        if mode == 'AS':
            self.num = 1039268
        elif mode == 'PD+AS':
            self.num = len(self.videos)    

    def __getitem__(self, index):
        # inputs = {
        #     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path, torch.device('cpu'))
        # }
        audio_embs = torch.from_numpy(self.audios[index])
        # video_path = self.videos[index]
        video_embs = torch.from_numpy(self.videos[index])
        assert audio_embs.shape == torch.Size([1024])
        assert video_embs.shape == torch.Size([768])
        return audio_embs, video_embs
        # return torch.randn([1,3,1,128,204]), torch.randn(768)
    def __len__(self):
        # return len(self.videos)
        return self.num
