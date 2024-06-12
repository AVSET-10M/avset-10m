from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from Models import IB2VITG_Head, ib_with_proj
from datasets import Audioset_raw, Audiocaps_raw, Audio_Visual_train, Audio_Visual_test, Vid_10M_FT
from metric import Retrieval_metrics, compute_retrieval

from loss import cross_entropy, get_CLIP_loss, get_L2_loss
from utils import *

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from lightning.pytorch.strategies import FSDPStrategy

from imagebind.models.transformer import MultiheadAttention, BlockWithMasking, Mlp
# from functools import partial
# # 1. Import a suiting wrapping policy from PyTorch
# from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


IB_AUDIOCAPS_DIR    = 'embedding/audiocaps_allinone_dict.pt'
IB_FLICKR_DIR       = 'imagebind_val/IB_Flickr_embs_IB.pt'
IB_AVE_DIR          = 'imagebind_val/IB_AVE_audio_embs_IB.pt'
IB_VGGSS_DIR        = 'embedding/IB_VGGSS_Audio.pt'
VITG_AUDIOCAPS_DIR  = 'imagebind_val/Audiocaps_text_test.pt'
VITG_FLICKR_DIR     = 'imagebind_val/Flickr_image.pt'
VITG_AVE_DIR        = 'imagebind_val/AVE_image.pt'
VITG_VGGSS_DIR      = 'imagebind_val/VGGSS_image.pt'


def validate_audiocaps(model: IB2VITG_Head):
    model.eval()
    device = model.get_device()
    audio_emb = torch.load(IB_AUDIOCAPS_DIR, map_location='cpu')['ib_audio_embs']
    text_emb  = torch.load(VITG_AUDIOCAPS_DIR, map_location='cpu')['text_embs']
    audio_emb = audio_emb.to(device)
    text_emb  = text_emb.to(device)
    audio_emb = model.Head2(model.Head1(audio_emb))

    # print(audio_emb.shape, text_emb.shape)
    a2t_sim = torch.einsum('nb,tb->nt', audio_emb, text_emb)
    t2a_sim = a2t_sim.T

    a2t_metrics = compute_retrieval(a2t_sim)
    t2a_metrics = compute_retrieval(t2a_sim)

    ave_mrr = (a2t_metrics['mrr'] + t2a_metrics['mrr']) / 2
    del text_emb, audio_emb
    return ave_mrr, a2t_metrics, t2a_metrics

def validate_flickr(model: IB2VITG_Head):
    model.eval()
    device = model.get_device()
    image_emb = torch.load(VITG_FLICKR_DIR, map_location='cpu')['image_embs']
    audio_emb = torch.load(IB_FLICKR_DIR, map_location='cpu')['audio_emb']
    audio_emb = audio_emb.to(device)
    image_emb = image_emb.to(device)

    audio_emb = model.Head1(audio_emb)
    audio_emb = model.Head2(audio_emb)

    # print(audio_emb.shape, image_emb.shape)
    a2i_sim = torch.einsum('nb,tb->nt', audio_emb, image_emb)
    i2a_sim = a2i_sim.T

    a2i_metrics = compute_retrieval(a2i_sim)
    i2a_metrics = compute_retrieval(i2a_sim)

    ave_mrr = (a2i_metrics['mrr'] + i2a_metrics['mrr']) / 2
    return ave_mrr, a2i_metrics, i2a_metrics

def validate_ave(model: IB2VITG_Head):
    model.eval()
    device = model.get_device()
    image_emb = torch.load(VITG_AVE_DIR, map_location='cpu')['image_embs']
    audio_emb = torch.load(IB_AVE_DIR, map_location='cpu')['audio_emb']
    audio_emb  = audio_emb.to(device)
    image_emb  = image_emb.to(device)

    audio_emb = model.Head1(audio_emb)
    audio_emb = model.Head2(audio_emb)

    # print(audio_emb.shape, image_emb.shape)
    a2i_sim = torch.einsum('nb,tb->nt', audio_emb, image_emb)
    i2a_sim = a2i_sim.T

    a2i_metrics = compute_retrieval(a2i_sim)
    i2a_metrics = compute_retrieval(i2a_sim)

    ave_mrr = (a2i_metrics['mrr'] + i2a_metrics['mrr']) / 2
    return ave_mrr, a2i_metrics, i2a_metrics

def validate_vggss(model: IB2VITG_Head):
    model.eval()
    device = model.get_device()
    image_emb = torch.load(VITG_VGGSS_DIR, map_location='cpu')['image_embs']
    audio_emb = torch.load(IB_VGGSS_DIR, map_location='cpu')
    audio_emb  = audio_emb.to(device)
    image_emb  = image_emb.to(device)

    audio_emb = model.Head1(audio_emb)
    audio_emb = model.Head2(audio_emb)

    # print(audio_emb.shape, image_emb.shape)
    a2i_sim = torch.einsum('nb,tb->nt', audio_emb, image_emb)
    i2a_sim = a2i_sim.T

    a2i_metrics = compute_retrieval(a2i_sim)
    i2a_metrics = compute_retrieval(i2a_sim)

    ave_mrr = (a2i_metrics['mrr'] + i2a_metrics['mrr']) / 2
    return ave_mrr, a2i_metrics, i2a_metrics

class Projector(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = ib_with_proj(imagebind_model.imagebind_huge(pretrained=True))

        for p in self.model.ib.parameters():
            p.requires_grad = False

        self.batch_num = 0

        self.loss_dict = {'align_loss':0,'train_loss':0}
        self.best_val_dict = {}
        self.best_ave_map = 0
        self.best_epoch = 0

        self.vggss_audio_embs = []
        self.vggss_video_embs = []

    def training_step(self, batch, batch_idx):
        audio_input, image_embs= batch
        # inputs = {
        #     ModalityType.AUDIO: audio_input.squeeze(1),
        # }
        temp = self.cfg.temperature
        # with torch.no_grad():
        #     audio_embs = self.model.ib(inputs)[ModalityType.AUDIO]
        audio_embs = audio_input
        audio_embs = self.model.ib_head(audio_embs)
        loss = get_CLIP_loss(audio_embs , image_embs , temp)

        self.log('loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self.vggss_audio_embs = []
        self.vggss_video_embs = []

    def validation_step(self, batch, batch_idx):
        audio_input, image_embs= batch
        inputs = {
            ModalityType.AUDIO: audio_input.squeeze(1),
        }
        audio_embs = self.model.ib(inputs)[ModalityType.AUDIO]
        # audio_embs = audio_input
        audio_embs = self.model.ib_head(audio_embs)
        self.vggss_audio_embs.append(audio_embs)
        self.vggss_video_embs.append(image_embs)
    
    def on_validation_epoch_end(self):
        if len(self.vggss_audio_embs) == 0:
            return
        else:
            self.vggss_audio_embs = F.normalize(torch.cat(self.vggss_audio_embs, dim=0), dim=-1)
            self.vggss_video_embs = F.normalize(torch.cat(self.vggss_video_embs, dim=0), dim=-1)

            vggss_a2i_sim = torch.einsum('nb,tb->nt', self.vggss_audio_embs, self.vggss_video_embs)
            vggss_i2a_sim = vggss_a2i_sim.T

            torch.save(vggss_a2i_sim.cpu(),'ave_a2i_sim_video_10M.pt')

            vggss_a2i_metrics = compute_retrieval(vggss_a2i_sim)
            vggss_i2a_metrics = compute_retrieval(vggss_i2a_sim)

            val_dict = {}
            val_dict['vggss_top1']    = (vggss_a2i_metrics['top_1'] + vggss_i2a_metrics['top_1'])/2.0
            val_dict['vggss_top5']    = (vggss_a2i_metrics['top_5'] + vggss_i2a_metrics['top_5'])/2.0
            if self.batch_num < 5:
                print('init zero')
                ave_map = 0
            else:
                ave_map = val_dict['vggss_top5']
            
            print(val_dict)
            self.logger.log_metrics(val_dict, step=self.batch_num)
            self.log('vggss_top5', val_dict['vggss_top5'], on_step=False, on_epoch=True, prog_bar = True,logger=False)
            if ave_map > self.best_ave_map:
                self.best_ave_map = ave_map
                self.best_epoch = self.current_epoch+1
                self.best_val_dict = val_dict
                torch.save(self.model.state_dict(), os.path.join(self.cfg.save_path, 'best.pt'))
        

    def start_validate(self):
        # validate
        with torch.no_grad():
            audiocaps_mrr, audiocaps_a2t_metrics, audiocaps_t2a_metrics = validate_audiocaps(self.model)
            vggss_mrr, vggss_a2i_metrics, vggss_i2a_metrics = validate_vggss(self.model)
            flickr_mrr, flickr_a2i_metrics, flickr_i2a_metrics = validate_flickr(self.model)
            ave_mrr, ave_a2i_metrics, ave_i2a_metrics = validate_ave(self.model)
            torch.cuda.empty_cache()
            val_dict = {}
            val_dict['vggss_mrr'] = vggss_mrr
            val_dict['flickr_mrr'] = flickr_mrr
            val_dict['audiocaps_mrr'] = audiocaps_mrr
            val_dict['ave_mrr'] = ave_mrr

            val_dict['flickr_top5']    = (flickr_a2i_metrics['top_5'] + flickr_i2a_metrics['top_5'])/2.0
            val_dict['vggss_top5']    = (vggss_a2i_metrics['top_5'] + vggss_i2a_metrics['top_5'])/2.0
            val_dict['audiocaps_top5']    = (audiocaps_a2t_metrics['top_5'] + audiocaps_t2a_metrics['top_5'])/2.0
            val_dict['ave_top5']     = (ave_a2i_metrics['top_5'] + ave_i2a_metrics['top_5'])/2.0

            self.log('flickr_mrr', flickr_mrr, on_step=True, prog_bar = True,logger=False)
            self.log('audiocaps_mrr', audiocaps_mrr, on_step=True, prog_bar = True,logger=False)
            self.log('vggss_mrr', vggss_mrr, on_step=True, prog_bar = True,logger=False)
            self.log('ave_mrr', ave_mrr, on_step=True, prog_bar = True,logger=False)

            self.logger.log_metrics(val_dict, step=self.batch_num)

            # ave_map = (audiocaps_mrr + ave_mrr)/2.0
            ave_map = (flickr_mrr + vggss_mrr + ave_mrr)
            # ave_map = audiocaps_mrr
            if ave_map > self.best_ave_map:
                self.best_ave_map = ave_map
                self.best_epoch = self.current_epoch+1
                self.best_val_dict = val_dict
                checkpoint = { 
                    'epoch': self.best_epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizers.state_dict(),
                    'lr_sched': self.lr_schedulers,
                    'cfg': self.cfg}
                torch.save(checkpoint, os.path.join(self.cfg.save_path, 'best.pt'))
                # torch.save(self.model.state_dict(), os.path.join(self.cfg.save_path, 'best.pt'))
                print("Saved Best Model!")
        torch.cuda.empty_cache()
        
    def configure_optimizers(self):
        # param_dicts = [
        #     {"params": [p for n, p in self.model.named_parameters() if "domain" in n and p.requires_grad],
        #      "lr": self.cfg.discr_lr_factor * self.cfg.lr},
        #     {"params": [p for n, p in self.model.named_parameters() if "domain" not in n and p.requires_grad],
        #      "lr": self.cfg.lr}
        # ]
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.betas, eps=self.cfg.eps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.epoch)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:

        
        if (self.batch_num + 1) % self.cfg.val_step == 0 or self.batch_num == 0:
            # self.start_validate()
            pass

        self.batch_num += 1
        return super().on_train_batch_end(outputs, batch, batch_idx)

def main(cfg):
    torch.set_float32_matmul_precision('medium')
    # Create the dataset and data loader
    pl.seed_everything(cfg.seed)

    print(cfg)
    model = Projector(cfg)

    val_dataset = Audio_Visual_test('AVE')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)

    lightning_logger = TensorBoardLogger(os.path.join(cfg.save_path,'stage1'))
    
    strategy = FSDPStrategy(
                    # activation_checkpointing_policy={
                    #     BlockWithMasking
                    # }, 
                    auto_wrap_policy={
                        BlockWithMasking
                    },
                    # cpu_offload=True
                    )
    
    
    if cfg.evaluate:
        model.model.load_state_dict(torch.load(os.path.join(cfg.save_path, 'best.pt'), map_location='cpu'))
        trainer = pl.Trainer(accelerator="gpu", devices=[1] )
        trainer.validate(model, val_loader)
    else:
        if cfg.continue_train:
            print('continue training')
            model.model.load_state_dict(torch.load(os.path.join(cfg.continue_path, 'best.pt'), map_location='cpu'))
        # train_data = load_data(mode='audioset_raw')
        # train_dataset = Audio_Visual_train(mode=['VGG'])
        train_dataset = Vid_10M_FT(mode = 'AS')
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)

        trainer = pl.Trainer(accelerator="gpu",
                            strategy="ddp_find_unused_parameters_true",
                        # strategy="fsdp",
                        #  strategy=FSDPStrategy(),
                        #  strategy=strategy,
                            devices=4, 
                            max_epochs=cfg.epoch, 
                        #  accumulate_grad_batches=4,
                        #  precision=16,
                        #  max_steps=6000,
                            logger=lightning_logger)
        trainer.fit(model, train_loader, val_loader)

        
def load_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=44)
    parser.add_argument("--batch_size", default=1024)
    parser.add_argument("--epoch", default=30)
    parser.add_argument("--val_step", default=250)
    parser.add_argument("--lr", default=1.0e-3)
    parser.add_argument("--weight_decay", default=1e-5)
    parser.add_argument("--betas", default=(0.9,0.99))
    parser.add_argument("--eps", default=1e-7)
    parser.add_argument("--temperature", default=0.05)

    parser.add_argument("--align_loss_factor", default=0.1)
    parser.add_argument("--discr_lr_factor", default=0.1)
    
    parser.add_argument("--variance", default=0.005)
    parser.add_argument("--modality_offset", default=False)

    parser.add_argument("--evaluate", default=False)
    parser.add_argument('--save_path', default=os.path.join('./output/ft_on_vgg_as', 'MLP_10M+as->as')) #Visual-Direct
    # 0529_MLP_vgg+as

    parser.add_argument("--continue_train", default=False)
    parser.add_argument("--continue_path", default=os.path.join('./output/ft_on_vgg_as', 'MLP_10M+as'))
    

    return parser

if __name__ == '__main__':
    parser = load_parser()
    args = parser.parse_args()
    main(args)

