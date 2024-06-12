# Visually Guided Sound Separation

Our approach to visually guided sound separation builds on the CLIPSEP framework, renowned for its proficiency in image-queried sound separation. In our adaptation, we substitute the original clip embeddings with the more sophisticated features (V) from the internvid model, thereby enhancing the model’s ability to understand audio-visual temporal consistency. For additional information about CLIPSEP, and detailed implementation details, please visit the CLIPSEP [GitHub repository](https://github.com/sony/CLIPSep.git).


## Train
Train with video feature

```bash
python train.py -o exp/vggsound/clipsep_nit -t data/vggsound/train.csv -v data/vggsound/val.csv --image_model internvid
```

## Evaluate 


Evaluate on VGGSound + VGGSound  

```bash
OMP_NUM_THREADS=1 python evaluate.py -o exp/AudioSet/clipsep_nit/ -l exp/AudioSet/clipsep_nit/eval_woPIT_VGGS_VGGSN.txt -t data/vggsound-v1/test.csv -t2 data/vggsound-v1/test.csv --no-pit --prompt_ens
```