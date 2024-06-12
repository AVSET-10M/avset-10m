# Audio-Video Retrieval

Our audio-video retrieval method is based on the FreeBind framework. In our work, we utilize both image and video features for querying, which enhances the model's ability to comprehend audio-visual temporal consistency. For more information about FreeBind and detailed implementation details, please visit the FreeBind [GitHub repository](https://github.com/zehanwang01/FreeBind).

## Feature Extract

### video feature
    python InternVid_Feature.py

### image/audio feature
    python InternVid_Feature.py

## Train
Train with video feature

```bash
python ft_imagebind_proj.py
```


