AVSET-10M
======

AVSET-10M: An Open Large-Scale Audio-Visual Dataset with High Correspondence

## Overview
The AVSET-10M dataset is a comprehensive collection of audio-visual samples designed for research in multimedia content analysis, audio-visual recognition, and machine learning. It is divided into two distinct subsets: AVSET-700K and AVSET-10M (excluding AVSET-700K). This dataset provides a rich set of meta-information, enhancing its utility for diverse research applications.


## Getting Data

Please visit the [Hugging Face page](https://huggingface.co/datasets/avset10m/avset10m) for AVSET-10M to access the dataset.

## Detailed statistics

| Datasets             | Video | AV-C | #Class | #Clips | #Dur.(hrs) | #Avg Dur.(s) |
|----------------------|:-----:|:----:|:------:|-------:|-----------:|-------------:|
| DCASE2017 [29]       |   ✗   |  ✗   |   17   |    57K |         89 |          3.9 |
| FSD [11]             |   ✗   |  ✗   |  398   |    24K |        119 |         17.4 |
| AudioSet [13]        |   ✔   |  ✗   |  527   |   2.1M |       5.8K |           10 |
| AudioScope-V2 [40]   |   ✔   |  ✗   |   -    |   4.9M |       1.6K |            5 |
| ACAV100M [23]†       |   ✔   |  ✗   |   -    |   100M |     277.7K |           10 |
| HD-VILA-100M [45]    |   ✔   |  ✗   |   -    |   103M |     371.5K |         13.4 |
| Panda-70M [8]        |   ✔   |  ✗   |   -    |  70.8M |     166.8K |          8.5 |
| AVE [37]             |   ✔   |  ✔   |   28   |     4K |         11 |           10 |
| VGGSound [6]         |   ✔   |  ✔   |  309   |   200K |        550 |           10 |
| AVSET-700K (ours)    |   ✔   |  ✔   |  527   |   728K |       2.0K |           10 |
| AVSET-10M (ours)     |   ✔   |  ✔   |  527   |  10.9M |      30.4K |         10.3 |


## Dataset Composition
AVSET-10M is released as two subsets:

1. **AVSET-700K**
- **Description**: This subset consists of 727,530 audio-visual corresponding samples meticulously filtered from the AudioSet.
- **Features**:
  - Each video segment is accompanied by a manually labeled audio category.
  - Ensures accurate categorization and relevance of the audio-visual samples.

2. **AVSET-10M (w/o. AVSET-700K)**
- **Description**: This subset includes 9,877,475 audio-visual corresponding samples, filtered from the expansive Panda-70M dataset.
- **Features**:
  - Focus on semantically coherent video segments that concentrate on a single event.
  - Includes text descriptions sourced from the original Panda70M dataset.
  - Pseudo-labels for audio categories are provided, derived using [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn), along with their corresponding confidence scores.




## AVSET-10M Dataset Structure

Repository for the AVSET-10M dataset, containing two subsets. Below is the directory structure with links to significant components:
```
AVSET-10M/
│
├── [AVSET-700K/](#avset-700k)
│   ├── unbalanced_train_segments_part00.json
│   ├── ...
│   └── unbalanced_train_segments_part40.json
│
├── [AVSET-10M(excluding-700K)/]
│   ├── split_000.json
│   ├── ...
│   └── split_399.json
│
└──[ontology.json]
```

## Meta-Information
Each video clip in both subsets of the AVSET-10M dataset includes comprehensive meta-information:
- **Youtube ID**: YoutubeID to get the download url.
- **Start Time**: Specific start times for each video clip.
- **End Time**: Specific end times for each video clip.
- **AVC**: Measures the similarity between the audio and visual components of the clip.
- **Flag (Sound Separation)**: Indicates whether sound separation is required for the clip. "1" for the need for sound separation.
- **Label**: Relevant textual descriptions or labels associated with each clip.

**Additional Details for AVSET-10M (w/o. AVSET-700K)**
- **pseudo-label**: Audio categories with confidence scores to assist in dataset partitioning and analysis.

## Data Filtering Pipeline
    

The data filtering pipeline consists of four stages:
- Data Collection 
- [Audio-Visual Correspondence Filtering](Data_Filtering_Pipeline/AVC-Filtering/filter.py)
- [Voice-Over Filtering](/Data_Filtering_Pipeline/PANNs/README.md)
- [Sample Recycling with Sound Separation](/Data_Filtering_Pipeline/SoundSeparation/README.md)

## DownStreamTask
We benchmark two tasks on AVSET-10M:
- [Audio-Video Retrieval](/DownStreamTask/AV-Retrieval/README.md)
- [Vision-Queried Sound Separation](/DownStreamTask/VSS/README.md)

## License
See [license](LICENSE). The video samples are collected from a publicly available dataset. Users must follow the related [license](https://raw.githubusercontent.com/microsoft/XPretrain/main/hd-vila-100m/LICENSE) to use these video samples.


