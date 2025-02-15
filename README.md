# Face Recognition analysis for multimodal approach

The repository contains the code necessary for the analysis and the analysis of the face recognition models performed for the multimodal verification task. 

## Dataset

The analysis was performed on the **VoxCeleb2** dataset. Typically, models for facial recognition are trained on collections of photographs, where the images are usually of good quality. The collections mainly contain clear and frontal faces, but few faces captured from different angles, blurred, with a lot of occlusion or in heavy lighting.

**VoxCeleb2** is a collection of recordings, so the faces are moving. Performing good quality face verification on such a collection is a challenge because you have to deal with in-the-wild verification, where the captured faces can be of very poor quality - blurred, hidden or poorly lit.

## Models

Three pretrained models for face recognition was tested:
- AdaFace
- ArcFace
- Facenet

## Repository description

Most of the useful scripts are placed in the *src* folder.

### Face recognition performance

#### Preparing dataset

*prepare_dataset.py* is used to create a collection of images from video files. It extracts ramdomly the indicated number of frames from the video. The frames must contain the frontal face, which is checked using **FSANet**. The faces are also cropped using **MTCNN**.

```python
python prepare_dataset.py
```
Arguments:
```
-p --path           Path to the folder with videos to process
-m --min_videos     Minimum number of videos to keep a folder
-s --sample         Number of images to sample from a single video
--suffix            The suffix for folder name (as it stayes the same + suffix)
```

#### Config file

*config.yml* contains information on the models and data used:
- models - a list of models to use 
- data - images folders to use
- data_path - path to a folder with data folders

#### Generating embeddings

*get_feats.py* calculates the emebddings for each model and data folder indicated in the config file. The results are saved to */data/embeddings/* folder. 

#### Calculating samples similarity

*base_face_reco.py* is used to calculate the similatiry score between all the embeddings. The results are stored in */data/scores/* folder.

#### Anaysis

The analysis was performed in notebooks placed in */notebooks/* folder:
- all - performance for ramdomly chosen frames
- front - performance for ramdomly chosen frontal frames
- aligned - performance for ramdomly chosen frontal, aligned frames

The metrics are stored in */results/* folder.

### Extracting faces for multimodal approach

For the multi-modal approach, a different way of selecting frames had to be used so that it was compatible with voice extraction. It was decided to do extraction based on a sliding window. A predefined time window that is shifted along the length of the video by a predefined skip. One random frontal frame is extracted from each window. The extracted frames are combined into a single tensor. 

*prep_vid_v2.py* is used for this task. The *get_images_tensor()* function extractsthe relevant frames.