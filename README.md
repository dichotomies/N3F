
# Neural Feature Fusion Fields (N3F): 3D Distillation of Self-Supervised 2D Image Representations

## [Project Page](https://www.robots.ox.ac.uk/~vadim/n3f/) | [Paper](https://www.robots.ox.ac.uk/~vgg/publications/2022/Tschernezki22/tschernezki22.pdf) | [Video](https://youtu.be/qAIpStmMHjY)

<p class="left">
  <img src="https://www.robots.ox.ac.uk/~vadim/n3f/resources/code_release/scene_edit/nd-orig.gif" width="200"/>
  <img src="https://www.robots.ox.ac.uk/~vadim/n3f/resources/code_release/scene_edit/nd-edit.gif" width="200" />
</p>

<p float="left">
  <img src="https://www.robots.ox.ac.uk/~vadim/n3f/resources/code_release/scene_edit/nerf-orig.gif" width="200" />
  <img src="https://www.robots.ox.ac.uk/~vadim/n3f/resources/code_release/scene_edit/nerf-fg.gif" width="200" />
  <img src="https://www.robots.ox.ac.uk/~vadim/n3f/resources/code_release/scene_edit/nerf-bg.gif" width="200" />
</p>

## About

This repository contains the official implementation of the paper *Neural Feature Fusion Fields: 3D Distillation of Self-Supervised 2D Image Representations* by [Vadim Tschernezki](https://github.com/dichotomies), [Iro Laina](https://campar.in.tum.de/Main/IroLaina), [Diane Larlus](https://dlarlus.github.io/) and [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/). Published at 3DV22 as Oral.

We provide the code for the experiments of the NeRF-N3F setting (three scenes: flower, horns, fern).

**Abstract**: We present Neural Feature Fusion Fields (N3F), a method that improves dense 2D image feature extractors when the latter are applied to the analysis of multiple images reconstructible as a 3D scene. Given an image feature extractor, for example pre-trained using self-supervision, N3F uses it as a teacher to learn a student network defined in 3D space. The 3D student network is similar to a neural radiance field that distills said features and can be trained with the usual differentiable rendering machinery. As a consequence, N3F is readily applicable to most neural rendering formulations, including vanilla NeRF and its extensions to complex dynamic scenes. We show that our method not only enables semantic understanding in the context of scene-specific neural fields without the use of manual labels, but also consistently improves over the self-supervised 2D baselines. This is demonstrated by considering various tasks, such as 2D object retrieval, 3D segmentation, and scene editing, in diverse sequences, including long egocentric videos in the EPIC-KITCHENS benchmark.

## Updates

### 20.10.22: Extracting DINO features from custom images/additional scenes

We provide a script for extracting DINO features from custom images and additional scenes for the NeRF setting. You can find the code in `feature_extractor`. To use the extractor, run following commands from the main directory:

```
# download the DINO model
sh feature_extractor/download_dino.sh

# extract features for other scenes; we use images down-scaled by a factor of 8
python -m feature_extractor.extract --dir_images data/nerf_llff_data/fern/images_8

# this will extract the features for the corresponding scene into `/data/dino/pca64`
ls data/dino/pca64/
# results in e.g. `fern.pt`
```

If you want to extract features for custom images, then simply structure your data in the same format as for the NeRF setting, and adjust `--dir_images` so that it points to your images.

## Getting started

### Setting up the Environment

We suggest to setup the environment through conda and pip.
1. Create and activate the specified conda anvironment.
2. Install the required packages from `requirements.txt`.

```
conda create -n n3f python=3.8
conda activate n3f
pip install -r requirements.txt
```

Since we demonstrate the experiments through a jupyter notebook, you'll have to install the jupyter kernel:

```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=n3f
```

If you are getting the following error: `CUDA error: no kernel image is available for execution on the device`, then try installing pytorch with a different CUDA kernel, e.g.: `pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`.

### Dataset and Pretrained Models

The dataset and pretrained models can be found on [google drive](https://drive.google.com/drive/folders/1sZ26AHd7N3xXWiP3ZTZ6uxd1T6kQDxu1?usp=sharing).

Download both files `logs.tar.gz` and `data.tar.gz` and extract them into the main directory. The checkpoints are located in the logs directory. The data directory contains the flower scene and the features extracted with DINO for this scene and the remaining scenes shown in the paper. This allows you to train your own models if you have downloaded the [NeRF checkpoints](https://drive.google.com/drive/folders/1jIr8dkvefrQmv737fFm2isiT6tqpbTbv) and [datasets for the remaining scenes](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

If you want to try out N3F with additional scenes from the NeRF setting, then download them from [google drive](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7) and place them into `data/nerf_llff_data`. After that, proceed with the extraction of the features as described in the update from 20.10.22 (see above).

## Reproducing results

We are providing a notebook that contains the code to reproduce the results for the flower scene. The other scenes will be added in the next days.

### Decomposed rendering

First, you can visualise the selected patch and calculate a histogram for the query feature vector vs. the retrieval vectors. This allows you to select a threshold for the scene decomposition in the next step.

<p float="left">
  <img src="https://www.robots.ox.ac.uk/~vadim/n3f/resources/code_release/patch.jpg" width="400" />
</p>

After that, you can render the source view and render the decomposed target view that shows the complete image, a version that includes only the queried object and another version that excludes the queried object.

<p float="left">
  <img src="https://www.robots.ox.ac.uk/~vadim/n3f/resources/code_release/decompose.jpg" width="600" />
</p>

### Comparison with DINO

Finally, we can also compare the PCA reduced features and feature distance maps of NeRF-N3F + DINO vs. vanilla DINO:

<p float="left">
  <img src="https://www.robots.ox.ac.uk/~vadim/n3f/resources/code_release/dino.jpg" width="400" />
</p>

## Citation

If you found our code or paper useful, then please cite our work as follows.

```bibtex
@inproceedings{tschernezki22neural,
  author     = {Vadim Tschernezki and Iro Laina and
                Diane Larlus and Andrea Vedaldi},
  booktitle  = {Proceedings of the International Conference
                on {3D} Vision (3DV)},
  title      = {Neural Feature Fusion Fields: {3D} Distillation
                of Self-Supervised {2D} Image Representations},
  year       = {2022}
}
```

## Concurrent work

We suggest to check out the concurrent work by [Kobayashi et al.](https://pfnet-research.github.io/distilled-feature-fields/) They propose to fuse features in the same manner and mainly differ in the example applications, including the use of multiple modalities, such as text, image patches and point-and-click seeds, to generate queries for segmentation and, in particular, scene editing.

## Acknowledgements

Our implementation is based on [this](https://github.com/yenchenlin/nerf-pytorch) (unofficial pytorch-NeRF) repository.
