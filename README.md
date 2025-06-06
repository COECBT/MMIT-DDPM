# MMIT-DDPM – Multilateral Medical Image Translation with Class and Structure Supervised Diffusion-Based Model

We provide the official PyTorch implementation of the paper [MMIT-DDPM – Multilateral Medical Image Translation with Class and Structure Supervised Diffusion-Based Model]

The implementation of Denoising Diffusion Probabilistic Models presented in the paper is based on [openai/improved-diffusion](https://github.com/openai/improved-diffusion) and [JuliaWolleb/Diffusion-based-Segmentation](https://github.com/JuliaWolleb/Diffusion-based-Segmentation).

The paper is published in Computers in Biology and Medicine Jounal. Here is the link to the paper: [ MMIT-DDPM – Multilateral medical image translation with class and 
structure supervised diffusion-based model](https://doi.org/10.1016/j.compbiomed.2024.109501)


## Data

MMIT-DDPM was evaluated on [BRATS2021 dataset](http://www.braintumorsegmentation.org/).
MMIT-DDPM requires the data-directory in particular format which can be depicted below:

```
data
└───training
│       │   brats_train_XXXXX_ft2_XX_w.nii.gz
│       │   brats_train_XXXXX_ft1ce_XX_w.nii.gz
│       │   brats_train_XXXXX_t1cet2_XX_w.nii.gz
│       │   brats_train_XXXXX_t1ceft_XX_w.nii.gz
│       │   brats_train_XXXXX_t2f_XX_w.nii.gz
│       │   brats_train_XXXXX_t2t1ce_XX_w.nii.gz
│       │  ...
└───testing
│       │   brats_test_XXXXX_ft2_XX_w.nii.gz
│       │   brats_test_XXXXX_ft1ce_XX_w.nii.gz
│       │   brats_test_XXXXX_t1cet2_XX_w.nii.gz
│       │   brats_test_XXXXX_t1ceft_XX_w.nii.gz
│       │   brats_test_XXXXX_t2f_XX_w.nii.gz
│       │   brats_test_XXXXX_t2t1ce_XX_w.nii.gz
│       │  ...

```

MMIT-DDPM was trained on a six different combinations of three modalities, T1ce, T2, and Flair. For instance: ft1ce contains a slice of flair and t1ce. In the above data structure, XXXXX and XX denotes sequence and slice respectively.
The images will automatically be scaled and center-cropped by the data-loading pipeline. Simply pass --data_dir path/to/images to the training script, and it will take care of the rest.

Our full training and testing dataset can be downloaded from [MMIT-DDPM Data](https://drive.google.com/file/d/1dLTVF7-oBhpBqriIkJ9GPYizrzcwKhJR/view?usp=sharing)

## Usage

Set the flags:
```
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
```
Train the MMIT-DDPM model using

```
python scripts/train_translation.py --data_dir ./data/training $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
```
The model will be saved in the *results* folder.
Sampling an ensemble of 5 translated output with the MMIT-DDPM approach using:

```
python scripts/sampling_translation.py  --data_dir ./data/testing  --model_path ./results/savedmodel.pt --num_ensemble=5 $MODEL_FLAGS $DIFFUSION_FLAGS
```
The generated outputs will be stored in the *results* folder. A visualization of the sampling process is done using [Visdom](https://github.com/fossasia/visdom).

