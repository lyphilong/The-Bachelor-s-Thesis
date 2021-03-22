# The-Bachelor-s-Thesis
- Folder `Source`: contain code for this project
- Folder `Report`: contain file present(pdf), file report(latex,pdf)
---
Official implementation of my Bachelor's Thesis **"Multi-scale Learning a GAN from a Single representative"**.

My Bachelor's Thesis based on:
- [SinGAN: Learning a Generative Model from a Single Natural Image by Tamar Rott Shaham, Tali Dekel, and Tomer Michaeli.](https://arxiv.org/abs/1905.01164)
- ["Improved Techniques for Training Single-Image GANs" by Tobias Hinz, Matthew Fisher, Oliver Wang, and Stefan Wermter.](https://arxiv.org/abs/2003.11512)
- ["Structural-analogy from a Single Image Pair" by Sagie Benaim, Ron Mokady, Amit Bermano, Daniel Cohen-Or, and Lior Wolf.](https://arxiv.org/abs/2004.02222)

Abstract: The task of unsupervised image-to-image translation has seen substantial advancements in recent years through the use of deep neural networks. My thesis is based on [ConSinGAN model](https://arxiv.org/abs/2004.02222) and [model's Benaim et al](https://arxiv.org/abs/2004.02222). In this thesis, we explore the capabilities of neural networks to understand image structure by given a single video A and a single image B. The model will learn the mapping  crose to fine with different resolution of the image for each scale. The model will create the video which captures the feature of single image B and have an animation of video A by a combination of the [ConSinGAN model](https://arxiv.org/abs/2004.02222) and [model's Benaim et al](https://arxiv.org/abs/2004.02222). As a result, the image will be transformed from the most general structure to the smallest details in the image. However, the combination of [ConSinGAN model](https://arxiv.org/abs/2004.02222) and [model's Benaim et al](https://arxiv.org/abs/2004.02222) has produced negative results and that becomes the premise that should be avoided for these stellar studies.

# Installation
- python 3.5
- pytorch 1.1.0

```
pip install -r requirements.txt --quiet
```

# Result

However, the outcome was not good but this is the premise for future advances in the model.

# Video Translation
```
cd video
```
Video to images:
```
python video2imgs.py --input [input_video] --out [folder contain frames of video]
```
In some cases, quantization of the images helps to achieve better results (not true for most cases). For quantization:
```
python quant_vid.py --root_dir [folder contain frames of video] --out [folder contain frames of video with quantiztion] --quant_level 2
```
Where quant_level control the level of quantization (higher increase in the number of colors). 
Training:
```
cd -
python main_train.py --train_mode video --video_dir [folder contain frames of video with quantiztion]  --num_images 250 --input_b [Image have a style you want to learn] --gpu 0 --out [Folder contain frames of video out] --vid_ext .png --min_size 25 --niter 2000 --beta 10
```
Evaluation:
```
!python3 imgs2video.py --input [input_dir_images] --out [ouput_dir_GIF] --fps 25.0 --frames 249
```
# Acknowledgements
Our implementation is based on this implementation of the [ConSinGAN](https://github.com/tohinz/ConSinGAN/blob/master/README.md) paper and [Structural-analogy](https://github.com/rmokady/structural-analogy).
