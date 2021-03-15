# The-Bachelor-s-Thesis

---
Official implementation of my Bachelor's Thesis "Multi-scale Learning a GAN from a Single representative".
My Bachelor's Thesis based on:
- [SinGAN: Learning a Generative Model from a Single Natural Image by Tamar Rott Shaham, Tali Dekel, and Tomer Michaeli.](https://arxiv.org/abs/1905.01164)
- ["Improved Techniques for Training Single-Image GANs" by Tobias Hinz, Matthew Fisher, Oliver Wang, and Stefan Wermter.](https://arxiv.org/abs/2003.11512)
- ["Structural-analogy from a Single Image Pair" by Sagie Benaim, Ron Mokady, Amit Bermano, Daniel Cohen-Or, and Lior Wolf.](https://arxiv.org/abs/2004.02222)

# Installation
- python 3.5
- pytorch 1.1.0

```
pip install -r requirements.txt --quiet
```

# Unconditional Generation

# Animation
## Video Translation:
```
cd video
```
Video to images:
```
python video2imgs.py --input ../video_data/04.mp4 --out ./04_imgs/
```
In some cases, quantization of the images helps to achieve better results (not true for most cases). For quantization:
```
python quant_vid.py --root_dir ./04_imgs/ --out ./04_q0/ --quant_level 2
```
Where quant_level control the level of quantization (higher increase in the number of colors). 
Training:
```
cd -
python main_train.py --train_mode video --video_dir ./video/volcano_q0/ --num_images 250 --input_b Images/Generation/10.png --gpu 0 --out ./video/vid_out/ --vid_ext .png --min_size 25 --niter 2000 --beta 10
```
Evaluation:
```
!python3 imgs2video.py --input [input_dir_images] --out [ouput_dir_GIF] --fps 25.0 --frames 249
```
## Image Translation:

# Acknowledgements
Our implementation is based on this implementation of the ConSinGAN paper and .

