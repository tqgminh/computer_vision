# Cats and Dogs Segmentation

In this project, we practice some methods for image segmentation with two class including cats and dogs. Our methods is based on two approachs which are Classical techniques (K-Means, Mean Shift and Region Growing) and Deep Learning (U-Net and Mask R-CNN). All methods are build in a web app based on Flask framework.

Two deep learning models are trained on the Oxford-IIIT Pet dataset. More info and download at: https://www.robots.ox.ac.uk/~vgg/data/pets/

# Installation

You need to download two pretrained weight files and move them to `model` folder.

Mask R-CNN Weight file: https://drive.google.com/file/d/1-o2xMhRs9axF_dkoptyW4mrNt7y5H2j5/view?usp=sharing

U-Net with DenseNet121 Encoder Weight file: https://drive.google.com/file/d/1-2rBS9tQ1_8HxLKoZDpoUHEdfYM2Ebj6/view?usp=sharing

Clone our repository:

```
git clone https://github.com/tqgminh/computer_vision
cd computer_vision
```

Install `Detectron` framework:

```
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Install all required packages:

```
pip install -r requirements.txt
```

Run `app.py`:

```
python app.py
```

Then type this link in your browser:

```
http://127.0.0.1:5000/
```

# Usage

There are 5 options to segment an image containing dogs or cats.

![alt text](https://github.com/tqgminh/computer_vision/blob/main/images/options.png?raw=true)

You can consider choosing `K-Means`, `Mean Shift` or `Region Growing` to clustering similar pixels in color.

![alt text](https://github.com/tqgminh/computer_vision/blob/main/images/kmeans.png?raw=true)

If you need to determine the pixels belonging to dogs or cats, let's choose `U-Net segmentation`.

![alt text](https://github.com/tqgminh/computer_vision/blob/main/images/unet.png?raw=true)

Furthermore, to distinguish cats and dogs instance, let's choose `Mask R-CNN segmentation`.

![alt text](https://github.com/tqgminh/computer_vision/blob/main/images/mask_rcnn.png?raw=true)
