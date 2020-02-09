# Faster Art-CNN: An Extremely Fast Style Transfer Network

This repository contains the official codebase for Faster Art-CNN.  This codebase is an extension of the work performed by L. Engstrom, located [here](https://github.com/lengstrom/fast-style-transfer).

The following documentation is replicated (with formatting corrections) from the repository of L. Engstrom.

## Documentation
### Training Style Transfer Networks
Use `style.py` to train a new style transfer network. Run `python style.py` to view all the possible parameters. Training takes 4-6 hours on a Maxwell Titan X. [More detailed documentation here](docs.md#stylepy). **Before you run this, you should run `setup.sh`**. Example usage:

    python style.py --style path/to/style/img.jpg \
      --checkpoint-dir checkpoint/path \
      --test path/to/test/img.jpg \
      --test-dir path/to/test/dir \
      --content-weight 1.5e1 \
      --checkpoint-iterations 1000 \
      --batch-size 20

### Evaluating Style Transfer Networks
Use `evaluate.py` to evaluate a style transfer network. Run `python evaluate.py` to view all the possible parameters. Evaluation takes 100 ms per frame (when batch size is 1) on a Maxwell Titan X. [More detailed documentation here](docs.md#evaluatepy). Takes several seconds per frame on a CPU. **Models for evaluation are [located here](https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ?usp=sharing)**. Example usage:

    python evaluate.py --checkpoint path/to/style/model.ckpt \
      --in-path dir/of/test/imgs/ \
      --out-path dir/for/results/

## style.py 

`style.py` trains networks that can transfer styles from artwork into images.

**Flags**

* `--checkpoint-dir`: Directory to save checkpoint in. Required.

* `--style`: Path to style image. Required.

* `--train-path`: Path to training images folder. Default: `data/train2014`.

* `--test`: Path to content image to test network on at at every checkpoint iteration. Default: no image.

* `--test-dir`: Path to directory to save test images in. Required if `--test` is passed a value.

* `--epochs`: Epochs to train for. Default: `2`.

* `--batch_size`: Batch size for training. Default: `4`.

* `--checkpoint-iterations`: Number of iterations to go for between checkpoints. Default: `2000`.

* `--vgg-path`: Path to VGG19 network (default). Can pass VGG16 if you want to try out other loss functions. Default: `data/imagenet-vgg-verydeep-19.mat`.

* `--content-weight`: Weight of content in loss function. Default: `7.5e0`.

* `--style-weight`: Weight of style in loss function. Default: `1e2`.

* `--tv-weight`: Weight of total variation term in loss function. Default: `2e2`.

* `--learning-rate`: Learning rate for optimizer. Default: `1e-3`.

* `--slow`: For debugging loss function. Direct optimization on pixels using Gatys' approach. Uses `test` image as content value, `test_dir` for saving fully optimized images.

## evaluate.py
`evaluate.py` evaluates trained networks given a checkpoint directory. If evaluating images from a directory, every image in the directory must have the same dimensions.

**Flags**

* `--checkpoint`: Directory or `ckpt` file to load checkpoint from. Required.

* `--in-path`: Path of image or directory of images to transform. Required.

* `--out-path`: Out path of transformed image or out directory to put transformed images from in directory (if `in_path` is a directory). Required.

* `--device`: Device used to transform image. Default: `/cpu:0`.

* `--batch-size`: Batch size used to evaluate images. In particular meant for directory transformations. Default: `4`.

* `--allow-different-dimensions`: Allow different image dimensions. Default: not enabled

### Requirements
You will need the following to run the above:

* TensorFlow 0.11.0

* Python 2.7.9, Pillow 3.4.2, scipy 0.18.1, numpy 1.11.2

* If you want to train (and don't want to wait for 4 months):

  * A decent GPU

  * All the required NVIDIA software to run TF on a GPU (cuda, etc)

* ffmpeg 3.1.3 if you want to stylize video

### Citation
```
  @misc{engstrom2016faststyletransfer,
    author = {Logan Engstrom},
    title = {Fast Style Transfer},
    year = {2016},
    howpublished = {\url{https://github.com/lengstrom/fast-style-transfer/}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project could not have happened without the advice (and GPU access) given by [Anish Athalye](http://www.anishathalye.com/). 
  - The project also borrowed some code from Anish's [Neural Style](https://github.com/anishathalye/neural-style/)
- Some readme/docs formatting was borrowed from Justin Johnson's [Fast Neural Style](https://github.com/jcjohnson/fast-neural-style)
- The image of the Stata Center at the very beginning of the README was taken by [Juan Paulo](https://juanpaulo.me/)

### License
Copyright (c) 2016 Logan Engstrom. Contact me for commercial use (or rather any use that is not academic research) (email: engstrom at my university's domain dot edu). Free for research use, as long as proper attribution is given and this copyright notice is retained.

End of replicated documentation.

## Faster Art-CNN Results
Below are sample images from Faster Art-CNN, along with the iterative technique of Gatys and Fast Residual technique of Johnson.  From left to right, the images are trained style, content, output of Gatys' iterative technique, output of Johnson's Fast Residual technique, and the output of Faster Art-CNN.

![Results 1](/images/keanuReevesHoriz.jpg)
![Results 2](/images/mariaGuidaHoriz.jpg)
![Results 3](/images/jackieChanHoriz.jpg)
![Results 4](/images/paulBettanyHoriz.jpg)