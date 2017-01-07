# Keras Image Classification

Classifies an image as containing either a dog or a cat (using Kaggle's <a href="https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data">public dataset</a>), but could easily be extended to other image classification problems.

To run these scripts/notebooks, you must have keras, numpy, scipy, and h5py installed, and enabling GPU acceleration is highly recommended if that's an option.

## img_clf.py
After playing around with hyperparameters a bit, this reaches around 96-98% accuracy on the validation data, and when tested on Kaggle's hidden test data achieved a log loss score around 0.18.

Most of the code / strategy here was based on <a href="https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html">this</a> Keras tutorial.

Pre-trained VGG16 model weights can be downloaded <a href="https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3">here</a>.

The data directory structure I used was:

* project
  * data
    * train
      * dogs
      * cats
    * validation
      * dogs
      * cats
    * test
      * test

## cats_n_dogs.ipynb:
This produced a slightly better score (.161 log loss on kaggle test set). The better score most likely comes from having larger images and ensembling a few models, despite the fact there's no image augmentation in the notebook. 

Might run into memory errors because of the large image dimensions -- if so reducing the number of folds and saving the model weights rather than keeping the models in memory should do the trick. The notebook uses a slightly flatter directory structure, with the validation split happening after the images are loaded:

* project
  * data
    * train
      * dogs
      * cats
    * test
      * test
            
## cats_n_dogs_BN.ipynb:
This produced the best score (0.069 loss without any ensembling). The notebook incorporates some of the techniques from Jeremy Howard's <a href="http://course.fast.ai/">deep learning class</a> , with the inclusion of batch normalization being the biggest factor. I also added extra layers of augmentation to the prediction script, which greatly improved performance.

Pre-trained model weights for VGG16 w/ batch normalization can be downloaded <a href="http://www.platform.ai/models/">here</a>.

The VGG16BN class is defined in <em>vgg_bn.py</em>, and the data directory structure used was: 

* project
  * data
    * train
      * dogs
      * cats
    * validation
      * dogs
      * cats
    * test
      * test
