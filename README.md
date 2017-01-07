# Keras Image Classification

Classifies an image as containing either a dog or a cat (using Kaggle's <a href="https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data">public dataset</a>), but could easily be extended to other image classification problems.

To run these scripts/notebooks, you must have keras, numpy, scipy, and h5py installed, and enabling GPU acceleration is highly recommended if that's an option.

## img_clf.py
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
    
After playing around with hyperparameters a bit, this reaches around 96-98% accuracy on the validation data, and when tested in Kaggle's competition maxed out with a log loss score around 0.18.

Most of the code / strategy here was based on <a href="https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html">this</a> Keras tutorial.

Pre-trained VGG16 model weights can be downloaded <a href="https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3">here</a>.

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
            

