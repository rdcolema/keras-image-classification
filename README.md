# Keras Image Classification

Classifies an image as containing either a dog or a cat (using Kaggle's <a href="https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data">public dataset</a>), but could easily be extended to other image classification problems.

To run this script, you must have keras, numpy, scipy, and h5py installed, and GPU acceleration with keras is <em>highly</em> recommended.

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

### UPDATE:
Added jupyter notebook that produced a slightly better score (.161 log loss on kaggle test set). Might run into memory errors because of the large image dimensions -- if so reducing the number of folds and saving the model weights rather than keeping the models in memory should do the trick. The notebook uses a slightly flatter directory structure, with the validation split happening after the images are loaded.
            

