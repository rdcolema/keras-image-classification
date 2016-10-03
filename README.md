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

Pre-trained vg166 model weights can be downloaded <a href="https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3">here</a>.
            

