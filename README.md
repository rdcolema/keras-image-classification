# Keras Image Classification

Classifies an image as containing either a dog or a cat (using Kaggle's <a href="https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data">public dataset</a>).

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
    
Most of the code / strategy here was based on <a href="https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html">this</a> Keras tutorial.
            

