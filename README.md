# Road-Symbols-Classification
## Classification of 43 road symbols using Deep Learning.
Some of the Road symbols in the dataset are,
  * Speed limit
  * No passing
  * Right-of-way at the next intersection
  * Stop
  * No vechiles
  * Slippery road
  * Bicycles crossing
  * Turn right ahead
  * Turn left ahead
  
Please look at the data/german-traffic-signs/signnames.csv for further details about the data.

# Dataset
* Data taken from https://bitbucket.org/jadslim/german-traffic-signs

# Techniques used
* Data augmentation: Applying image transformation techniques on the training images to improve accuracy.
* Dropouts: Adding dropout layer to avoid overfitting.
* LENET styled Convolutional Neural Network

# Accuracy Obtained
* Training accuracy - 0.9624 
* Validation accuracy - 0.9946 
* Test accuracy - 0.9790
