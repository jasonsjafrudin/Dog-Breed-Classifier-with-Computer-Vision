# Dog-Breed-Classifier-with-Computer-Vision

This is a repo for a Dog Breed Classifier project. I had build a custom Convolutional Neural Network using PyTorch Library that takes image input data and classifies the breed of the dog. I have also built a streamlit web application to deploy the ML model.


## My custom CNN Structure:
<img width="508" alt="image" src="https://github.com/jasonsjafrudin/Dog-Breed-Classifier-with-Computer-Vision/assets/61297201/e8516291-6736-4d82-bdea-9f58e147037d">


## Training Process:
- Trained using Adam optimizer (learning rate = 0.001)
-  Forward propagation, loss calculation using CrossEntropyLoss
-  Backpropagation to update weights
-  Dropout is used during training as a form of regularization to prevent overfitting

## Validation & Testing:
- Epochs:  15
- Learning rate: adjusted with StepLR scheduler (step size = 7 epochs / gamma = 0.1)
- Performance monitored on a validation set to check for overfitting
- Test Accuracy: 85% / loss: 0.486031 on test dataset


You can find the training data, validation data, testing data, ML Model Code, and Streamlit Code all in this repo.



## Fun Model Testing Experiment:
I also wanted to see how my model would react to images that are not straightforward (in the sense that it is not just a simple and clear dog picture).

<img width="506" alt="image" src="https://github.com/jasonsjafrudin/Dog-Breed-Classifier-with-Computer-Vision/assets/61297201/1e92d468-3d99-4c26-8518-cda704632780">

- 90 Degree flips: 100% accuracy
- 180 Degree Flips: 86.67% accuracy
- 251 Degree Flips: 93.33% accuracy
- Motion Blur: 100% accuracy
- Brightness Contrast: 100% accuracy
- Impulse Noise: 73.33% accuracy
- Block Cutout: 100% accuracy



<img width="277" alt="image" src="https://github.com/jasonsjafrudin/Dog-Breed-Classifier-with-Computer-Vision/assets/61297201/9747c18e-695d-4b8d-815e-26d24b30d0a1">
<img width="226" alt="image" src="https://github.com/jasonsjafrudin/Dog-Breed-Classifier-with-Computer-Vision/assets/61297201/68cd3291-66c8-44d5-89ff-d7d9df6c2599">
<img width="216" alt="image" src="https://github.com/jasonsjafrudin/Dog-Breed-Classifier-with-Computer-Vision/assets/61297201/9d2c6b66-458b-4bae-8dbd-d07babc9c3ec">


As you can see, even with these odd/not straightforward images, my model is still predicting very well!


