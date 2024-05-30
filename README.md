# Dog-Breed-Classifier-with-Computer-Vision

This is a repo for a Dog Breed Classifier project. I had build a custom Convolutional Neural Network using PyTorch Library that takes image input data and classifies the breed of the dog. I have also built a streamlit web application to deploy the ML model.


My custom CNN Structure:
<img width="508" alt="image" src="https://github.com/jasonsjafrudin/Dog-Breed-Classifier-with-Computer-Vision/assets/61297201/e8516291-6736-4d82-bdea-9f58e147037d">


Training Process:
- Trained using Adam optimizer (learning rate = 0.001)
-  Forward propagation, loss calculation using CrossEntropyLoss
-  Backpropagation to update weights
-  Dropout is used during training as a form of regularization to prevent overfitting

Validation & Testing:
- Epochs:  15
- Learning rate: adjusted with StepLR scheduler (step size = 7 epochs / gamma = 0.1)
- Performance monitored on a validation set to check for overfitting
- Test Accuracy: 85% / loss: 0.486031 on test dataset


You can find the training data, validation data, testing data, ML Model Code, and Streamlit Code all in this repo.
