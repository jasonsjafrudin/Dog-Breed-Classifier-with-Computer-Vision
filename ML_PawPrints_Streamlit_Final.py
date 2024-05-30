# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 01:44:30 2024

@author: 24221
"""


import torch
import streamlit as st
from torchvision import transforms
import json

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import pickle
import os

import tempfile
import zipfile
import shutil

num_classes = 3
seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)

def prepare_data(uploaded_file, transform, root_name):
    if uploaded_file is not None:
        # Create a temporary directory to extract the zip file
        temp_dir = tempfile.mkdtemp()
        temp_dir2 = tempfile.mkdtemp()
        # Open the uploaded file as a zip file
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)  # Extract all the contents into the temporary directory
        for child_dir in os.listdir(temp_dir + "\\" + root_name):
            shutil.copytree(temp_dir + "\\" + root_name + "\\" + child_dir, temp_dir2 + "\\" + child_dir)
        shutil.rmtree(temp_dir)
        # Use the extracted directory with ImageFolder
        data = datasets.ImageFolder(root=temp_dir2, transform=transform)
        print(f"data classes: {data.classes}")
        loader = DataLoader(data, batch_size=20, shuffle=True, drop_last=True)
        return loader, temp_dir2
    return None, None


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # Output: 16 channels
        self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization after Convolutional layers
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        #self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        #self.bn4 = nn.BatchNorm2d(128)
        self.adap_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 3)  # Assuming 3 classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.adap_pool(x)
        x = x.view(-1, 7*7*64)
        x = self.dropout(x)
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model_path = 'model.pt'
def train(n_epochs, data_loaders, model, optimizer, criterion, scheduler, use_cuda, model_path, progress_placeholder, status_text_placeholder, last_validation_loss=None):
    # Initialize tracker for minimum validation loss
    valid_loss_min = np.Inf if last_validation_loss is None else last_validation_loss
    epoch_loss_info = []

    for epoch in range(1, n_epochs + 1):
        # Monitoring training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # Training the model
        model.train()
        for batch_idx, (data, target) in enumerate(data_loaders['train']):
            # Moving to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        # Validating the model
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loaders['valid']):
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        # Optional: Adjust LR
        scheduler.step(valid_loss)

        # Print stats
        epoch_loss_info.append((epoch, train_loss.item(), valid_loss.item()))  # convert losses to scalar
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        progress = epoch / n_epochs
        progress_placeholder.progress(progress)
        status_text_placeholder.text(f'Epoch {epoch}/{n_epochs} - Training Loss: {train_loss:.6f}, Validation Loss: {valid_loss:.6f}')

        
        

        # Save model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), model_path)
            valid_loss_min = valid_loss
    progress_placeholder.progress(100)
    status_text_placeholder.text(f'Training completed: {n_epochs} epochs finished.')

    # Save model parameters to JSON file in a temporary directory
    model_params = get_model_params(model)
    temp_dir = tempfile.mkdtemp()
    json_path = os.path.join(temp_dir, 'model_params.json')
    with open(json_path, 'w') as f:
        json.dump(model_params, f, indent=4)
    

    return model, train_loss.item(), valid_loss.item(), epoch_loss_info, temp_dir, valid_loss_min

def get_model_params(model):
    """Extracts model parameters as a dictionary"""
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.data.cpu().numpy().tolist()  # Convert to list for JSON
    return params

# Function to save model parameters as a pickle file
def save_model_params(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model.state_dict(), f)


def data_test(data_loaders, model, criterion, use_cuda):
    # model.eval()  # Set the model to evaluation mode
    test_loss = 0.
    correct = 0.
    total = 0.
    predictions = []
    targets = []

    print("\n")

    for batch_idx, (data, target) in enumerate(data_loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += pred.eq(target.data.view_as(pred)).sum().item()
        total += data.size(0)

        # Append predictions and targets for confusion matrix calculation
        predictions.extend(pred.cpu().numpy())
        targets.extend(target.cpu().numpy())

        # # Print predicted and actual labels for each image
        # for i in range(len(data)):
        #     print(f"Predicted: {pred[i]}, Actual: {target[i]}")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(targets, predictions)

    # Calculate precision and recall
    precision = precision_score(targets, predictions, average=None, zero_division=0)
    recall = recall_score(targets, predictions, average=None, zero_division=0)

    return test_loss, correct / total, conf_matrix, precision, recall


model_scratch = Net()

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the appropriate device
model_scratch.to(device)

# Then set up your loss function, optimizer, etc.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_scratch.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
uploaded_train = st.file_uploader("Choose a training dataset ZIP", type=['zip'])
uploaded_valid = st.file_uploader("Choose a validation dataset ZIP", type=['zip'])
uploaded_test = st.file_uploader("Choose a testing dataset ZIP", type=['zip'])

# Define your transformations here
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}
# Define placeholders for results at the beginning of your app

progress_placeholder = st.progress(0)
status_text_placeholder = st.empty()

from PIL import Image
import io
epoch_loss_info = []
train_losses = []
valid_losses = []

def preprocess_image(uploaded_file):
    # Read the file into a PIL image
    image = Image.open(uploaded_file).convert('RGB')  # Convert to RGB in case the image is RGBA or grayscale

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply the transformation
    image = transform(image).unsqueeze(0)
    return image

def predict(uploaded_file, model, class_labels):
    # Preprocess the uploaded image
    image = preprocess_image(uploaded_file)
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Perform prediction
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)[0]
        predicted_class_index = probabilities.argmax().item()
        predicted_label = class_labels[predicted_class_index]
        probabilities = probabilities.numpy()
        return predicted_label, probabilities
# Initialize tmodel in session state (default to None).
if 'tmodel' not in st.session_state:
  st.session_state['tmodel'] = None # This is required to pass the tmodel variable from the train button to the test button




if uploaded_train and uploaded_valid and uploaded_test:
    train_loader, train_temp_dir = prepare_data(uploaded_train, data_transforms['train'], "train")
    valid_loader, valid_temp_dir = prepare_data(uploaded_valid, data_transforms['val'], "valid")
    test_loader, test_temp_dir = prepare_data(uploaded_test, data_transforms['test'], "test")

    if train_loader and valid_loader and test_loader:
        data_loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
        model = model_scratch
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Training the model
        if st.button('Train Model'):
            with st.spinner('Training model...'):
                tmodel, train_loss, valid_loss, epoch_loss_info, temp_dir, valid_loss_min = train(
                    15, data_loaders, model, optimizer, criterion, scheduler, torch.cuda.is_available(),
                    'model.pt', progress_placeholder, status_text_placeholder
                )
                st.session_state['tmodel'] = tmodel  # Save trained model to session state
                
                st.success('Model trained successfully!')

            # Plotting the loss after training
            for epoch, train_loss, valid_loss in epoch_loss_info:
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                st.write(f'Epoch {epoch} - Training Loss: {train_loss:.6f}, Validation Loss: {valid_loss:.6f}')

            # Create a DataFrame to store the epoch numbers alongside the loss values
            loss_data = pd.DataFrame({
                'Epoch': list(range(1, len(train_losses) + 1)),
                'Training Loss': train_losses,
                'Validation Loss': valid_losses
            })

            # Create the line chart
            st.line_chart(loss_data.set_index('Epoch'))


            # Create a DataFrame to store the epoch numbers alongside the loss values
            loss_data1 = pd.DataFrame({
                'Epoch': list(range(1, len(train_losses) + 1)),
                'Training Loss': train_losses,
                'Validation Loss': valid_losses,
                'Lowest Loss': valid_loss_min
            })

            # Filter the DataFrame to keep only the row where Validation Loss equals the Lowest Loss
            lowest_loss_data = loss_data1[loss_data1['Validation Loss'] == loss_data1['Lowest Loss']]

            # Get the final training loss from the 1-value row in the 'Training Loss' column
            final_training_loss = lowest_loss_data['Training Loss'].iloc[0]

            # Get the final validation loss from the 1-value row in the 'Lowest Loss' column
            final_validation_loss = lowest_loss_data['Lowest Loss'].iloc[0]

            # Display the final training and validation losses
            st.write(f'Final Training Loss: {final_training_loss}')
            st.write(f'Final Validation Loss: {final_validation_loss}')


            # Get class labels
        class_labels = data_loaders['test'].dataset.classes
        # Testing the model
        if st.button('Test Model'):
            with st.spinner('Testing model...'):
                if st.session_state['tmodel'] is not None:  # Check if model is trained
                    # Use the retrieved model from session state
                    test_loss, accuracy, conf_matrix, precision, recall = data_test(
                        data_loaders, st.session_state['tmodel'], criterion, torch.cuda.is_available())
            st.success(f'Model tested successfully! Test Loss: {test_loss:.6f}, Accuracy: {accuracy:.2%}')
            # Display confusion matrix with labels
            st.write("Confusion Matrix:")
            st.write(pd.DataFrame(conf_matrix, columns=class_labels, index=class_labels))
            # st.write(f'Confusion Matrix:\n{conf_matrix}') # old code
                # st.write(f'Precision per class: {precision}') # old code
                # st.write(f'Recall per class: {recall}') # old code
            st.write(f'Precision per class: {np.round(precision, 2).tolist()}')
            st.write(f'Recall per class: {np.round(recall, 2).tolist()}')
        if st.button('Visualize a single prediction!'):
            st.session_state['predict_button_clicked'] = True
    # Create a variable in session state to check if the image is uploaded after clicking the button
            
        if 'predict_button_clicked' in st.session_state and st.session_state['predict_button_clicked']:
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="new-upload")
            if uploaded_file is not None and 'tmodel' in st.session_state and st.session_state['tmodel'] is not None:
        # Display the uploaded image
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Load the model from the session state
                model = st.session_state['tmodel']
                model.eval()  # Set the model to evaluation mode

        # Get class labels dynamically from the DataLoader of the test set
                class_labels = data_loaders['test'].dataset.classes

        # Make prediction
                predicted_label, probabilities = predict(uploaded_file, model, class_labels)

        # Display predicted label and probabilities
                st.write(f'Predicted Label: {predicted_label}')
                st.write('Probabilities:')
                st.write(pd.DataFrame([probabilities], columns=class_labels))

        # Reset the prediction button clicked state
                st.session_state['predict_button_clicked'] = False

                
            # Cleanup temporary directories
        for temp_dir in [train_temp_dir, valid_temp_dir, test_temp_dir]:
            print(temp_dir)
            print(os.listdir(temp_dir))
            if temp_dir:
                shutil.rmtree(temp_dir)
else:
    st.error("Error: Could not prepare data loaders. Check the uploaded files.")

def download_zip_from_path(filepath, filename="model_params.zip"):
# Downloads a zip file directly from its path, triggering a "Save As" window.
  with open(filepath, 'rb') as f:
      file_size = os.path.getsize(filepath)
      st.header(f'Download {filename}')
      st.write(f'<a href="{filepath}" download="{filename}">Download Zip File</a>', unsafe_allow_html=True)

def create_model_properties():
# Summarize model properties
  data = {
      "name": "dog_breed_cnn",
      "description": "CNN built to train on datasets consisting of images of 3 separate dog breeds. It learns to classify input images as one of the three breeds.",
      "codeType": "python",
      "algorithm": "cnn",
      "function": "image classification",
      "imports": "streamlit, torch, torchvision, sklearn",
      "dataPreparation": "random resizing and cropping (training only), horizontal flipping (training only), normalization (all datasets)",
      "activationMethod": "ReLU",
      "poolingMethod": "max pooling layers for downsampling, adaptive average pooling for varying input sizes",
      "batchSize": "20",
      "learningRateScheduler": "initial learning rate lr=-.001, step_size=7, gamma=0.1)",
      "probabilityOutput": "model estimates the probabilities that a given image belongs to Class 1, Class 2, or Class 3",
      "prediction": "model predicts that a given image belongs to the Class with the largest probability estimate",
      "tool": "Python 3",
      "toolVersion": "3.11.5"
  }
  with open("model_properties.json", "w") as f:
      import json
      json.dump(data, f, indent=4)

def create_input_info():
# In lieu of a json input file, the app will provide the following overview to the user.
  data = {
      "NOTE": "A JSON INPUT FILE IS NOT NEEDED FOR THIS APP. ONLY IMAGES MAY BE UPLOADED",
      "-------------": "-------------",
      "inputInstructions": "Prepare three zip folders of dog images, which may be called 'train', 'valid', and 'test' or similar. The structure of the input zip folder is: Root ZIP Folder: 'train'... First-Level Folder: 'train'... Second-Level Folders: 'Breed1', 'Breed2', 'Breed3'. Substitute the names of your selected breeds as the second-level folder names. These names will become the class labels of the confusion matrix displayed in the app.",
      "imagePreprocessingDescription": "Images are resized to a uniform dimension (224x224 pixels) to match expected input size. Data augmentation techniques such as random cropping, flipping, and rotation to improve model generalization. Additional transformations include normalization (using mean and standard deviation values of the ImageNet dataset)."
  }
  with open("input_info.json", "w") as f:
      import json
      json.dump(data, f, indent=4)

def create_output_info():
# description of model output
    data = [
    {
        "name": "Probabilities",
        "description": "a probability is estimated for each data class (aka dog breed). data classes labels are scraped from the folder names provided by the app user",
        "role": "output",
        "type": "decimal",
        "level": "interval",
        "format": "#.####",
        "aggregation": "",
    },
    {
        "name": "Predicted Label",
        "description": "the model finds the maximum DATA_CLASS_PROBABILITY value and assigns its class label to the image in question",
        "role": "output",
        "type": "string",
        "level": "nominal",
        "format": "",
        "aggregation": "",
    }
    ]

    with open("output_info.json", "w") as f:
      import json
      json.dump(data, f, indent=4)

# Define the content of the score_code.py file
score_code_content = "# Score is obtained via the test button in the Streamlit app. No score code file is necessary."
# Save the content to score_code.py
with open("score_code.py", "w") as f:
    f.write(score_code_content)

# Export model parameters button
if st.button("Export Model"):
    model = model_scratch # Initialize the model
    model_params_dir = tempfile.mkdtemp()  # Temporary directory for model parameters

    # Dummy training to get model parameters (replace with actual training code)
    dummy_data = torch.randn(2, 3, 224, 224)
    dummy_output = model(dummy_data)

    # Save model parameters to JSON
    model_params = get_model_params(model)
    json_path = os.path.join(model_params_dir, 'model_params.json')
    with open(json_path, 'w') as f:
        json.dump(model_params, f, indent=4)

    # Save model parameters to a pickle file
    pkl_path = os.path.join(model_params_dir, 'model.pkl')
    save_model_params(model_scratch, pkl_path)

    # Save the informational json files
    create_model_properties()
    mp_json_path = "model_properties.json"
    create_input_info()
    input_json_path = "input_info.json"
    create_output_info()
    output_json_path = "output_info.json"

    # Zip model parameters
    zip_filename = 'model_params.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zip_ref:
        zip_ref.write(json_path, arcname=os.path.basename(json_path))
        zip_ref.write(pkl_path, arcname=os.path.basename(pkl_path))
        zip_ref.write(model_path, arcname=os.path.basename(model_path))
        zip_ref.write(mp_json_path, arcname=os.path.basename(mp_json_path))
        zip_ref.write(input_json_path, arcname=os.path.basename(input_json_path))
        zip_ref.write(output_json_path, arcname=os.path.basename(output_json_path))
        zip_ref.write("score_code.py")

    # Download the zip file using st.download_button
    st.download_button(
        label="Download Zip File",
        data=open(zip_filename, 'rb').read(),
        file_name=zip_filename,
        mime="application/zip"
    )












# # Load the model
# model_path = "C:/Users/24221/Downloads/dogImages/saved_models/model_scratch.pt"  # Replace with your model path
# model = Net()
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()

# # Function to preprocess the image
# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     image = transform(image).unsqueeze(0)
#     return image

# # Function to make prediction
# def predict(image):
#     # Preprocess the image
#     image = preprocess_image(image)
    
#     # Perform prediction
#     with torch.no_grad():
#         output = model(image)
#         probabilities = torch.softmax(output, dim=1)[0]
#         predicted_class_index = torch.argmax(probabilities).item()
#         return predicted_class_index, probabilities

# class_to_breed = {0: '006.American_eskimo_dog', 1: '040.Bulldog', 2: '124.Poodle'}

# # Streamlit app
# st.title("Hi yall! Let's see if this model works!")
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# if uploaded_file is not None:
#     # Load and display the image
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_column_width=True)
    
#     # Make prediction
#     predicted_class_index, probabilities = predict(image)
#     predicted_breed = class_to_breed.get(predicted_class_index, 'Unknown')
#     st.write(f'Predicted Dog Breed: {predicted_breed}')
#     st.write(f'Probabilities: {probabilities}')

