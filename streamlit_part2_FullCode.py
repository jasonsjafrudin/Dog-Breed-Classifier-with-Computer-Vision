# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 01:44:30 2024

@author: Xulin Chen
"""

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# Define CSV path for storing predictions and feedback
feedback_file_path = 'feedback_history_full.csv'

# Classifier model
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # Output: 16 channels
       self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization after Convolutional layers
       self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
       self.bn2 = nn.BatchNorm2d(32)
       self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
       self.bn3 = nn.BatchNorm2d(64)
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

# Load the model
model_path = "C:/Users/24221/Downloads/model.pt" # Replace with your model path
model = Net()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Function to preprocess the image
def preprocess_image(image):
    # If the image is not already a PIL Image, open it
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    # Convert the image to RGB regardless of its initial mode
    image = image.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to make prediction
def predict(image):
   image = preprocess_image(image)
   with torch.no_grad():
       output = model(image)
       probabilities = torch.softmax(output, dim=1)[0]
       predicted_class_index = torch.argmax(probabilities).item()
       return predicted_class_index, probabilities

class_to_breed = {0: 'American Eskimo', 1: 'Dalmatian', 2: 'Golden Retriever', 3: 'Other Breed/Not a Dog'}
breed_to_class = {v: k for k, v in class_to_breed.items()}

# Initialize or load feedback history
if os.path.exists(feedback_file_path):
    feedback_df = pd.read_csv(feedback_file_path)
else:
    feedback_df = pd.DataFrame(columns=['filename', 'predicted', 'true'])

# Initialize session state variables
if 'submit_feedback_clicked' not in st.session_state:
    st.session_state.submit_feedback_clicked = False
if 'review_feedback_clicked' not in st.session_state:
    st.session_state.review_feedback_clicked = False
if 'sidebar_password_visible' not in st.session_state:
    st.session_state.sidebar_password_visible = False
if 'password' not in st.session_state:
    st.session_state.password = ''
if 'password_entered' not in st.session_state:
    st.session_state.password_entered = False

# Calculate confusion matrix, metrics, & accuracy
if not feedback_df.empty:
    y_true = feedback_df['true']
    y_pred = feedback_df['predicted']
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(class_to_breed.keys()))
    column_names = pd.MultiIndex.from_product([["Predicted"], list(class_to_breed.values())])
    row_names = pd.MultiIndex.from_product([["Actual"], list(class_to_breed.values())])
    conf_matrix_df = pd.DataFrame(conf_matrix, index=row_names, columns=column_names)
    conf_matrix_df_melt = pd.DataFrame(conf_matrix,
                                       index=list(class_to_breed.values()),
                                       columns=list(class_to_breed.values()))
    conf_matrix_df_melt = conf_matrix_df_melt.reset_index().melt(id_vars='index')
    conf_matrix_df_melt.columns = ['Actual', 'Predicted', 'Count']
    conf_matrix_df_melt['Percentage'] = \
      conf_matrix_df_melt.groupby('Actual')['Count'].transform(lambda x: x / x.sum() * 100)
    
    clasf_report = classification_report(
        list(map(lambda x: class_to_breed[x], y_true)),
        list(map(lambda x: class_to_breed[x], y_pred)),
        output_dict = True, labels = list(class_to_breed.values()))
    clasf_report.pop("micro avg", None)
    clasf_report.pop("accuracy", None)
    clasf_report_df = pd.DataFrame(clasf_report).T

    clasf_accuracy = accuracy_score(y_true, y_pred)
    
    print("Confusion matrix calculated:")
    print(conf_matrix_df)
else:
    conf_matrix_df = None
    print("No feedback data found.")

st.title("Dog Breed Classifier")
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files:
    predictions = []
    feedback = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)

        predicted_class_index, probabilities = predict(image)
        predicted_breed = class_to_breed.get(predicted_class_index, 'Unknown')
        predictions.append((uploaded_file.name, predicted_breed, probabilities))

        st.write(f'**{uploaded_file.name}:** Predicted Breed: {predicted_breed}')
        st.write('Probabilities:')
        for breed, prob in zip(class_to_breed.values(), probabilities):
            st.write(f'{breed}: {prob:.4f}')

        feedback_option = st.selectbox(
            f'Select the correct breed for {uploaded_file.name}:',
            ['Select'] + list(class_to_breed.values()),
            key=f'feedback_{uploaded_file.name}'
        )
        if feedback_option != 'Select':
            true_class = breed_to_class[feedback_option]
            feedback.append((uploaded_file.name, predicted_class_index, true_class))

    # Show the 'Submit Feedback' button
    if not st.session_state.submit_feedback_clicked:
        if st.button('Submit Feedback'):
            st.session_state.submit_feedback_clicked = True

    # Process feedback after 'Submit Feedback' button is clicked
    if st.session_state.submit_feedback_clicked:
        if feedback:
            new_feedback_df = pd.DataFrame(feedback, columns=['filename', 'predicted', 'true'])
            feedback_df = pd.concat([feedback_df, new_feedback_df])\
                          .drop_duplicates(subset=['filename'], keep="last")\
                          .reset_index(drop=True)
            feedback_df.to_csv(feedback_file_path, index=False)

            if not new_feedback_df.empty:
               new_y_true = new_feedback_df['true']
               new_y_pred = new_feedback_df['predicted']
               new_conf_matrix = confusion_matrix(new_y_true, new_y_pred, labels=list(class_to_breed.keys()))
               new_conf_matrix_df_melt = pd.DataFrame(new_conf_matrix,
                                                index=list(class_to_breed.values()),
                                                columns=list(class_to_breed.values()))
               new_conf_matrix_df_melt = new_conf_matrix_df_melt.reset_index().melt(id_vars='index')
               new_conf_matrix_df_melt.columns = ['Actual', 'Predicted', 'Count']
               new_conf_matrix_df_melt['Percentage'] = \
               new_conf_matrix_df_melt.groupby('Actual')['Count'].transform(lambda x: x / x.sum() * 100)
         
            st.success('Feedback submitted successfully!')
        else:
            st.warning('No feedback provided.')

        # Show the 'Review Feedback' button after 'Submit Feedback' is clicked
        if not st.session_state.review_feedback_clicked:
            if st.button('Review Feedback'):
                st.session_state.review_feedback_clicked = True

    # Show the sidebar password input after 'Review Feedback' button is clicked
    if st.session_state.review_feedback_clicked:
        if not st.session_state.sidebar_password_visible:
            st.session_state.password = st.sidebar.text_input('Enter the password to view feedback summary:', type='password')
            if st.sidebar.button('Enter'):
                if st.session_state.password == 'pawsword':
                    st.session_state.password_entered = True
                    st.session_state.sidebar_password_visible = True  # Set visible only after correct password
                else:
                    st.sidebar.error('Incorrect password. Access denied.')

        # Display the confusion matrix if the correct password is entered
        if st.session_state.password_entered:
            if conf_matrix_df is not None:
                st.write('Confusion Matrix:')
                st.write(conf_matrix_df.to_html(), unsafe_allow_html=True) # Display confusion matrix as table

                color_map = {
                    'American Eskimo': 'blue',
                    'Dalmatian': 'red',
                    'Golden Retriever': 'green',
                    'Other Breed/Not a Dog': 'purple'
                }

                # Create the stacked bar chart
                conf_matrix_chart = go.Figure()
                new_conf_matrix_chart = go.Figure()
##                conf_matrix_chart_count = go.Figure()

                for actual_label in class_to_breed.values():
                    colors = []
                    patterns = []
                    for pred_label in class_to_breed.values():
                        if actual_label == pred_label:
                            patterns.append('')  # No pattern for correct predictions
                        else:
                            patterns.append('x')  # 'x' pattern for incorrect predictions
                        colors.append(color_map[actual_label])
                    
                    conf_matrix_chart.add_trace(go.Bar(
                        x=conf_matrix_df_melt[conf_matrix_df_melt['Predicted'] == actual_label]['Actual'],
                        y=conf_matrix_df_melt[conf_matrix_df_melt['Predicted'] == actual_label]['Count'],
                        name=f'Predicted {actual_label}',
                        marker_color=colors,
                        marker_pattern_shape=patterns,
                        showlegend=False
                    ))
                    new_conf_matrix_chart.add_trace(go.Bar(
                        x=new_conf_matrix_df_melt[new_conf_matrix_df_melt['Predicted'] == actual_label]['Actual'],
                        y=new_conf_matrix_df_melt[new_conf_matrix_df_melt['Predicted'] == actual_label]['Count'],
                        name=f'Predicted {actual_label}',
                        marker_color=colors,
                        marker_pattern_shape=patterns,
                        showlegend=False
                    ))

                for actual_label, color in color_map.items():
                    conf_matrix_chart.add_trace(go.Bar(
                        x=[None], y=[None],
                        marker=dict(color=color),
                        name=f'{actual_label}',
                        showlegend=True
                    ))
                    new_conf_matrix_chart.add_trace(go.Bar(
                        x=[None], y=[None],
                        marker=dict(color=color),
                        name=f'{actual_label}',
                        showlegend=True
                    ))


                correct_trace = go.Bar(
                    x=[None], y=[None],
                    marker_pattern_shape='',
                    marker=dict(color='black'),
                    name='Correct Prediction',
                    showlegend=True
                )
                incorrect_trace = go.Bar(
                    x=[None], y=[None],
                    marker_pattern_shape='x',
                    marker=dict(color='black'),
                    name='Incorrect Prediction',
                    showlegend=True
                )
                conf_matrix_chart.add_trace(correct_trace)
                conf_matrix_chart.add_trace(incorrect_trace)
                new_conf_matrix_chart.add_trace(correct_trace)
                new_conf_matrix_chart.add_trace(incorrect_trace)


                # Customize the layout
                conf_matrix_chart.update_layout(barmode='stack', title='Cumulative - Confusion Matrix - Stacked Bar Chart',
                                                xaxis_title='Actual', yaxis_title='Count',
                                                legend_title='Predicted')
                new_conf_matrix_chart.update_layout(barmode='stack', title='Session - Confusion Matrix - Stacked Bar Chart',
                                                xaxis_title='Actual', yaxis_title='Count',
                                                legend_title='Predicted')

                st.plotly_chart(conf_matrix_chart)
                st.plotly_chart(new_conf_matrix_chart)
                
                st.write('\nPrecision, Recall, F1-score, Support Table:')
                st.write(clasf_report_df.to_html(), unsafe_allow_html=True)
                st.write(f"\nAccuracy: {clasf_accuracy}")
            else:
                st.warning('No feedback history found.')
