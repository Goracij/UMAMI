############################################################################################
# IMPORTS
############################################################################################
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

#import tensorflow.keras.backend as K
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay


import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
#import tensorflow.keras as keras

from tensorflow.keras import layers
from tensorflow.keras import regularizers


from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Flatten

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical



from IPython import display
from PIL import Image


import pathlib
import shutil
import tempfile
import concurrent

import pickle
import visualkeras

import json
import glob

#import panel as pn
import streamlit
from streamlit_image_zoom import image_zoom
import joblib


print(tf.config.list_physical_devices())
print('Streamlit ver.: ' + streamlit.__version__)
print('SciKit-learn ver.: ' + sklearn.__version__)
print('TensorFlow ver.: ' + tf.__version__)
print('Joblib ver.: ' + joblib.__version__)

gpu_dev = tf.config.experimental.list_physical_devices('GPU')
for itm in gpu_dev:
    tf.config.experimental.set_memory_growth(itm, True)
############################################################################################
# 1.0 CONSTANTS
############################################################################################
TREE_MODEL = './models/final/random_forest_model.pkl'
LAYER_2 = './models/final/Nadine_food-101-EfNetB3-A0.__-earlystop-E__of45-B32_softCat_v1.1b.keras'
SEED = 111
TABLE = './data/food-101/Final_table_dish_info.csv'

TEST_MODEL = './models/final/6_7_final_classification101_EfficientNet.keras'
############################################################################################
# 1.1 FUNCTIONS - https://www.analyticsvidhya.com/blog/2023/12/grad-cam-in-deep-learning/
############################################################################################

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    ## First, we create a model that maps the input image to the activations
    ## of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    ## Then, we compute the gradient of the top predicted class for our input image
    ## for the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    ## We are doing transfer learning on last layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    ## This is a vector where each entry is the mean intensity of the gradient
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    ## calculates a heatmap highlighting the regions of importance in an image
    ## for a specific 
    ## predicted class by combining the output of the last convolutional layer
    ## with the pooled gradients.
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    ## For visualization purpose
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def display_gradcam(img, heatmap, alpha=0.4):
    ## Loading the original image
    img = tf.keras.utils.img_to_array(img)

    ## Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    ## Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    ## Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    ## Superimpose the heatmap on original image
    Superimposed_img = jet_heatmap * alpha + img
    Superimposed_img = tf.keras.utils.array_to_img(Superimposed_img)

    ## Displaying Grad CAM
    #plt.imshow(Superimposed_img)
    #plt.show()
    return Superimposed_img
############################################################################################
# 2.0 TABLE LOAD
############################################################################################
data_df = pd.read_csv(TABLE)
data_df.columns = [val.lower().strip().replace(' ','_') for val in data_df.columns.tolist()]

############################################################################################
# 2.1 MODELS LOAD
############################################################################################
layer_1_base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
layer_1_tree = joblib.load(TREE_MODEL)
layer_2 = tf.keras.models.load_model(LAYER_2)

############################################################################################
# 5. STREAMLIT - THE BEGINNING
############################################################################################
option = streamlit.selectbox('Select image...', ('Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'))
match option:
    case 'Image 1': image_path='./data/sushi.jpg'
    case 'Image 2': image_path='./data/sandwich.jpg'
    case 'Image 3': image_path='./data/railroad_train.jpg'
    case 'Image 4': image_path='./data/pizza.jpg'
    case 'Image 5': image_path='./data/test_apple_pie_image.jpg'

force_heatmap = streamlit.checkbox('Force layer 2 and heatmap', value=False, key=None, help=None, on_change=None, label_visibility="visible")
############################################################################################
# 5.1 IMAGE LOAD
############################################################################################
streamlit.session_state.uploaded = streamlit.sidebar.file_uploader(
        "Choose an image file",
        accept_multiple_files=False,
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        key="file_uploader1")
    
if streamlit.session_state.uploaded is not None:
        image_path = streamlit.session_state.uploaded
else:
        image_path = './data/sushi.jpg'



img_full = tf.keras.utils.load_img(image_path)

img_l1 = tf.keras.utils.load_img(image_path, target_size=(224, 224))
img_array_l1 = tf.keras.utils.img_to_array(img_l1)

img_l2 = tf.keras.utils.load_img(image_path, target_size=(300, 300))
img_array_l2 = tf.keras.utils.img_to_array(img_l2)

streamlit.sidebar.page_link(page="main_app.py", label="Main", icon="🏠")
#streamlit.sidebar.page_link(page="pages/test.py", label="Test", icon="🔮")
streamlit.image(img_full, width=600)
if streamlit.button("Predict", type="primary"):
    ############################################################################################
    # 5.2 LAYER 1 PREDICT
    ############################################################################################
    features = layer_1_base.predict(tf.keras.applications.resnet.preprocess_input(np.expand_dims(img_array_l1, axis=0))) # Get feature predictions
    prediction = layer_1_tree.predict(features.reshape(1, -1))

    l1_prob = prediction
    l1_label = "Food" if prediction[0] == 0 else "Non-Food"
    streamlit.write(f"Predicted class: {l1_label}.")
    if (l1_label=='Food') or (force_heatmap):
        ############################################################################################
        # 5.3 LAYER 2 PREDICT
        ############################################################################################
        pred_l2 = layer_2.predict(tf.keras.preprocessing.image.smart_resize(tf.expand_dims(img_array_l2, axis=0), size=(300,300)))
        l2_label = str(data_df.dish_name[np.argmax(pred_l2, axis = -1)].astype(str).values)
        l2_prob = pred_l2.max()
        streamlit.write(f'Predicted class: {str(l2_label)}, probability: {str(l2_prob)}.')
        ############################################################################################
        # 4. IMAGE and GRADCAM HEATMAP SHOW
        ############################################################################################
        #plt.imshow(img_full)
        #plt.show()
        heatmap = make_gradcam_heatmap(img_array=tf.expand_dims(img_array_l2, axis=0), model=layer_2, last_conv_layer_name=layer_2.layers[6].name)
        #plt.matshow(heatmap)
        #plt.show()
        heated_image = display_gradcam(img=img_full, heatmap=heatmap, alpha=2)
        #streamlit.image(heated_image)
        image_zoom(heated_image, mode="scroll", keep_aspect_ratio=True, size=(600,600), zoom_factor=4.0, increment=0.2)
    else:
            streamlit.write('No food was found on the image.')
else:
    streamlit.write('Example image.')
