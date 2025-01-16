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
import PIL

#import panel as pn
import streamlit as st
from streamlit_image_zoom import image_zoom
import joblib
from copy import deepcopy

print(tf.config.list_physical_devices())
print('Streamlit ver.: ' + st.__version__)
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
LOGO = './images/Umami_logo_vertical.png'



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

def goto(linenum):
    global line
    line = linenum

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
st.sidebar.image(LOGO, width=300)
left_col, right_col = st.columns(2)
camera_on = False

source = st.sidebar.radio("Select source of an image", ("Built-in examples","Load from computer","Camera"),index=0)

match source:
    case "Built-in examples":
        option = st.sidebar.selectbox('Try out built-in examples...',('Example 1 - sushi',
                                            'Example 2 - sandwich',
                                            'Example 3 - DeutscheBahn',
                                            'Example 4 - pizza',
                                            'Example 5 - apple pie'), index=0)
        match option:
            case 'Example 1 - sushi': image_path='./data/sushi.jpg'
            case 'Example 2 - sandwich': image_path='./data/sandwich.jpg'
            case 'Example 3 - DeutscheBahn': image_path='./data/railroad_train.jpg'
            case 'Example 4 - pizza': image_path='./data/pizza.jpg'
            case 'Example 5 - apple pie': image_path='./data/test_apple_pie_image.jpg'
        with left_col:
            st.image(tf.keras.utils.load_img(image_path), width=400)
    case "Load from computer":
        st.session_state.uploaded = st.sidebar.file_uploader(
            "Choose your own image file:",
            accept_multiple_files=False,
            type=["png", "jpg", "jpeg", "tiff", "bmp"],
            key="file_uploader1")
        if st.session_state.uploaded is not None:
            image_path = st.session_state.uploaded
        else:
            image_path = './data/sushi.jpg'
        with left_col:
            st.image(tf.keras.utils.load_img(image_path), width=400)
    case "Camera":
        with left_col:
            camera_on = True
            picture = st.camera_input("Take a picture", disabled=False)
            if picture is not None:
                bytes_data = picture.getvalue()
                img_tensor = tf.io.decode_image(bytes_data, channels=3, dtype=tf.uint8)




force_heatmap = st.sidebar.checkbox('Force food recognition.', value=False, key=None, help=None, on_change=None, label_visibility="visible")
############################################################################################
# 5.1 IMAGE LOAD
############################################################################################



if camera_on and (picture is not None):
    tf.keras.utils.save_img('./data/CAMERA.jpg', img_tensor, data_format=None, file_format=None, scale=True)
    img_full = tf.keras.utils.load_img('./data/CAMERA.jpg')
    img_l1 = tf.keras.utils.load_img('./data/CAMERA.jpg', target_size=(224, 224))
    img_l2 = tf.keras.utils.load_img('./data/CAMERA.jpg', target_size=(300, 300))
    if os.path.isfile('./data/CAMERA.jpg'):
        os.remove('./data/CAMERA.jpg')
    img_array_l1 = tf.keras.utils.img_to_array(img_l1)
    img_array_l2 = tf.keras.utils.img_to_array(img_l2)


#    This way it DOESN'T work!?
#    img_full = tf.keras.utils.array_to_img(img_tensor)
#    img_l1 = tf.image.resize(img_full, size=(224,224))
#    img_l2 = tf.image.resize(img_full, size=(300,300))
#    img_array_l1 = tf.keras.utils.img_to_array(img_l1).setflags(write=1)
#    img_array_l2 = tf.keras.utils.img_to_array(img_l2).setflags(write=1)
#    st.write(img_array_l1.flags)


#    img_l1 = tf.keras.utils.array_to_img(img_full, dtype=tf.uint8)
#    img_l2 = tf.keras.utils.array_to_img(img_full, dtype=tf.uint8)


elif not camera_on:
    img_full = tf.keras.utils.load_img(image_path)

    img_l1 = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_l2 = tf.keras.utils.load_img(image_path, target_size=(300, 300))
        
    img_array_l1 = tf.keras.utils.img_to_array(img_l1)
    img_array_l2 = tf.keras.utils.img_to_array(img_l2)
else:
    st.write("Waiting for the photo...")













#st.sidebar.page_link(page="main_app.py", label="Main", icon="🏠")
#st.sidebar.page_link(page="pages/test.py", label="Test", icon="🔮")

if (not camera_on) or (picture is not None):
    if st.button("What is it?", type="primary"):
        ############################################################################################
        # 5.2 LAYER 1 PREDICT
        ############################################################################################
        features = layer_1_base.predict(tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img_array_l1, axis=0)))
        prediction = layer_1_tree.predict(features.reshape(1, -1))

        l1_prob = prediction
        l1_label = "Food" if prediction[0] == 0 else "Non-Food"
        probable_labels = []
        probable_cert = []

        if (l1_label=='Food'):
            st.write(f"Well, it's a :green[**food**].")
            ############################################################################################
            # 5.3 LAYER 2 PREDICT
            ############################################################################################
            pred_l2 = layer_2.predict(tf.keras.preprocessing.image.smart_resize(tf.expand_dims(img_array_l2, axis=0), size=(300,300)))
            l2_label = str(data_df.at[int(np.argmax(pred_l2, axis = -1)), 'dish_name'])
            l2_prob = pred_l2.max() * 100
            l2_label = str(l2_label).replace("['","").replace("']","").replace("_"," ").upper()
            if (l2_prob.round(2) >=90):
                st.write(f"I can bet it is :red[**{l2_label}**]! {data_df.at[int(np.argmax(pred_l2, axis = -1)), 'flags']} (certainty {str(l2_prob.round(2))}%)")
            elif (l2_prob.round(2) < 90) and (l2_prob.round(2) >= 75):
                st.write(f"I pretty much sure it is :red[**{l2_label}**]. {data_df.at[int(np.argmax(pred_l2, axis = -1)), 'flags']} (certainty {str(l2_prob.round(2))}%)")
            elif (l2_prob.round(2) <75) and (l2_prob.round(2) >=55):
                st.write(f"I tend to believe it is :red[**{l2_label}**]. {data_df.at[int(np.argmax(pred_l2, axis = -1)), 'flags']} (certainty {str(l2_prob.round(2))}%)")
            elif (l2_prob.round(2) <55) and (l2_prob.round(2) >=35):
                st.write(f"I'd guess it is :red[**{l2_label}**]. {data_df.at[int(np.argmax(pred_l2, axis = -1)), 'flags']} (certainty {str(l2_prob.round(2))}%)")
            elif (l2_prob.round(2) <35) and (l2_prob.round(2) >=15):
                st.write(f"Oh, this one is tough! Maybe :red[**{l2_label}**]?. {data_df.at[int(np.argmax(pred_l2, axis = -1)), 'flags']} (certainty {str(l2_prob.round(2))}%)")
            else:
                st.write(f"If I had to guess, I'd say it is :red[**{l2_label}**]. Are you sure you can eat THAT? {data_df.at[int(np.argmax(pred_l2, axis = -1)), 'flags']} (certainty {str(l2_prob.round(2))}%)")
            ############################################################################################
            # 4. IMAGE and GRADCAM HEATMAP SHOW
            ############################################################################################
            heatmap = make_gradcam_heatmap(img_array=tf.expand_dims(img_array_l2, axis=0), model=layer_2, last_conv_layer_name=layer_2.layers[6].name)
            #plt.matshow(heatmap)
            #plt.show()
            heated_image = display_gradcam(img=img_full, heatmap=heatmap, alpha=2)
            with right_col:
                st.image(heated_image, width=400)


            
            
            
            st.subheader("Be aware, as it may contain:")
            allergy_columns = []
            col_01, col_02, col_03, col_04, col_05, col_06, col_07, col_08 = st.columns(8)


            
            tab1, tab2 = st.tabs(["Ingredients", "Nutrition"])
            with tab1:
                st.subheader("Main ingredients:")
                st.write(data_df.at[int(np.argmax(pred_l2, axis = -1)), 'classic_recipe'].replace("Time:","Cooking time, ca.:"))
                st.subheader("Optional ingredients")
                st.write(data_df.at[int(np.argmax(pred_l2, axis = -1)), 'optional_ingredients'])
            with tab2:
                st.subheader(f"{data_df.at[int(np.argmax(pred_l2, axis = -1)), 'nutrients_per_100g']} per 100g")
                nutr_df = data_df[['proteins','fats','carbs']].iloc[int(np.argmax(pred_l2, axis = -1))].apply(lambda x: x.split("g")[0])
                pie_data = [int(nutr_df.proteins),  int(nutr_df.fats), int(nutr_df.carbs)]
                labels = ['Proteins', 'Fats', 'Carbs']
                colours = {'Proteins': '#00b67b',
                            'Fats': '#ffd962',
                            'Carbs': '#7ca5f9'}
                fig, ax = plt.subplots()
                ax.pie(pie_data, explode=(0.1, 0, 0), labels=labels, autopct='%1.1f%%',shadow=True, startangle=90, colors=[colours[key] for key in labels])
                ax.axis('equal')  
                st.pyplot(fig)

            
            st.markdown("___")
            with st.expander("It may also be..."):
                for itr in range(5):
                    probable_labels.append( str(data_df.at[int(np.argmax(pred_l2, axis = -1)), 'dish_name']).replace("['","").replace("']","").replace("_"," ").upper() )
                    probable_cert.append(pred_l2.max()*100)
                    pred_l2 = np.delete(pred_l2, np.argmax(pred_l2))
                    if itr >0:
                        st.write(f"{itr}. {probable_labels[itr].upper()}, certainty: {probable_cert[itr].round(2)}%")


            
            
            
            




        elif (force_heatmap) and (l1_label=='Non-Food'):
            st.write(f"Since you insist...")
            ############################################################################################
            # 5.3 LAYER 2 PREDICT
            ############################################################################################
            pred_l2 = layer_2.predict(tf.keras.preprocessing.image.smart_resize(tf.expand_dims(img_array_l2, axis=0), size=(300,300)))
            l2_label = str(data_df.at[int(np.argmax(pred_l2, axis = -1)), 'dish_name'])
            l2_prob = pred_l2.max()
            l2_label = str(l2_label).replace("['","").replace("']","").replace("_"," ").upper()
            if (l2_prob.round(2) >=90):
                st.write(f"I can bet it is :red[**{l2_label}**]! {data_df.at[int(np.argmax(pred_l2, axis = -1)), 'flags']} (certainty {str(l2_prob.round(2))}%)")
            elif (l2_prob.round(2) < 90) and (l2_prob.round(2) >= 75):
                st.write(f"I pretty much sure it is :red[**{l2_label}**]. {data_df.at[int(np.argmax(pred_l2, axis = -1)), 'flags']} (certainty {str(l2_prob.round(2))}%)")
            elif (l2_prob.round(2) <75) and (l2_prob.round(2) >=55):
                st.write(f"I tend to believe it is :red[**{l2_label}**]. {data_df.at[int(np.argmax(pred_l2, axis = -1)), 'flags']} (certainty {str(l2_prob.round(2))}%)")
            elif (l2_prob.round(2) <55) and (l2_prob.round(2) >=35):
                st.write(f"I'd guess it is :red[**{l2_label}**]. {data_df.at[int(np.argmax(pred_l2, axis = -1)), 'flags']} (certainty {str(l2_prob.round(2))}%)")
            elif (l2_prob.round(2) <35) and (l2_prob.round(2) >=15):
                st.write(f"Oh, this one is tough! Maybe :red[**{l2_label}**]? {data_df.at[int(np.argmax(pred_l2, axis = -1)), 'flags']} (certainty {str(l2_prob.round(2))}%)")
            else:
                st.write(f"Are you sure you can eat THAT? If I had to guess, I'd say it is :red[**{l2_label}**]. {data_df.at[int(np.argmax(pred_l2, axis = -1)), 'flags']} (certainty {str(l2_prob.round(2) * 100)}%)")
            ############################################################################################
            # 4. IMAGE and GRADCAM HEATMAP SHOW
            ############################################################################################
            heatmap = make_gradcam_heatmap(img_array=tf.expand_dims(img_array_l2, axis=0), model=layer_2, last_conv_layer_name=layer_2.layers[6].name)
            #plt.matshow(heatmap)
            #plt.show()
            heated_image = display_gradcam(img=img_full, heatmap=heatmap, alpha=2)
            with right_col:
                st.image(heated_image, width=400)
            


            
            with st.expander("Other most probable classes:"):
                for itr in range(5):
                    probable_labels.append( str(data_df.at[int(np.argmax(pred_l2, axis = -1)), 'dish_name']).replace("['","").replace("']","").replace("_"," ").upper() )
                    probable_cert.append(pred_l2.max()*100)
                    pred_l2 = np.delete(pred_l2, np.argmax(pred_l2))
                    if itr >0:
                        st.write(f"{itr}. {probable_labels[itr].upper()}, probability {probable_cert[itr].round(2)}")


        else:
                st.write('No food was found on the image.')
    
