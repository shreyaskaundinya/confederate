import os
import pickle
import random
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

from p import predicted_label


def predict(img):
    img_size = 224
    ro = "C://Users//Shrey//Documents//PROJECTS//weal-hack//flwr_server//model.h5"
    model = keras.models.load_model(ro)
    try:
        model = pickle.load(open('model.h5', 'rb'))
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_resize = cv2.resize(img, (img_size, img_size))
        img_resize = img_resize.reshape(1, img_size, img_size, 1)
        print(img_resize.shape)
        res = model.predict(img_resize)
        predicted_label(res[0][0])
    except:
        pass
    time.sleep(3)
    # z = 1/(1 + np.exp(-/x))
    return predicted_label(img)

st.title('ConFederate')
st.subheader("Pneumonia Detection System using ConFederate")

image = st.file_uploader("Pick an image!")

if image:
    # print("GOT EM IMAGE")
    root = f"C://Users//Shrey//Documents//PROJECTS//weal-hack//flwr_server//{image.name}"

    with st.spinner('Wait for it...'):
        predict(root+image.name)

    st.image(cv2.imread(root))
    # run through model
    label = "pneumonia"

    if label == "pneumonia":
        st.warning('Label: ' + label, icon="ℹ️")
    else:
        st.success('Label: ' + label, icon="ℹ️")
