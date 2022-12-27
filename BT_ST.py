# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 18:48:31 2022

@author: derli
"""

import numpy as np
import keras
from PIL import Image,ImageOps
#from keras.models import load_model
import streamlit as st


def getResult(img,file_name):
    file_name = r'D:\DL\brain tumor detection\BrainTumorcategorical.h5'
    model=keras.models.load_model(file_name)
    data=np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)
    image=img
    size=(64,64)
    image=ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array=np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0]=normalized_image_array
    prediction=model.predict(data)
    return np.argmax(prediction)
	
	
uploaded_file=st.file_uploader("Browse a file ...", type=["jpg"])
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption='Uploaded MRI.', use_column_width=True)
    label=getResult(image,'D:\DL\brain tumor detection\BrainTumorcategorical.h5')
    if label==0:
        st.write("The MRI scan is healthy")
    else:
        st.write("The MRI scan has a brain tumor")
        
        
def main():
    st.title("Brain Tumor MRI Classification")
    st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")
    

if __name__== '__main__':
    main()