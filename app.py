import streamlit as st
import tensorflow as tf
# st.set_option('deprication.showfileUploderEncoding',False)
# st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def load_model():
    model=tf.keras.models.load_model('my_model.hdf5')
    return model
model=load_model()
st.write(""" image classification """)
file=st.file_uploader("please upload your file",type=['jpg','png'])
import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(Image_data,model):
    size=(32,32,3)
    image=ImageOps(Image_data, size)
    img=np.arrayshape(image)
    img.reshape= img[32,32,3]
    prediction=model.predict(img.reshape)
    return prediction
if file is None:
    st.text("please upload a pic")
else:
    image=Image.open(file)
    model=tf.keras.models.load_model('my_model.hdf5')
    st.image(image, use_column_width=True)
    preditions=import_and_predict(image,model)
    class_name=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    string=["this is most likey :"+class_names[np.argmax(predictions)]]
    st.success(string)
