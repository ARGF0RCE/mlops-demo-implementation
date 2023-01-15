import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Simple Leaf Disease Classifier')
st.text("Provide URL of bean Image for image classifcation")

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('/app/models/')
    return model

global model
with st.spinner('Loading model into memory...'):
    model = load_model()

classes = ['angular_leaf_spot', 'bean_rust', 'healthy']

def decode_img(image):
    img = tf.image.decode_jpeg(image, channels=3)
    img = tf.image.resize(img, [224, 224])
    return np.expand_dims(img, axis=0)

path = st.text_input('Enter Image URL to classify: ' ,'https://thumbs.dreamstime.com/b/kidney-bean-leaf-white-closeup-isolated-background-69756304.jpg')
if path is not None:
    content = requests.get(path).content

    st.write('Predicted class: ')
    with st.spinner('Classifying...'):
        label = np.argmax(model.predict(decode_img(content)), axis=1)
        st.write(classes[label[0]])
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption="Classifying Bean Image", use_column_width=True)
    