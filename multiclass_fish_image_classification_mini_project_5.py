# Mutliclass fish image classification

import os, zipfile, numpy as np, streamlit as st, tensorflow as tf
from PIL import Image

ZIP_PATH = r"C:\Users\H T Abhinav\Downloads\Dataset.zip"
EXT_PATH = r"C:\Users\H T Abhinav\Downloads\Dataset"
W_PATH = r"C:\Users\H T Abhinav\Downloads\best_fish_classifier_mobilenet.h5"

@st.cache_resource
def init_app():
    if not os.path.exists(EXT_PATH) and os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as z: z.extractall(EXT_PATH)
    t_dir = os.path.join(EXT_PATH, "images.cv_jzk6llhf18tm3k0kyttxz", "data", "train")
    lbls = sorted([f for f in os.listdir(t_dir) if os.path.isdir(os.path.join(t_dir, f))]) if os.path.exists(t_dir) else [f"Class_{i}" for i in range(11)]
    
    base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    mdl = tf.keras.Sequential([
        base, tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, 'relu'), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(lbls), 'softmax')
    ])
    mdl.load_weights(W_PATH)
    return mdl, lbls

st.title("🐟 Fish Species Classifier")
model, classes = init_app()
file = st.file_uploader("Upload fish image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB").resize((224, 224))
    st.image(img, use_container_width=True)
    
    with st.spinner("Analyzing image..."):
        x = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(np.array(img), 0))
        preds = model.predict(x)[0]
        idx = np.argmax(preds)
        conf = float(preds[idx])
        
    st.subheader(f"Prediction: {classes[idx]}")
    st.metric("Confidence Score", f"{conf:.2%}")
    st.progress(conf)


















