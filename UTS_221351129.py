import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model dan tools
interpreter = tf.lite.Interpreter(model_path="crop_recommendation.tflite")
interpreter.allocate_tensors()

scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul
st.title("Prediksi Kesehatan Janin")
st.write("Masukkan parameter hasil CTG untuk memprediksi status kesehatan janin.")

# Form input
baseline_value = st.number_input("Baseline Value", value=120.0)
accelerations = st.number_input("Accelerations", value=0.003)
fetal_movement = st.number_input("Fetal Movement", value=0.0)
uterine_contractions = st.number_input("Uterine Contractions", value=0.0)
light_decelerations = st.number_input("Light Decelerations", value=0.0)
severe_decelerations = st.number_input("Severe Decelerations", value=0.0)
prolongued_decelerations = st.number_input("Prolongued Decelerations", value=0.0)
abnormal_short_term_var = st.number_input("Abnormal Short Term Variability", value=0.0)
mean_short_term_var = st.number_input("Mean Short Term Variability", value=0.5)
percentage_long_term_var = st.number_input("Percentage of Long Term Variability", value=70.0)
mean_long_term_var = st.number_input("Mean Long Term Variability", value=2.0)

if st.button("Prediksi"):
    # Data input pengguna (11 fitur asli)
    input_data = np.array([[baseline_value, accelerations, fetal_movement,
                            uterine_contractions, light_decelerations,
                            severe_decelerations, prolongued_decelerations,
                            abnormal_short_term_var, mean_short_term_var,
                            percentage_long_term_var, mean_long_term_var]])

    # Tambah 10 kolom dummy agar total 21 fitur
    while input_data.shape[1] < 21:
        input_data = np.insert(input_data, -1, 0.0, axis=1)

    # Skala input
    input_scaled = scaler.transform(input_data).astype(np.float32)

    # Prediksi menggunakan model TFLite
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)

    # Ambil nilai label numerik
    predicted_label_num = label_encoder.inverse_transform([predicted_class])[0]

    # Mapping ke label teks
    label_dict = {
        1.0: "Normal",
        2.0: "Suspect",
        3.0: "Pathological"
    }
    predicted_label = label_dict.get(predicted_label_num, "Tidak diketahui")

    # Tampilkan hasil
    st.success(f"Prediksi kesehatan janin: **{predicted_label.upper()}**")

