import streamlit as st
import joblib
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load('speech_emotion_recognition_model.pkl')

def extract_features(y, sr):
    # Extract MFCCs, Chroma, and Mel features (the same number as during training)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60)  # 60 MFCCs
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)     # Add Chroma feature (12 features)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)      # Add Mel feature (e.g., 128 features)
    
    # Combine the features to match the training feature dimensions (200)
    result = np.hstack((np.mean(mfccs.T, axis=0), np.mean(chroma.T, axis=0), np.mean(mel.T, axis=0)))
    return result
    
st.set_page_config(page_title="Speech Emotion Recognition", page_icon=":musical_note:")

# Header and subheader
st.title("🎤 Speech Emotion Recognition")
st.subheader("Upload an audio file and predict its emotion.")

audio_file = st.file_uploader("Choose an audio file", type=["wav"])

if audio_file is not None:
    st.write(f"**Uploaded File:** {audio_file.name}")
    
    y, sr = librosa.load(audio_file, sr=None)
    feature = extract_features(y, sr)
    
    # Display waveform of the audio (optional for aesthetic appeal)
    st.subheader("Audio Waveform")
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, len(y)/sr, len(y)), y)
    ax.set(xlabel="Time (s)", ylabel="Amplitude", title="Waveform of Audio")
    st.pyplot(fig)
    
    # Add a button to trigger the prediction
    if st.button("Predict Emotion"):
        prediction = model.predict([feature])
        
        st.write(f"**Predicted Emotion:** {prediction[0]}")
        
        # Add some styling (colors, icons, etc.) to make the interface more interactive
        if prediction[0] == 'happy':
            st.markdown('<h2 style="color: #FF6347;">😊 Happy</h2>', unsafe_allow_html=True)
        elif prediction[0] == 'sad':
            st.markdown('<h2 style="color: #1E90FF;">😢 Sad</h2>', unsafe_allow_html=True)
        elif prediction[0] == 'angry':
            st.markdown('<h2 style="color: #DC143C;">😡 Angry</h2>', unsafe_allow_html=True)
        elif prediction[0] == 'neutral':
            st.markdown('<h2 style="color: #808080;">😐 Neutral</h2>', unsafe_allow_html=True)
        elif prediction[0] == 'fearful':
            st.markdown('<h2 style="color: #FF4500;">😨 Fearful</h2>', unsafe_allow_html=True)
        elif prediction[0] == 'disgust':
            st.markdown('<h2 style="color: #008000;">🤢 Disgust</h2>', unsafe_allow_html=True)
        elif prediction[0] == 'calm':
            st.markdown('<h2 style="color: #32CD32;">😌 Calm</h2>', unsafe_allow_html=True)
        else:
            st.markdown('<h2 style="color: #32CD32;">🤔 Unknown Emotion</h2>', unsafe_allow_html=True)

# footer
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f8d568;
        padding: 10px;
        text-align: center;
        font-family: 'Arial', sans-serif;
        color: #555;
        border-top: 1px solid #ccc;
    }
    .footer a {
        color: #4CAF50;
        font-weight: bold;
        text-decoration: none;
    }
    .footer p {
        font-size: 14px;
        margin: 5px 0;
    }
    </style>
    <div class="footer">
        <p>Built with ❤️ and a dash of creativity by <span style="font-weight:bold;">zen`</span></p>
        <p>Feel free to view the <a href="https://github.com/zen-0wl/Speech-Recognition-avec-librosa">source code</a>!</p>
    </div>
    """, unsafe_allow_html=True
)