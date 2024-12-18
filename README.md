# Speech Emotion Recognition

This project is a Speech Emotion Recognition system that classifies emotions from audio files using machine learning. The model is trained using various audio features, and predictions are made based on the input speech. The app is deployed on Streamlit for easy interaction and access.

## Features

- **Emotion Prediction**: The app can predict the emotion of a given audio file from several predefined emotions.
- **Audio File Input**: Users can upload their own audio files (in `.wav` format) for emotion analysis.
- **Real-Time Emotion Recognition**: After uploading an audio file, the app provides the predicted emotion in real-time.

## Emotions Recognized

The model can recognize the following emotions:

- 😊 Happy
- 😢 Sad
- 😡 Angry
- 😐 Neutral
- 😨 Fearful
- 🤢 Disgust
- 😌 Calm

## Installation

To run this project locally, follow the instructions below:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Set up a Virtual Environment (Optional but recommended)

You can create a virtual environment to manage dependencies:

```
python -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
```

### 3. Install the Dependencies

Make sure to install all required packages by using `requirements.txt`.

```
pip install -r requirements.txt
```

### 4. Run the App

Once all dependencies are installed, run the app locally with Streamlit:

```
streamlit run app.py
```

This will start a local server at `http://localhost:8501`.

## Deployment

The Speech Emotion Recognition model has been deployed on Streamlit Cloud, and you can access the live version of the app [here](https://speech-emotion-recogniser.streamlit.app/).

## How to Use the App

1. **Upload an Audio File** : Click the “Choose an audio file” button to upload a `.wav` audio file.
2. **Emotion Prediction** : After uploading the file, the app will process the audio and predict the emotion from the speech.
3. **View the Prediction** : The predicted emotion will be displayed in a styled format with an emoji representing the emotion.

## Model Used

The model used for emotion recognition is a machine learning model (MLPClassifier) trained on audio features extracted from the uploaded audio files. The features used include Mel-frequency cepstral coefficients (MFCCs), chroma features, and Mel spectrograms.

## Contributions

Feel free to contribute to the project by forking the repository, submitting pull requests, or reporting any issues. You can also improve the model by adding new features or training with different datasets.

### How to Contribute

1. Fork the repository.
2. Clone your forked repository.
3. Create a new branch for your changes.
4. Make your changes and commit them.
5. Push to your forked repository.
6. Submit a pull request.

## Acknowledgements

* [Librosa](https://librosa.org/): For audio feature extraction.
* [Scikit-learn](https://scikit-learn.org/): For machine learning algorithms.
* [Streamlit](https://streamlit.io/): For building the interactive app.
