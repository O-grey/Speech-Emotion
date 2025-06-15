import streamlit as st
import librosa
import numpy as np
import pickle
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.title("🔊 Emotion Recognition Web App")
st.write("Upload a WAV file and see its predicted emotion!(by Tushar Verma)")

# Load artifacts
scaler = joblib.load("scalerii.pkl")
le_gender = joblib.load("le_genderii.pkl")
le_intensity = joblib.load("le_intensityii.pkl")
le_emotion = joblib.load("le_emotionii.pkl")
feature_order = joblib.load("feature_order.pkl")
model = load_model("emotion_ann_modelii.h5")


def extract_metadata_from_filename(file_name):
    emotion_map = {
        "01": "Neutral", "02": "Calm", "03": "Happy", "04": "Sad",
        "05": "Angry", "06": "Fearful", "07": "Disgust", "08": "Surprised"
    }
    intensity_map = {"01": "Normal", "02": "Strong"}
    modality_map = {"01": "Audio-Video", "02": "Video-Only", "03": "Audio-Only"}

    parts = file_name.replace(".mp4", "").replace(".wav", "").split("-")
    if len(parts) != 7:
        return None

    modality_code = parts[0]
    emotion_code = parts[2]
    intensity_code = parts[3]
    actor_code = parts[6]

    modality = modality_map.get(modality_code)
    emotion = emotion_map.get(emotion_code)
    intensity = intensity_map.get(intensity_code)

    try:
        actor_id = int(actor_code)
        gender = "Male" if actor_id % 2 == 1 else "Female"
    except:
        gender = "Unknown"
        actor_id = None

    if not (modality and emotion and intensity):
        return None

    return {
        "filename": file_name,
        "emotion": emotion,
        "intensity": intensity,
        "modality": modality,
        "gender": gender,
        "actor_id": actor_id
    }


FRAME_SIZE = 1024
HOP_LENGTH = 512

def amplitude_envelope(signal, frame_size, hop_length):
    """Calculate the amplitude envelope of a signal with given frame size and hop length."""
    amplitude_envelope = []
    for i in range(0, len(signal), hop_length):
        current_frame = signal[i:i+frame_size]
        if len(current_frame) == 0:
            continue
        amplitude_envelope_current_frame = max(np.abs(current_frame))
        amplitude_envelope.append(amplitude_envelope_current_frame)
    return np.array(amplitude_envelope)

def rmse(signal, frame_size, hop_length):
    rmse = []

    # calculate rmse for each frame
    for i in range(0, len(signal), hop_length):
        rmse_current_frame = np.sqrt(sum(signal[i:i+frame_size]**2) / frame_size)
        rmse.append(rmse_current_frame)
    return np.array(rmse)

# ---------- Function to Compute 128 Mel Spectrogram Bands ----------
# ---------- Function to Compute 128 Mel Spectrogram Vector ----------
def compute_mel_spectrogram_vector_from_array(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    try:
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        mel_means = np.mean(log_mel_spectrogram, axis=1)  # shape: (128,)

        if mel_means.shape[0] != 128 or np.any(np.isnan(mel_means)):
            return [np.nan] * 128
        else:
            return mel_means

    except Exception as e:
        return [np.nan] * 128


    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return [np.nan] * 128
    
def extract_mfcc_means_from_array(y, sr, n_mfcc=40):
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)

        if not np.isnan(mfcc_mean).any():
            return mfcc_mean
        else:
            return [np.nan] * n_mfcc
    except Exception as e:
        return [np.nan] * n_mfcc
    
## Band energy ratio
def band_energy_ratio(spectrogram, split_frequency_bin=185):
    power_spectrogram = np.abs(spectrogram) ** 2
    power_spectrogram = power_spectrogram.T  # Shape: (frames, bins)

    ber_values = []
    for frame in power_spectrogram:
        low_freq_energy = frame[:split_frequency_bin].sum()
        high_freq_energy = frame[split_frequency_bin:].sum()
        ber = low_freq_energy / high_freq_energy if high_freq_energy != 0 else 0
        ber_values.append(ber)

    return np.array(ber_values)


def compute_ber_mean_from_array(y, sr, split_frequency_bin=185):
    try:
        spectrogram = librosa.stft(y, n_fft=2048, hop_length=512)
        ber = band_energy_ratio(spectrogram, split_frequency_bin)
        ber_mean = np.mean(ber)

        return ber_mean if not np.isnan(ber_mean) else np.nan
    except Exception:
        return np.nan

def extract_spectral_feature_means_from_array(y, sr, frame_size=1024, hop_length=512):
    try:
        sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]

        sc_mean = np.mean(sc)
        sb_mean = np.mean(sb)

        return sc_mean, sb_mean if not any(np.isnan([sc_mean, sb_mean])) else (np.nan, np.nan)
    except Exception:
        return np.nan, np.nan




uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    try:
        y, sr = librosa.load(uploaded_file, sr=22050)
        st.audio(uploaded_file, format='audio/wav')

        metadata = extract_metadata_from_filename(uploaded_file.name)
        if metadata:
            st.write("Extracted metadata:")
            st.json(metadata)

            # Convert metadata dict to a single-row DataFrame
            df = pd.DataFrame([metadata])
            
            # Now you can pass metadata_df to your feature extraction / preprocessing pipeline
            # For example, if your model expects features from this DataFrame:
            # features = extract_features(metadata_df)
            # X_scaled = scaler.transform(features)
            # prediction = model.predict(X_scaled)
                # Calculate amplitude envelope
            ae = amplitude_envelope(y, FRAME_SIZE, HOP_LENGTH)
            rms = rmse(y, FRAME_SIZE, HOP_LENGTH)
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

            # Extract 128 Mel spectrogram vector from raw signal
            mel_vector = compute_mel_spectrogram_vector_from_array(y, sr=sr)

            # Extract MFCC mean features
            mfcc_means = extract_mfcc_means_from_array(y, sr, n_mfcc=40)
                # Compute BER mean
            ber_mean = compute_ber_mean_from_array(y, sr, split_frequency_bin=185)
            # Spectral Centroid & Bandwidth
            spec_centroid_mean, spec_bandwidth_mean = extract_spectral_feature_means_from_array(y, sr)

            


 


    
        # Append mean of amplitude envelope to the 'AE' column
            df['AE_mean'] = [ae.mean()]
            df['RMSE_mean']=[rms.mean()]
            df['ZCR_mean']=[zcr.mean()]
                # Create column names and append to DataFrame
            mel_spec_cols = [f"Mel_{i}" for i in range(128)]
            for i, val in enumerate(mel_vector):
                df[mel_spec_cols[i]] = [val]
            mfcc_cols = [f"MFCC_{i+1}" for i in range(40)]
            for i, val in enumerate(mfcc_means):
                df[mfcc_cols[i]] = [val]
            df["BER_Mean"] = [ber_mean]
            df["Spec_Centroid_Mean"] = [spec_centroid_mean]
            df["Spec_Bandwidth_Mean"] = [spec_bandwidth_mean]
            
            st.write("Metadata DataFrame:")
            st.dataframe(df)

            # Load the same encoders and scaler you used in training
            # Assuming: le_dict contains label encoders for 'gender' and 'intensity'
            #           scaler is the trained StandardScaler


            # Assume df is your new inference DataFrame and has already had features extracted
                        # Your input DataFrame (after extracting all features)
            # # Drop unnecessary columns
            df_infer = df.drop(columns=['filename', 'actor_id', 'modality'])

            # # Encode categorical columns
            # df_infer['gender'] = le_gender.transform(df_infer['gender'])
            # df_infer['intensity'] = le_intensity.transform(df_infer['intensity'])

            # # Ensure exact same column order used during training
            # df_infer = df_infer[feature_order]  # Make sure feature_order includes gender and intensity

            # # Scale only float columns, exclude gender/intensity
            # float_cols = df_infer.select_dtypes(include=['float32', 'float64', 'int64']).columns.difference(['gender', 'intensity'])
            # df_infer_scaled = df_infer.copy()
            # df_infer_scaled[float_cols] = scaler.transform(df_infer[float_cols])

            # # ✅ Now use the fully reconstructed df_infer_scaled (with gender and intensity) for prediction
            # pred_probs = model.predict(df_infer_scaled)

            # # Decode labels
            # pred_labels = le_emotion.inverse_transform(np.argmax(pred_probs, axis=1))
            # st.success(f"Predicted Emotion: {pred_labels[0]}")
            # Load saved files


# Encode gender and intensity
            df_infer['gender'] = le_gender.transform(df_infer['gender'])
            df_infer['intensity'] = le_intensity.transform(df_infer['intensity'])

            # Ensure correct column order
            df_infer = df_infer[feature_order]

            # Scale float features (not gender/intensity)
            float_cols = df_infer.select_dtypes(include=['float32', 'float64', 'int64']).columns.difference(['gender', 'intensity'])
            df_infer[float_cols] = scaler.transform(df_infer[float_cols])

            # Predict
            pred_probs = model.predict(df_infer)
            pred_labels = le_emotion.inverse_transform(np.argmax(pred_probs, axis=1))
            st.success(f"Predicted Emotion: {pred_labels[0]}")
            







        else:
            st.error("Filename format not recognized or contains unknown codes.")

    except Exception as e:
        st.error(f"Error loading audio file: {e}")

