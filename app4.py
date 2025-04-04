import streamlit as st
import os
import time
import numpy as np
from scipy.io.wavfile import write
from datetime import datetime
import pandas as pd
from gtts import gTTS
import tempfile
from pathlib import Path
import pyaudio
import wave
import json
import google.generativeai as genai

# List of API keys to rotate through
API_KEYS = [
    "AIzaSyA4XoQocx6O7ffw413LfZkUjd6cFrLozuE",
    "AIzaSyAs3xtBgM9jogT_oxKjPPupi96YY4RtyGo",  # Replace with your second API key
    "AIzaSyBtqE7KUn0hwunNHMhxqTDpd074e3n5DxQ",  # Replace with your third API key
    "AIzaSyBmlRXXXCLIQMPfWel_xVbV_qpSHd1tED8",  # Replace with your fourth API key
    "AIzaSyBSpZFh5zw5muWJUAWPJb404OljN1eTTnU",  # Replace with your fifth API key
    "AIzaSyCyf9I53pIjv2vokunrGrB_zbMCnGCopnI",  # Replace with your sixth API key
]

# Global variable to track the current API key index
if 'current_api_key_index' not in st.session_state:
    st.session_state.current_api_key_index = 0

# Configure API key with rotation logic
def configure_gemini_api():
    global API_KEYS
    api_key = API_KEYS[st.session_state.current_api_key_index]
    try:
        genai.configure(api_key=api_key)
        return genai
    except Exception as e:
        st.error(f"Failed to configure API key {api_key}: {str(e)}")
        # Move to the next API key if there's an issue
        switch_to_next_api_key()
        return configure_gemini_api()  # Recursive call with next key

# Switch to the next API key in the list
def switch_to_next_api_key():
    st.session_state.current_api_key_index = (st.session_state.current_api_key_index + 1) % len(API_KEYS)
    st.warning(f"Switched to API key index {st.session_state.current_api_key_index}")

# Initialize Gemini model with retry logic
def initialize_gemini():
    max_attempts = len(API_KEYS)
    attempts = 0
    
    while attempts < max_attempts:
        try:
            gemini = configure_gemini_api()
            model = gemini.GenerativeModel('gemini-1.5-pro')
            return model
        except Exception as e:
            st.error(f"Failed to initialize Gemini with API key {st.session_state.current_api_key_index}: {str(e)}")
            switch_to_next_api_key()
            attempts += 1
    
    st.error("All API keys failed to initialize Gemini. Proceeding with limited functionality.")
    return None  # Return None if all keys fail

# System database simulation (unchanged)
class RecordingsDatabase:
    def __init__(self):
        self.db_file = "voice_recordings.json"
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r') as f:
                self.records = json.load(f)
        else:
            self.records = {}
    
    def save_recording(self, user_id, recording_path, transcript, translation, summary="", target_language="English", translation_audio=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if user_id not in self.records:
            self.records[user_id] = []
            
        record = {
            "timestamp": timestamp,
            "recording_path": recording_path,
            "transcript": transcript,
            "translation": translation,
            "summary": summary,
            "target_language": target_language,
            "translation_audio": translation_audio
        }
        
        self.records[user_id].append(record)
        
        with open(self.db_file, 'w') as f:
            json.dump(self.records, f)
            
    def get_user_recordings(self, user_id):
        return self.records.get(user_id, [])

# Record audio function (unchanged)
def record_audio(duration=10, sample_rate=44100):
    st.write("🔴 Recording...")
    progress_bar = st.progress(0)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
    frames = []
    
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
        progress_bar.progress((i + 1) / int(sample_rate / 1024 * duration))
        time.sleep(0.001)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    filename = f"recordings/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    st.success(f"Recording saved: {filename}")
    return filename

# Transcription and translation with retry logic
def transcribe_and_translate(model, audio_file, source_language="auto", target_language="English"):
    with open(audio_file, "rb") as f:
        audio_data = f.read()
    
    import base64
    audio_b64 = base64.b64encode(audio_data).decode()
    
    prompt_transcribe = f"""
    Please transcribe this audio file accurately.
    The language may be {source_language} or could be any language if auto-detected.
    Return the transcription in the original language, exactly as spoken.
    """
    
    try:
        response = model.generate_content([
            prompt_transcribe,
            {"mime_type": "audio/wav", "data": audio_b64}
        ])
        transcript = response.text
    except Exception as e:
        st.error(f"Transcription failed with API key {st.session_state.current_api_key_index}: {str(e)}")
        switch_to_next_api_key()
        model = initialize_gemini()  # Re-initialize with the next key
        if model:
            return transcribe_and_translate(model, audio_file, source_language, target_language)
        transcript = "Transcription failed after all API retries."

    prompt_translate = f"""
    Translate the following text to {target_language}.
    Provide only the direct translation without any explanations, notes, or additional context.
    
    Text: {transcript}
    """
    
    try:
        translation_response = model.generate_content(prompt_translate)
        translation = translation_response.text
    except Exception as e:
        st.error(f"Translation failed with API key {st.session_state.current_api_key_index}: {str(e)}")
        switch_to_next_api_key()
        model = initialize_gemini()
        if model:
            return transcribe_and_translate(model, audio_file, source_language, target_language)
        translation = transcript  # Fallback to transcript
    
    return transcript, translation

# Generate content summary (unchanged except for retry logic)
def generate_content_summary(model, translation):
    prompt = f"""
    Based on the following transcript, create a concise summary that captures the main points.
    
    Transcript: {translation}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Summary generation failed with API key {st.session_state.current_api_key_index}: {str(e)}")
        switch_to_next_api_key()
        model = initialize_gemini()
        if model:
            return generate_content_summary(model, translation)
        return "Unable to generate summary due to API error."

# Text-to-speech function (unchanged)
def text_to_speech(text, language='en'):
    language_map = {
        'English': 'en', 'Spanish': 'es', 'French': 'fr', 'Chinese': 'zh-CN',
        'Arabic': 'ar', 'Hindi': 'hi', 'Japanese': 'ja', 'German': 'de',
        'Portuguese': 'pt', 'Russian': 'ru'
    }
    lang_code = language_map.get(language, 'en')
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    audio_file_path = temp_file.name
    temp_file.close()
    
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(audio_file_path)
        return audio_file_path
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

# Main Streamlit app
def main():
    st.title("Cube AI Voice Assistant")
    
    db = RecordingsDatabase()
    if 'recorded_data' not in st.session_state:
        st.session_state.recorded_data = None
    
    model = initialize_gemini()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Voice Recorder", "My Recordings", "Settings"])
    
    if page == "Voice Recorder":
        st.header("Voice Recorder")
        
        col1, col2 = st.columns(2)
        with col1:
            user_id = st.text_input("User ID", key="user_id_input")
        with col2:
            source_language = st.selectbox("Source Language (if known)", 
                                       ["auto", "English", "Spanish", "French", "Chinese", "Arabic", 
                                        "Hindi", "Japanese", "German", "Portuguese", "Russian"],
                                       key="source_language_select")
        
        col3, col4 = st.columns(2)
        with col3:
            target_language = st.selectbox("Target Language", 
                                       ["English", "Spanish", "French", "Chinese", "Arabic", 
                                        "Hindi", "Japanese", "German", "Portuguese", "Russian"],
                                       key="target_language_select")
        with col4:
            duration = st.slider("Recording Duration (seconds)", min_value=5, max_value=60, value=10, key="duration_slider")
        
        if st.button("Start Recording", key="record_button"):
            if not user_id:
                st.error("Please enter a User ID")
            else:
                recording_file = record_audio(duration=duration)
                st.session_state.recorded_data = {
                    "user_id": user_id,
                    "recording_file": recording_file,
                    "source_language": source_language,
                    "target_language": target_language
                }
                
                with st.spinner("Transcribing and translating..."):
                    transcript, translation = transcribe_and_translate(
                        model, recording_file, source_language, target_language
                    )
                    st.session_state.recorded_data["transcript"] = transcript
                    st.session_state.recorded_data["translation"] = translation
                
                with st.spinner("Generating content summary..."):
                    summary = generate_content_summary(model, translation)
                    st.session_state.recorded_data["summary"] = summary
                
                with st.spinner("Generating audio for translation..."):
                    translation_audio = text_to_speech(translation, target_language)
                    st.session_state.recorded_data["translation_audio"] = translation_audio
        
        if st.session_state.get('recorded_data'):
            st.subheader("Translation Results")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original:**")
                st.write(st.session_state.recorded_data["transcript"])
                st.audio(st.session_state.recorded_data["recording_file"])
            with col2:
                st.markdown("**Translation:**")
                st.write(st.session_state.recorded_data["translation"])
                if st.session_state.recorded_data.get("translation_audio"):
                    st.audio(st.session_state.recorded_data["translation_audio"])
            
            if st.button("Save Recording", key="save_button"):
                data = st.session_state.recorded_data
                db.save_recording(
                    data["user_id"], data["recording_file"], data["transcript"], 
                    data["translation"], data.get("summary", ""), 
                    data["target_language"], data.get("translation_audio", None)
                )
                st.success("Recording saved successfully")
                st.session_state.recorded_data = None
    
    elif page == "My Recordings":
        st.header("My Recordings")
        user_id = st.text_input("Enter User ID to view recordings", key="user_search_input")
        
        if st.button("Search", key="search_button"):
            if not user_id:
                st.error("Please enter a User ID")
            else:
                records = db.get_user_recordings(user_id)
                if not records:
                    st.warning(f"No recordings found for User ID: {user_id}")
                else:
                    st.success(f"Found {len(records)} recordings for User ID: {user_id}")
                    record_data = [{"Recording #": i+1, "Date/Time": r['timestamp'], "Target Language": r.get('target_language', 'English')} for i, r in enumerate(records)]
                    st.dataframe(pd.DataFrame(record_data), use_container_width=True)
                    
                    record_options = [f"Recording {i+1} - {r['timestamp']}" for i, r in enumerate(records)]
                    selected_record = st.selectbox("Select recording to view details:", record_options, key="record_select")
                    
                    if selected_record:
                        record_idx = int(selected_record.split(" ")[1]) - 1
                        record = records[record_idx]
                        st.subheader(f"Recording Details - {record['timestamp']}")
                        st.markdown("**Original Recording:**")
                        if os.path.exists(record['recording_path']):
                            st.audio(record['recording_path'])
                        else:
                            st.error(f"Audio file not found: {record['recording_path']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Transcript:**")
                            st.write(record['transcript'])
                        with col2:
                            st.markdown(f"**Translation ({record.get('target_language', 'English')}):**")
                            st.write(record['translation'])
                        if record.get('translation_audio') and os.path.exists(record['translation_audio']):
                            st.markdown("**Translation Audio:**")
                            st.audio(record['translation_audio'])
    
    elif page == "Settings":
        st.header("Settings")
        st.subheader("API Configuration")
        st.write(f"Current API Key Index: {st.session_state.current_api_key_index}")
        st.write(f"Active API Key: {API_KEYS[st.session_state.current_api_key_index][:10]}... (masked)")
        
        if st.button("Rotate to Next API Key", key="rotate_api_button"):
            switch_to_next_api_key()
            st.success(f"Switched to API key index {st.session_state.current_api_key_index}")
        
        model_version = st.selectbox("Gemini Model Version", ["gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro-vision"], index=0, key="model_select")
        
        st.subheader("System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Database Status: Connected")
            api_status = "Connected" if model else "Error"
            st.write(f"API Status: {api_status}")
        with col2:
            st.write("Storage: Available")
            recording_count = sum(len(records) for records in db.records.values())
            st.write(f"Total Recordings: {recording_count}")
            st.write(f"Users in Database: {len(db.records)}")

if __name__ == "__main__":
    main()
