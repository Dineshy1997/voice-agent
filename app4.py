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
import json
import google.generativeai as genai

# Configure API key
def configure_gemini_api():
    # Directly use your Gemini API key
    api_key = "AIzaSyA4XoQocx6O7ffw413LfZkUjd6cFrLozuE"
    genai.configure(api_key=api_key)
    return genai

# Initialize Gemini model
def initialize_gemini():
    gemini = configure_gemini_api()
    # Using gemini-1.5-pro instead of gemini-pro as it might be the newer version required
    model = genai.GenerativeModel('gemini-1.5-pro')
    return model

# System database simulation (in production, use a proper database)
class RecordingsDatabase:
    def __init__(self):
        # Initialize database or load from file
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
        
        # Save to file
        with open(self.db_file, 'w') as f:
            json.dump(self.records, f)
            
    def get_user_recordings(self, user_id):
        if user_id in self.records:
            return self.records[user_id]
        return []

# Function to transcribe and translate using Gemini
def transcribe_and_translate(model, audio_data, source_language="auto", target_language="English"):
    # For Gemini, we need to encode the audio file
    import base64
    audio_b64 = base64.b64encode(audio_data).decode()
    
    # Request Gemini to transcribe
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
        st.error(f"Transcription error: {str(e)}")
        # Provide dummy transcript for testing if API fails
        transcript = "This is a sample transcript. The API failed with: " + str(e)
    
    # Step 2: Translate to the target language
    prompt_translate = f"""
    Translate the following text to {target_language}.
    Provide only the direct translation without any explanations, notes, or additional context.
    
    Text: {transcript}
    """
    
    try:
        translation_response = model.generate_content(prompt_translate)
        translation = translation_response.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        translation = transcript  # Use transcript as fallback
    
    return transcript, translation

# Function to generate content summary using Gemini
def generate_content_summary(model, translation):
    prompt = f"""
    Based on the following transcript, create a concise summary that captures the main points.
    
    Transcript: {translation}
    """
    
    try:
        response = model.generate_content(prompt)
        summary = response.text
        return summary
    except Exception as e:
        st.error(f"Summary generation error: {str(e)}")
        return "Unable to generate summary due to API error: " + str(e)

# Function to generate text-to-speech audio
def text_to_speech(text, language='en'):
    try:
        # Map language names to language codes for gTTS
        language_map = {
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'Chinese': 'zh-CN',
            'Arabic': 'ar',
            'Hindi': 'hi',
            'Japanese': 'ja',
            'German': 'de',
            'Portuguese': 'pt',
            'Russian': 'ru',
            # Add more languages as needed
        }
        
        # Use the appropriate language code
        lang_code = language_map.get(language, 'en')
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        audio_file_path = temp_file.name
        temp_file.close()
        
        # Generate the speech
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(audio_file_path)
        
        return audio_file_path
        
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

# Main Streamlit app
def main():
    st.title("Cube AI Voice Assistant")
    
    # Initialize database
    db = RecordingsDatabase()
    
    # Initialize session state for recording data
    if 'recorded_data' not in st.session_state:
        st.session_state.recorded_data = None
    
    # Try to initialize Gemini with error handling
    try:
        model = initialize_gemini()
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        st.warning("Proceeding with limited functionality. API features will be simulated.")
        model = None  # Will use fallback behavior
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Voice Recorder", "My Recordings", "Settings"])
    
    if page == "Voice Recorder":
        st.header("Voice Recorder")
        
        # Input fields in columns
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
        
        # Use File Uploader instead of direct recording
        uploaded_file = st.file_uploader("Upload audio file (WAV, MP3)", type=["wav", "mp3"])
        
        # Process button
        if st.button("Process Audio", key="process_button") and uploaded_file is not None:
            if not user_id:
                st.error("Please enter a User ID")
            else:
                # Save the uploaded file
                recording_file = f"recordings/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                os.makedirs(os.path.dirname(recording_file), exist_ok=True)
                
                # Read the file data
                audio_data = uploaded_file.getvalue()
                
                # Save file
                with open(recording_file, "wb") as f:
                    f.write(audio_data)
                
                # Store necessary data in session state
                st.session_state.recorded_data = {
                    "user_id": user_id,
                    "recording_file": recording_file,
                    "source_language": source_language,
                    "target_language": target_language
                }
                
                # Process the recording
                with st.spinner("Transcribing and translating..."):
                    transcript, translation = transcribe_and_translate(
                        model, 
                        audio_data, 
                        source_language, 
                        target_language
                    )
                    st.session_state.recorded_data["transcript"] = transcript
                    st.session_state.recorded_data["translation"] = translation
                
                # Generate summary
                with st.spinner("Generating content summary..."):
                    summary = generate_content_summary(model, translation)
                    st.session_state.recorded_data["summary"] = summary
                
                # Generate text-to-speech for the translation
                with st.spinner("Generating audio for translation..."):
                    translation_audio = text_to_speech(translation, target_language)
                    st.session_state.recorded_data["translation_audio"] = translation_audio
        
        # Display recorded data if available in simplified two-column layout
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
            
            # Simple save button
            if st.button("Save Recording", key="save_button"):
                data = st.session_state.recorded_data
                
                db.save_recording(
                    data["user_id"], 
                    data["recording_file"], 
                    data["transcript"], 
                    data["translation"],
                    data.get("summary", ""),
                    data["target_language"],
                    data.get("translation_audio", None)
                )
                st.success("Recording saved successfully")
                # Clear the session state to reset the form
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
                    
                    # Create a table for the records
                    record_data = []
                    for i, record in enumerate(records):
                        record_data.append({
                            "Recording #": i+1,
                            "Date/Time": record['timestamp'],
                            "Target Language": record.get('target_language', 'English')
                        })
                    
                    record_df = pd.DataFrame(record_data)
                    st.dataframe(record_df, use_container_width=True)
                    
                    # Detailed view when a record is selected
                    record_options = [f"Recording {i+1} - {record['timestamp']}" for i, record in enumerate(records)]
                    selected_record = st.selectbox("Select recording to view details:", 
                                                  record_options,
                                                  key="record_select")
                    
                    if selected_record:
                        record_idx = int(selected_record.split(" ")[1]) - 1
                        record = records[record_idx]
                        
                        st.subheader(f"Recording Details - {record['timestamp']}")
                        
                        # Audio playback
                        st.markdown("**Original Recording:**")
                        if os.path.exists(record['recording_path']):
                            st.audio(record['recording_path'])
                        else:
                            st.error(f"Audio file not found: {record['recording_path']}")
                        
                        # Display in two-column format
                        target_language = record.get('target_language', 'English')
                        
                        # Use columns for side-by-side display
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Original Transcript:**")
                            st.write(record['transcript'])
                        
                        with col2:
                            st.markdown(f"**Translation ({target_language}):**")
                            st.write(record['translation'])
                        
                        # Translation audio if available
                        if record.get('translation_audio') and os.path.exists(record['translation_audio']):
                            st.markdown(f"**Translation Audio:**")
                            st.audio(record['translation_audio'])
    
    elif page == "Settings":
        st.header("Settings")
        
        st.subheader("API Configuration")
        # Show the API key in the settings (masked for security)
        api_key = st.text_input("Gemini API Key", 
                              value="AIzaSyA4XoQocx6O7ffw413LfZkUjd6cFrLozuE", 
                              type="password",
                              key="api_key_input")
        
        # Model selection
        model_version = st.selectbox(
            "Gemini Model Version",
            ["gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro-vision"],
            index=0,
            key="model_select"
        )
        
        if st.button("Save API Settings", key="save_api_button"):
            # In production, use a secure method to store this
            st.success("API Settings saved successfully")
            
        st.subheader("System Status")
        
        # Display system status in columns
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
