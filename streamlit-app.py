import streamlit as st
import requests
import time
import re
import json
import base64

st.set_page_config(
    page_title="AI Script Generator with Images and Audio",
    page_icon="📝",
    layout="wide"
)

# Set up the Replicate API
if 'REPLICATE_API_TOKEN' in st.secrets:
    REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
else:
    REPLICATE_API_TOKEN = st.sidebar.text_input("Enter your Replicate API token:", type="password")
    if not REPLICATE_API_TOKEN:
        st.warning("Please enter your Replicate API token to use this app.")
        st.stop()

def wait_for_prediction(prediction_id):
    """Poll for prediction results until ready"""
    try:
        while True:
            response = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers={
                    "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
                    "Content-Type": "application/json"
                }
            )
            
            # Check if the request was successful
            if response.status_code != 200:
                st.error(f"Error: API returned status code {response.status_code}")
                st.write("Response:", response.text)
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            prediction = response.json()
            
            if prediction["status"] == "succeeded":
                return prediction
            elif prediction["status"] == "failed":
                st.error(f"Prediction failed: {prediction.get('error', 'Unknown error')}")
                st.write("Full error response:", prediction)
                raise Exception(f"Prediction failed: {prediction.get('error', 'Unknown error')}")
            
            # Add debug info about status
            status_placeholder = st.empty()
            status_placeholder.write(f"Current status: {prediction['status']}")
            
            time.sleep(2)
    except Exception as e:
        st.error(f"Error in wait_for_prediction: {str(e)}")
        raise

def generate_script(theme, num_chapters, style_preference):
    """Generate script with Claude 3.7 Sonnet"""
    # Create prompt for Claude
    prompt = f"""
        Create a creative script about "{theme}" with exactly {num_chapters} chapters.
        Each chapter should be between 200-600 words.
        The script should have a clear beginning, middle, and end.
        Don't use any heading or chapter numbers in the text itself.
        Also, mark approximately every 30-60 words where an image would enhance the story with [IMAGE: brief image description].
        Format the output as a JSON object with the following structure:
        {{
            "title": "The title of the story",
            "chapters": [
                {{
                    "text": "The full text of chapter 1 with [IMAGE: descriptions] included",
                    "image_prompts": ["Description for image 1", "Description for image 2", ...]
                }},
                // More chapters...
            ]
        }}
    """
    
    # Call Claude 3.7 Sonnet via Replicate API
    response = requests.post(
        "https://api.replicate.com/v1/models/anthropic/claude-3.7-sonnet/predictions",
        headers={
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "input": {
                "prompt": prompt
            },
            "stream": False
        }
    )
    
    # Check if the request was successful
    if response.status_code != 200:
        st.error(f"Error: API returned status code {response.status_code}")
        st.write("Response:", response.text)
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    prediction = response.json()
    
    # Debug information
    st.write("API response:", prediction)
    
    # Check for errors in the response
    if "error" in prediction:
        raise Exception(f"Error creating prediction: {prediction['error']}")
    
    # Check if 'id' exists in the response
    if "id" not in prediction:
        st.error("API response does not contain an 'id' field")
        st.write("Full API response:", prediction)
        raise Exception("Invalid API response: missing 'id' field")
    
    # Wait for the prediction to complete
    prediction = wait_for_prediction(prediction["id"])
    
    # Parse the JSON result
    try:
        script_data = json.loads(prediction["output"])
        return script_data
    except:
        # Attempt to extract JSON if Claude didn't output pure JSON
        output = prediction["output"]
        match = re.search(r'({[\s\S]*})', output)
        if match:
            try:
                script_data = json.loads(match.group(1))
                return script_data
            except:
                raise Exception("Failed to parse script data from Claude's output")
        else:
            raise Exception("Could not extract JSON from Claude's output")

def generate_image(prompt, theme, style_preference):
    """Generate image with Flux Schnell"""
    full_prompt = f"{theme}, {prompt}, {style_preference} style, highly detailed"
    
    response = requests.post(
        "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
        headers={
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "input": {
                "prompt": full_prompt,
                "width": 768,
                "height": 768,
                "num_inference_steps": 4,
                "go_fast": True
            }
        }
    )
    
    # Check if the request was successful
    if response.status_code != 200:
        st.error(f"Error: API returned status code {response.status_code}")
        st.write("Response:", response.text)
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    prediction = response.json()
    
    # Check for errors in the response
    if "error" in prediction:
        raise Exception(f"Error creating prediction: {prediction['error']}")
    
    # Check if 'id' exists in the response
    if "id" not in prediction:
        st.error("API response does not contain an 'id' field")
        st.write("Full API response:", prediction)
        raise Exception("Invalid API response: missing 'id' field")
    
    # Wait for the prediction to complete
    prediction = wait_for_prediction(prediction["id"])
    
    return prediction["output"]

def generate_audio(text, voice):
    """Generate audio with Kokoro"""
    response = requests.post(
        "https://api.replicate.com/v1/models/jaaari/kokoro-82m/predictions",
        headers={
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "input": {
                "text": text,
                "voice": voice,
                "speed": 1.0
            }
        }
    )
    
    # Check if the request was successful
    if response.status_code != 200:
        st.error(f"Error: API returned status code {response.status_code}")
        st.write("Response:", response.text)
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    prediction = response.json()
    
    # Check for errors in the response
    if "error" in prediction:
        raise Exception(f"Error creating prediction: {prediction['error']}")
    
    # Check if 'id' exists in the response
    if "id" not in prediction:
        st.error("API response does not contain an 'id' field")
        st.write("Full API response:", prediction)
        raise Exception("Invalid API response: missing 'id' field")
    
    # Wait for the prediction to complete
    prediction = wait_for_prediction(prediction["id"])
    
    return prediction["output"]

# Streamlit UI
st.title("AI Script Generator with Images and Audio")

# Display debugging info
st.sidebar.header("Debug Information")
debug_expander = st.sidebar.expander("API Token Status")
with debug_expander:
    if REPLICATE_API_TOKEN:
        # Only show the first and last 4 characters
        token_preview = REPLICATE_API_TOKEN[:4] + "..." + REPLICATE_API_TOKEN[-4:]
        st.success(f"API Token is set: {token_preview}")
    else:
        st.error("API Token is not set")
    
    # Test API connection
    if st.button("Test API Connection"):
        try:
            response = requests.get(
                "https://api.replicate.com/v1/models",
                headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"}
            )
            if response.status_code == 200:
                st.success("Connection to Replicate API successful!")
            else:
                st.error(f"Connection failed with status code: {response.status_code}")
                st.text(response.text)
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

with st.form("input_form"):
    theme = st.text_input("Story Theme or Topic:", placeholder="e.g., Space adventure, Medieval fantasy, Underwater exploration")
    
    col1, col2 = st.columns(2)
    with col1:
        num_chapters = st.number_input("Number of Chapters:", min_value=1, max_value=10, value=3)
        style_preference = st.selectbox(
            "Image Style:",
            options=[
                "photorealistic", "fantasy art", "anime", "oil painting", 
                "digital art", "pencil sketch", "watercolor", "3d render"
            ],
            index=0
        )
    
    with col2:
        voice_preference = st.selectbox(
            "Narration Voice:",
            options=[
                "af_scarlett", "af_bella", "af_sky", "af_nova",
                "af_river", "af_zoe", "af_heart", "af_oren",
                "af_reed", "af_kai", "af_theo"
            ],
            format_func=lambda x: {
                "af_scarlett": "Scarlett (Female)",
                "af_bella": "Bella (Female)",
                "af_sky": "Sky (Female)",
                "af_nova": "Nova (Female)",
                "af_river": "River (Female)",
                "af_zoe": "Zoe (Female)",
                "af_heart": "Heart (Female)",
                "af_oren": "Oren (Male)",
                "af_reed": "Reed (Male)",
                "af_kai": "Kai (Male)",
                "af_theo": "Theo (Male)",
            }[x],
            index=0
        )
    
    submit_button = st.form_submit_button("Generate Story")

if submit_button and theme:
    try:
        with st.spinner("Generating your story..."):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Generate script
            status_text.text("Generating script...")
            
            # Debug mode to show exact API calls
            debug_container = st.container()
            with debug_container:
                st.write("Making API call to Claude 3.7 Sonnet...")
                
            try:
                script_data = generate_script(theme, num_chapters, style_preference)
                debug_container.success("Script generated successfully!")
            except Exception as e:
                debug_container.error(f"Script generation failed: {str(e)}")
                raise e
            
            progress_bar.progress(30)
            
            # Step 2: Generate images
            status_text.text("Creating images...")
            all_image_prompts = []
            for chapter in script_data["chapters"]:
                all_image_prompts.extend(chapter["image_prompts"])
            
            debug_container.write(f"Found {len(all_image_prompts)} image prompts")
            
            generated_images = []
            for i, prompt in enumerate(all_image_prompts):
                status_text.text(f"Creating image {i+1} of {len(all_image_prompts)}...")
                debug_container.write(f"Generating image for prompt: {prompt}")
                
                try:
                    image_url = generate_image(prompt, theme, style_preference)
                    generated_images.append({"url": image_url, "prompt": prompt})
                    debug_container.success(f"Image {i+1} generated!")
                except Exception as e:
                    debug_container.error(f"Image {i+1} generation failed: {str(e)}")
                    # Continue with other images instead of stopping completely
                    continue
                
                progress_bar.progress(30 + int((i+1) / len(all_image_prompts) * 50))
            
            # Step 3: Generate audio
            status_text.text("Generating audio narration...")
            full_text = script_data["title"] + "\n\n"
            for chapter in script_data["chapters"]:
                clean_text = re.sub(r'\[IMAGE:.*?\]', '', chapter["text"])
                full_text += clean_text + "\n\n"
            
            debug_container.write("Generating audio narration...")
            
            try:
                audio_url = generate_audio(full_text, voice_preference)
                debug_container.success("Audio generated successfully!")
            except Exception as e:
                debug_container.error(f"Audio generation failed: {str(e)}")
                audio_url = None  # Continue without audio rather than failing completely
            
            progress_bar.progress(100)
            status_text.text("Complete!")
            
            # Save results in session state
            st.session_state["script_data"] = script_data
            st.session_state["generated_images"] = generated_images
            st.session_state["audio_url"] = audio_url
            
            # Remove debug container after success
            debug_container.empty()
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your API token and try again.")

# Display results
if "script_data" in st.session_state:
    script_data = st.session_state["script_data"]
    generated_images = st.session_state["generated_images"]
    audio_url = st.session_state["audio_url"]
    
    st.header(script_data["title"])
    
    # Audio player
    st.subheader("Audio Narration")
    st.audio(audio_url)
    st.markdown(f"[Download Audio]({audio_url})")
    
    # Display chapters with images
    for chapter_idx, chapter in enumerate(script_data["chapters"]):
        st.subheader(f"Chapter {chapter_idx + 1}")
        
        # Split the text by image markers and insert images between text segments
        text_parts = re.split(r'\[IMAGE:.*?\]', chapter["text"])
        image_count = len(text_parts) - 1  # One fewer images than text parts
        
        for i, text_part in enumerate(text_parts):
            st.write(text_part)
            
            # If there's an image after this text segment, display it
            if i < image_count and i < len(generated_images):
                image_idx = sum(len(script_data["chapters"][j]["image_prompts"]) for j in range(chapter_idx)) + i
                if image_idx < len(generated_images):
                    st.image(generated_images[image_idx]["url"], 
                             caption=generated_images[image_idx]["prompt"],
                             use_column_width=True)
    
    # Button to create another story
    if st.button("Create Another Story"):
        # Clear the session state
        for key in ["script_data", "generated_images", "audio_url"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

# Add a sidebar with information
with st.sidebar:
    st.header("About this App")
    st.write("""
    This app uses three AI models from Replicate:
    
    1. **Claude 3.7 Sonnet** - Creates the story script
    2. **Flux Schnell** - Generates images for the story
    3. **Kokoro-82M** - Converts text to speech for narration
    
    The app requires a Replicate API token to function.
    """)
    
    st.header("Deployment")
    st.write("""
    To deploy your own version of this app:
    
    1. Create a [Streamlit Cloud](https://streamlit.io/cloud) account
    2. Connect your GitHub repository
    3. Add your Replicate API token as a secret
    
    The app will then be available online with its own URL.
    """)
