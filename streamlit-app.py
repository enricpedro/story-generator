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
            
            # Show current status for debugging
            status_placeholder = st.empty()
            if "status" in prediction:
                status_placeholder.write(f"Current status: {prediction['status']}")
            else:
                status_placeholder.write(f"Current status: Unknown")
                st.write("Full response:", prediction)
            
            # Check prediction status
            if "status" in prediction:
                if prediction["status"] == "succeeded":
                    return prediction
                elif prediction["status"] == "failed":
                    st.error(f"Prediction failed: {prediction.get('error', 'Unknown error')}")
                    st.write("Full error response:", prediction)
                    raise Exception(f"Prediction failed: {prediction.get('error', 'Unknown error')}")
                elif prediction["status"] == "canceled":
                    st.error("Prediction was canceled")
                    st.write("Full response:", prediction)
                    raise Exception("Prediction was canceled")
            else:
                st.error("No status field in prediction response")
                st.write("Full response:", prediction)
                raise Exception("Invalid API response format - missing status field")
            
            # Wait before polling again
            time.sleep(2)
    except Exception as e:
        st.error(f"Error in wait_for_prediction: {str(e)}")
        raise

def handle_streaming_response(response, prediction_id):
    """Handle a streaming response from Replicate API"""
    try:
        # If the response contains a URL for streaming, fetch the content from that URL
        if "stream" in response and response["stream"]:
            stream_url = response["stream"]
            st.write(f"Stream URL: {stream_url}")
            
            headers = {
                "Accept": "text/event-stream",
                "Cache-Control": "no-store",
                "Authorization": f"Bearer {REPLICATE_API_TOKEN}"
            }
            
            stream_response = requests.get(stream_url, headers=headers, stream=True)
            
            # Check if the streaming request was successful
            if stream_response.status_code != 200:
                st.error(f"Error: Stream API returned status code {stream_response.status_code}")
                st.write("Response:", stream_response.text)
                return None
            
            full_text = ""
            for line in stream_response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        data = decoded_line[5:].strip()
                        if data != "[DONE]":
                            try:
                                json_data = json.loads(data)
                                if "output" in json_data:
                                    current_text = json_data["output"]
                                    full_text += current_text
                                    # Display the streaming output
                                    st.write(current_text)
                            except json.JSONDecodeError:
                                st.error(f"Failed to parse JSON: {data}")
                                continue
            
            return {"output": full_text}
        
        # Otherwise, wait for the prediction to complete using regular polling
        return wait_for_prediction(prediction_id)
        
    except Exception as e:
        st.error(f"Error in handle_streaming_response: {str(e)}")
        # Fall back to regular polling
        return wait_for_prediction(prediction_id)

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
            "stream": True
        }
    )
    
    # Check if the request was created successfully (201 is the expected response code)
    if response.status_code not in [200, 201]:
        st.error(f"Error: API returned status code {response.status_code}")
        st.write("Response:", response.text)
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    prediction = response.json()
    
    # Debug information
    st.write("Initial API response:", prediction)
    
    # Check for errors in the response
    if "error" in prediction and prediction["error"]:
        raise Exception(f"Error creating prediction: {prediction['error']}")
    
    # Check if 'id' exists in the response
    if "id" not in prediction:
        st.error("API response does not contain an 'id' field")
        st.write("Full API response:", prediction)
        raise Exception("Invalid API response: missing 'id' field")
    
    # Handle streaming response or wait for prediction to complete
    prediction_output = handle_streaming_response(prediction, prediction["id"])
    
    # Parse the JSON result from the output field
    try:
        if prediction_output and "output" in prediction_output:
            if isinstance(prediction_output["output"], str):
                # Try to parse string as JSON
                try:
                    script_data = json.loads(prediction_output["output"])
                    return script_data
                except json.JSONDecodeError:
                    # If not valid JSON, look for a JSON object in the string
                    match = re.search(r'({[\s\S]*})', prediction_output["output"])
                    if match:
                        script_data = json.loads(match.group(1))
                        return script_data
                    else:
                        st.error("Could not extract JSON from output")
                        st.write("Raw output:", prediction_output["output"])
                        raise Exception("Failed to parse JSON from output")
            else:
                # Direct JSON object
                return prediction_output["output"]
        else:
            st.error("No output field in prediction result")
            st.write("Full prediction:", prediction_output)
            raise Exception("No output field in prediction result")
    except Exception as e:
        st.error(f"Error parsing script data: {str(e)}")
        st.write("Raw output:", prediction_output.get("output", "No output available"))
        raise Exception(f"Failed to parse script data: {str(e)}")

def generate_image(prompt, theme, style_preference):
    """Generate image with Flux Schnell"""
    full_prompt = f"{theme}, {prompt}, {style_preference} style, highly detailed"
    
    response = requests.post(
        "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
        headers={
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
            "Prefer": "wait"
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
    
    # Check if the request was created successfully (201 is the expected response code)
    if response.status_code not in [200, 201]:
        st.error(f"Error: API returned status code {response.status_code}")
        st.write("Response:", response.text)
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    prediction = response.json()
    
    # Check for errors in the response
    if "error" in prediction and prediction["error"]:
        raise Exception(f"Error creating prediction: {prediction['error']}")
    
    # Check if 'id' exists in the response
    if "id" not in prediction:
        st.error("API response does not contain an 'id' field")
        st.write("Full API response:", prediction)
        raise Exception("Invalid API response: missing 'id' field")
    
    # Wait for the prediction to complete
    prediction = wait_for_prediction(prediction["id"])
    
    # Extract the output URL
    if "output" in prediction:
        if isinstance(prediction["output"], list) and len(prediction["output"]) > 0:
            return prediction["output"][0]  # First image from the list
        else:
            return prediction["output"]  # Direct output URL
    else:
        st.error("No output URL in prediction result")
        st.write("Full prediction:", prediction)
        raise Exception("No output URL in prediction result")

def generate_audio(text, voice):
    """Generate audio with Kokoro"""
    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers={
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
            "Prefer": "wait"
        },
        json={
            "version": "jaaari/kokoro-82m:f559560eb822dc509045f3921a192123491b9173d4bf3daab2169b71c7a13",
            "input": {
                "text": text,
                "voice": voice,
                "speed": 1.0
            }
        }
    )
    
    # Check if the request was created successfully (201 is the expected response code)
    if response.status_code not in [200, 201]:
        st.error(f"Error: API returned status code {response.status_code}")
        st.write("Response:", response.text)
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    prediction = response.json()
    
    # Check for errors in the response
    if "error" in prediction and prediction["error"]:
        raise Exception(f"Error creating prediction: {prediction['error']}")
    
    # Check if 'id' exists in the response
    if "id" not in prediction:
        st.error("API response does not contain an 'id' field")
        st.write("Full API response:", prediction)
        raise Exception("Invalid API response: missing 'id' field")
    
    # Wait for the prediction to complete
    prediction = wait_for_prediction(prediction["id"])
    
    # Extract the output URL
    if "output" in prediction:
        return prediction["output"]
    else:
        st.error("No output URL in prediction result")
        st.write("Full prediction:", prediction)
        raise Exception("No output URL in prediction result")

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
                "af_reed", "af_kai", "af_theo", "af_nicole"
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
                "af_nicole": "Nicole (Female)"
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
    if audio_url:
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
