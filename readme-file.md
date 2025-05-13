# AI Script Generator with Images and Audio

A Streamlit web application that generates creative stories with AI-generated images and audio narration using Replicate's APIs.

## Features

- Creates original stories based on your chosen theme
- Generates matching images for key moments in the story
- Produces audio narration of the complete story
- Allows customization of story length, image style, and narration voice

## How It Works

This application uses three powerful AI models from Replicate:

1. **Claude 3.7 Sonnet** - Generates the story script with image markers
2. **Flux Schnell** - Creates images for the marked scenes in the story
3. **Kokoro-82M** - Converts the story text to natural-sounding speech

## Requirements

- Python 3.8 or higher
- A Replicate API token (get one at [replicate.com](https://replicate.com))

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ai-script-generator.git
   cd ai-script-generator
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## Deployment

To deploy this app on Streamlit Cloud:

1. Fork this repository to your GitHub account
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your forked repository
4. In the app settings, add your Replicate API token as a secret with the name `REPLICATE_API_TOKEN`
5. Deploy the app

## License

MIT

## Credits

- Claude by Anthropic
- Flux Schnell by Black Forest Labs
- Kokoro by jaaari