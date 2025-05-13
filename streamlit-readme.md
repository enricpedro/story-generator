# AI Script Generator with Images and Audio

This tool creates interactive stories with AI-generated images and audio narration using Replicate's APIs.

## Deployment

### Prerequisites
- Streamlit account (free at [streamlit.io](https://streamlit.io))
- GitHub account
- Replicate API key

## How to Deploy

### Step 1: Fork this Repository
- Click the Fork button at the top right of the repository page
- This creates your own copy of the code

### Step 2: Create a Streamlit Cloud Account
- Go to [streamlit.io](https://streamlit.io) and sign up (it's free)
- Connect your GitHub account

### Step 3: Deploy the App
- In Streamlit Cloud, click "New app"
- Select your repository, branch, and main file path (`streamlit_app.py`)
- In Advanced Settings, add your Replicate API token as a secret:
  - Name: `REPLICATE_API_TOKEN`
  - Value: `your_replicate_api_token_here`
- Click "Deploy"

Your app will now be available at a unique URL provided by Streamlit.

## Using the App

1. Enter a story theme or topic
2. Choose the number of chapters
3. Select an image style
4. Choose a narration voice
5. Click "Generate Story"
6. Enjoy your AI-generated story with images and audio narration!

## Credits

This app uses three AI models from Replicate:
- Claude 3.7 Sonnet by Anthropic for story generation
- Flux Schnell by Black Forest Labs for image generation
- Kokoro-82M by jaaari for text-to-speech narration

## License

MIT