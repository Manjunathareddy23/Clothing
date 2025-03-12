import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import base64
from io import BytesIO
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def get_image_base64(image_path):
    """Convert image to base64 encoding for API request."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def overlay_cloth_on_user(user_image, cloth_image):
    """Overlay the selected cloth image onto the user's photo."""
    cloth_resized = cv2.resize(cloth_image, (user_image.shape[1], user_image.shape[0]))
    for c in range(3):
        user_image[:, :, c] = (1 - cloth_resized[:, :, c] / 255) * user_image[:, :, c] + (cloth_resized[:, :, c] / 255) * cloth_resized[:, :, c]
    return user_image

def api_clothing_tryon(user_image, cloth_image):
    """Send images to an external API for virtual try-on."""
    
    # Get the API URL and Key from environment variables
    url = os.getenv("GEMINI_API_URL")  # Load the API URL from environment
    api_key = os.getenv("GEMINI_API_KEY")  # Load the API key from environment

    if not url or not api_key:
        st.error("API URL or API Key not set in environment variables.")
        return None

    # Convert images to base64 encoding
    user_image_base64 = get_image_base64(user_image)
    cloth_image_base64 = get_image_base64(cloth_image)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "user_image": user_image_base64,
        "cloth_image": cloth_image_base64
    }

    # Send the request to the API
    response = requests.post(url, json=data, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Process the response (assuming the API sends back an image or URL to the modified image)
        result_image_data = response.json().get("modified_image")
        modified_image = base64.b64decode(result_image_data)
        image = Image.open(BytesIO(modified_image))
        return image
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None

st.title("Virtual Clothing Try-On")

# Upload user's photo
user_image_file = st.file_uploader("Upload Your Photo", type=["jpg", "png", "jpeg"])

# Allow the user to upload the clothing image
cloth_file = st.file_uploader("Upload Clothing Image", type=["jpg", "png", "jpeg"])

if user_image_file and cloth_file:
    # Read the user's photo as an image
    user_image = np.array(Image.open(user_image_file))

    # Read the clothing image
    selected_cloth = cv2.imdecode(np.frombuffer(cloth_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    st.write("Processing...")

    # If using an external API for try-on (Gemini or similar), uncomment below
    # final_image = api_clothing_tryon(user_image, selected_cloth)

    # For local try-on without the API, fallback to overlay_cloth_on_user
    final_image = overlay_cloth_on_user(user_image, selected_cloth)

    # Convert the final result from OpenCV (BGR) to PIL (RGB) for Streamlit display
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    final_image_pil = Image.fromarray(final_image_rgb)

    # Display the final image with the clothing overlay
    st.image(final_image_pil, caption="Result", use_column_width=True)

    # Provide an option to download the final image
    final_image_bytes = final_image_pil.tobytes()
    st.download_button("Download Image", data=final_image_bytes, file_name="virtual_tryon_result.png", mime="image/png")

else:
    st.write("Please upload both a photo and a clothing image to try it on.")
