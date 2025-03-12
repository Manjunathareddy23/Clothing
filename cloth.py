import streamlit as st
import cv2
import numpy as np
from PIL import Image

def overlay_cloth_on_user(user_image, cloth_image):
    """
    This function overlays the selected cloth image onto the user's photo.
    The clothing is placed in the user's body area, assuming a simple direct replacement.
    """
    # Resize the clothing image to match the user's size
    cloth_resized = cv2.resize(cloth_image, (user_image.shape[1], user_image.shape[0]))

    # Define a mask for the cloth (simple full opacity mask for now)
    cloth_mask = np.ones_like(cloth_resized, dtype=np.uint8) * 255

    # We assume here that the user's clothes area is the entire image, and we replace it with the clothing
    # You can apply more sophisticated body part segmentation or pose detection for more advanced functionality.
    for c in range(3):  # For all color channels (RGB)
        # Replace the pixels with the clothing image, fully applying the mask (no transparency)
        user_image[:, :, c] = (1 - cloth_mask / 255) * user_image[:, :, c] + (cloth_mask / 255) * cloth_resized[:, :, c]

    return user_image

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

    # Overlay the clothing onto the user's photo
    final_image = user_image.copy()
    final_image = overlay_cloth_on_user(final_image, selected_cloth)

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
