import streamlit as st
import cv2
import numpy as np
from PIL import Image

def overlay_cloth_on_user(user_image, cloth_image):
    """Overlay the selected cloth image onto the user's photo."""
    # Resize the clothing image to match the user's photo size
    cloth_resized = cv2.resize(cloth_image, (user_image.shape[1], user_image.shape[0]))

    # Replace the pixels with the clothing image
    for c in range(3):  # For all color channels (RGB)
        user_image[:, :, c] = (1 - cloth_resized[:, :, c] / 255) * user_image[:, :, c] + (cloth_resized[:, :, c] / 255) * cloth_resized[:, :, c]
    
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
