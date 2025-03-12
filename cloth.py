import streamlit as st
import cv2
import numpy as np
from PIL import Image

def overlay_cloth_on_user(user_image, cloth_image):
    """Overlay the selected cloth image onto the user's body."""
    
    # Convert to grayscale for detecting body/clothing region (using a simple approach)
    gray_user_image = cv2.cvtColor(user_image, cv2.COLOR_BGR2GRAY)

    # Load a pre-trained body detector (Haar Cascade classifier for simple body detection)
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    # Detect bodies in the user's image (this is a simple method and may not be perfect)
    bodies = body_cascade.detectMultiScale(gray_user_image, 1.1, 2)

    if len(bodies) == 0:
        st.warning("No body detected in the image. Please upload a clearer image.")
        return user_image

    # Assuming the first detected body is the one we want to overlay clothing on
    x, y, w, h = bodies[0]

    # Resize the clothing image to fit within the detected body region
    cloth_resized = cv2.resize(cloth_image, (w, h))

    # Overlay the resized clothing onto the detected body region
    for c in range(3):  # For all color channels (RGB)
        user_image[y:y+h, x:x+w, c] = (1 - cloth_resized[:, :, c] / 255) * user_image[y:y+h, x:x+w, c] + (cloth_resized[:, :, c] / 255) * cloth_resized[:, :, c]
    
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

    # Overlay the clothing onto the detected body region of the user's photo
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
