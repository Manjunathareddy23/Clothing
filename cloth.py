import streamlit as st
import cv2
import numpy as np
from PIL import Image

def overlay_cloth_on_user(user_image, cloth_image, x_offset, y_offset, scale, opacity):
    """
    This function overlays the selected cloth image onto the user's photo.
    """
    # Resize the clothing image based on user scaling input
    cloth_resized = cv2.resize(cloth_image, None, fx=scale, fy=scale)
    h, w, _ = cloth_resized.shape
    user_h, user_w, _ = user_image.shape

    # Adjust cloth position on the user image with the specified offsets
    x_pos = min(max(x_offset, 0), user_w - w)
    y_pos = min(max(y_offset, 0), user_h - h)

    # Create a mask for the clothing to apply opacity
    cloth_mask = np.ones((h, w), dtype=np.uint8)
    for c in range(3):
        # Blending the cloth image onto the user image with opacity adjustment
        user_image[y_pos:y_pos + h, x_pos:x_pos + w, c] = \
            (1 - cloth_mask / 255) * user_image[y_pos:y_pos + h, x_pos:x_pos + w, c] + \
            (cloth_mask / 255) * cloth_resized[:, :, c]

    return user_image

st.title("Virtual Clothing Try-On")

# Upload user's photo
user_image_file = st.file_uploader("Upload Your Photo", type=["jpg", "png", "jpeg"])

# Allow the user to upload the clothing image
cloth_file = st.file_uploader("Upload Clothing Image", type=["jpg", "png", "jpeg"])

if user_image_file and cloth_file:
    # Read user's photo
    user_image = np.array(Image.open(user_image_file))
    
    # Read the clothing image
    selected_cloth = cv2.imdecode(np.frombuffer(cloth_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    st.write("Processing...")

    # Sliders for adjusting the position, scale, and opacity of the clothing
    x_offset = st.slider("Adjust X Position", min_value=-100, max_value=100, value=0)
    y_offset = st.slider("Adjust Y Position", min_value=-100, max_value=100, value=0)
    scale = st.slider("Adjust Scale", min_value=0.1, max_value=2.0, value=1.0)
    opacity = st.slider("Adjust Opacity", min_value=0.0, max_value=1.0, value=1.0)

    # Overlay the clothing onto the user's photo
    final_image = user_image.copy()
    final_image = overlay_cloth_on_user(final_image, selected_cloth, x_offset, y_offset, scale, opacity)

    # Convert to PIL image for display
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    final_image_pil = Image.fromarray(final_image_rgb)

    # Display the final image
    st.image(final_image_pil, caption="Result", use_column_width=True)
    
    # Provide an option to download the final image
    final_image_bytes = final_image_pil.tobytes()
    st.download_button("Download Image", data=final_image_bytes, file_name="virtual_tryon_result.png", mime="image/png")

else:
    st.write("Please upload both a photo and a clothing image to try it on.")
