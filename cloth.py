import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Set up MediaPipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_pose(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    return results

def overlay_cloth_on_user(user_image, cloth_image, keypoints, x_offset, y_offset, scale, opacity):
    cloth_resized = cv2.resize(cloth_image, None, fx=scale, fy=scale)
    h, w, _ = cloth_resized.shape
    user_h, user_w, _ = user_image.shape

    # Keypoints-based adjustments
    if keypoints:
        left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        y_pos = int((left_shoulder.y + right_shoulder.y) * 0.5 * user_h) - int(h / 2)
        x_pos = int((left_shoulder.x + right_shoulder.x) * 0.5 * user_w) - int(w / 2)
    else:
        x_pos = min(max(x_offset, 0), user_w - w)
        y_pos = min(max(y_offset, 0), user_h - h)

    # Blending the cloth image onto the user image
    cloth_mask = np.ones((h, w), dtype=np.uint8)
    for c in range(3):
        user_image[y_pos:y_pos + h, x_pos:x_pos + w, c] = \
            (1 - cloth_mask / 255) * user_image[y_pos:y_pos + h, x_pos:x_pos + w, c] + \
            (cloth_mask / 255) * cloth_resized[:, :, c]

    return user_image

st.title("Virtual Clothing Try-On")

user_image_file = st.file_uploader("Upload Your Photo", type=["jpg", "png", "jpeg"])

# Clothing category and item
cloth_categories = {
    'T-shirts': ['tshirt1.png', 'tshirt2.png'],
    'Pants': ['pants1.png', 'pants2.png']
}
category = st.selectbox("Select a Cloth Category", options=list(cloth_categories.keys()))
cloth_options = cloth_categories[category]
cloth_option = st.selectbox("Select Clothing", options=cloth_options)

# Load the selected clothing image
selected_cloth = cv2.imread(cloth_option)

if user_image_file:
    user_image = np.array(Image.open(user_image_file))
    results = detect_pose(user_image)

    if results.pose_landmarks:
        st.write("Pose detected, processing...")

        # Sliders for adjusting the cloth position, scale, and opacity
        x_offset = st.slider("Adjust X Position", min_value=-100, max_value=100, value=0)
        y_offset = st.slider("Adjust Y Position", min_value=-100, max_value=100, value=0)
        scale = st.slider("Adjust Scale", min_value=0.1, max_value=2.0, value=1.0)
        opacity = st.slider("Adjust Opacity", min_value=0.0, max_value=1.0, value=1.0)

        final_image = user_image.copy()
        final_image = overlay_cloth_on_user(final_image, selected_cloth, results.pose_landmarks, x_offset, y_offset, scale, opacity)

        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        final_image_pil = Image.fromarray(final_image_rgb)

        st.image(final_image_pil, caption="Result", use_column_width=True)
        
        final_image_bytes = final_image_pil.tobytes()  # Convert image to bytes
        st.download_button("Download Image", data=final_image_bytes, file_name="virtual_tryon_result.png", mime="image/png")
