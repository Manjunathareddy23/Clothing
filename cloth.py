import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Set up MediaPipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function for pose detection
def detect_pose(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    return results

# Function to apply wrinkle and shading effects on clothing
def apply_texture_effects(cloth_image):
    wrinkle_texture = cv2.imread('wrinkle_texture.png', cv2.IMREAD_UNCHANGED)
    if wrinkle_texture is not None:
        wrinkle_resized = cv2.resize(wrinkle_texture, (cloth_image.shape[1], cloth_image.shape[0]))
        cloth_image = cv2.addWeighted(cloth_image, 0.7, wrinkle_resized, 0.3, 0)
    return cloth_image

# Function to overlay cloth on user's body with dynamic adjustments based on pose
def overlay_cloth_on_user(user_image, cloth_image, keypoints, x_offset, y_offset, scale, opacity, cloth_color):
    # Resize the cloth image
    cloth_resized = cv2.resize(cloth_image, None, fx=scale, fy=scale)

    # Apply texture effects (wrinkles, shading, etc.)
    cloth_resized = apply_texture_effects(cloth_resized)

    # Change cloth color
    cloth_resized = cv2.convertScaleAbs(cloth_resized, alpha=cloth_color[0], beta=cloth_color[1])

    # Apply opacity
    cloth_resized = cv2.addWeighted(cloth_resized, opacity, np.zeros_like(cloth_resized), 0, 0)

    # Get clothing and user image size
    h, w, _ = cloth_resized.shape
    user_h, user_w, _ = user_image.shape

    # Keypoints adjustment: we use the shoulder and elbow positions to adjust the sleeves and fit
    if keypoints:
        left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = keypoints[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW]

        # Adjust positions dynamically based on pose
        # Adjust the Y position based on the average shoulder height
        y_pos = int((left_shoulder.y + right_shoulder.y) * 0.5 * user_h) - int(h / 2)

        # Adjust the X position based on the average shoulder width
        x_pos = int((left_shoulder.x + right_shoulder.x) * 0.5 * user_w) - int(w / 2)

        # Adjust sleeve position based on elbow position
        if left_elbow and right_elbow:
            sleeve_length = int(abs(left_elbow.y - left_shoulder.y) * user_h)
            cloth_resized = cv2.resize(cloth_resized, (w, sleeve_length))

    else:
        # Default position if pose landmarks aren't found
        x_pos = min(max(x_offset, 0), user_w - w)
        y_pos = min(max(y_offset, 0), user_h - h)

    # Alpha mask for blending
    cloth_mask = cloth_resized[:, :, 3] if cloth_resized.shape[2] == 4 else np.ones((h, w), dtype=np.uint8)
    for c in range(3):
        user_image[y_pos:y_pos + h, x_pos:x_pos + w, c] = \
            (1 - cloth_mask / 255) * user_image[y_pos:y_pos + h, x_pos:x_pos + w, c] + \
            (cloth_mask / 255) * cloth_resized[:, :, c]

    return user_image

# Streamlit UI
st.title("Virtual Clothing Try-On")
st.write("Upload a picture of yourself and a clothing item.")

# Upload images
user_image_file = st.file_uploader("Upload Your Photo", type=["jpg", "png", "jpeg"])

# Multiple cloth categories and items
cloth_categories = {
    'T-shirts': ['tshirt1.png', 'tshirt2.png'],
    'Pants': ['pants1.png', 'pants2.png'],
    'Dresses': ['dress1.png', 'dress2.png'],
    'Jackets': ['jacket1.png', 'jacket2.png']
}

category = st.selectbox("Select a Cloth Category", options=list(cloth_categories.keys()))
cloth_options = cloth_categories[category]
cloth_option = st.selectbox("Select Clothing", options=cloth_options)

# Simulate loading cloth images based on category selection
selected_cloth = cv2.imread(cloth_option)

# List of clothes selected for layering
selected_clothes = []

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
        cloth_color = st.color_picker("Select Cloth Color", "#ffffff")  # Hex color picker

        # Convert the hex color to BGR for OpenCV (because OpenCV uses BGR)
        cloth_color = [int(cloth_color[1:3], 16), int(cloth_color[3:5], 16), int(cloth_color[5:7], 16)]

        # Apply cloth in layers
        if st.button('Add Layer'):
            selected_clothes.append(selected_cloth)

        # Apply all selected clothes
        final_image = user_image.copy()
        for cloth in selected_clothes:
            final_image = overlay_cloth_on_user(final_image, cloth, results.pose_landmarks, x_offset, y_offset, scale, opacity, cloth_color)

        # Convert the final image back to RGB
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        final_image_pil = Image.fromarray(final_image_rgb)

        # Display the real-time preview of the final image
        st.image(final_image_pil, caption="Result", use_column_width=True)

        # Download option
        final_image_bytes = final_image_pil.tobytes()  # Convert image to bytes
        st.download_button("Download Image", data=final_image_bytes, file_name="virtual_tryon_result.png", mime="image/png")

    else:
        st.write("Pose landmarks not detected. Please upload a clear image.")
else:
    st.write("Please upload your photo to get started.")
