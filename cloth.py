import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def overlay_cloth_on_user(user_image, cloth_image):
    """Overlay the selected cloth image onto the user's body using pose landmarks."""
    # Convert to RGB for MediaPipe processing
    user_image_rgb = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(user_image_rgb)

    # Check if pose landmarks are detected
    if results.pose_landmarks is None:
        st.warning("No pose detected. Please upload a clearer image.")
        return user_image

    # Calculate the bounding box for the clothing overlay (using landmarks like shoulders and hips)
    shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hip_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    hip_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    # Calculate width and height of the detected body region (between shoulders and hips)
    body_width = int(abs(shoulder_right.x - shoulder_left.x) * user_image.shape[1])
    body_height = int(abs(hip_left.y - shoulder_left.y) * user_image.shape[0])

    # Resize the clothing image to match the detected body region
    cloth_resized = cv2.resize(cloth_image, (body_width, body_height))

    # Ensure the cloth image has 3 channels (RGB or BGR)
    if cloth_resized.ndim == 2:  # if the clothing image is grayscale
        cloth_resized = cv2.cvtColor(cloth_resized, cv2.COLOR_GRAY2BGR)
    
    # Ensure that cloth_resized has 3 channels
    if cloth_resized.shape[2] != 3:
        raise ValueError("Clothing image must have 3 channels (RGB or BGR).")

    # Calculate the x and y offset for positioning the clothing
    x_offset = int(shoulder_left.x * user_image.shape[1])
    y_offset = int(shoulder_left.y * user_image.shape[0]) - body_height // 2

    # Ensure the offsets are within bounds
    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)

    # Clip the clothing image to prevent it from going out of bounds of the user image
    y_end = min(y_offset + body_height, user_image.shape[0])
    x_end = min(x_offset + body_width, user_image.shape[1])
    
    cloth_resized = cloth_resized[:y_end - y_offset, :x_end - x_offset]

    # Alpha blending for smoother overlay
    for c in range(3):  # For all color channels (RGB)
        # Calculate the alpha blend (cloth over user image)
        alpha = cloth_resized[:, :, c] / 255.0
        user_image[y_offset:y_offset+cloth_resized.shape[0], x_offset:x_offset+cloth_resized.shape[1], c] = (
            (1 - alpha) * user_image[y_offset:y_offset+cloth_resized.shape[0], x_offset:x_offset+cloth_resized.shape[1], c] + 
            alpha * cloth_resized[:, :, c]
        )

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
