import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))

# Define the classes (only WBC)
classes = ["NONE", "WBC"]

# Function to load the YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    return model

# Perform detection and return results
def detect_and_plot(image, model):
    results = model.predict(image)[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    wbc_count = 0

    for detection in results.boxes:
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        conf = detection.conf[0].cpu().numpy()
        cls = detection.cls[0].cpu().numpy()

        if conf >= 0.1:
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='black',  # <-- Changed to BLACK
                facecolor='none'
            )
            ax.add_patch(rect)
            plt.text(
                x1, y1,
                f"{classes[int(cls)]} {conf:.2f}",
                color='white',
                fontsize=12,
                backgroundcolor='black'  # <-- Text background also black
            )
            wbc_count += 1

    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf, wbc_count

# Function to get Gemini response for WBC-related information
def get_gemini_response():
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = """Provide a detailed, beginner-friendly overview of White Blood Cells (WBCs). Include:

- What are WBCs?
- Their role in the immune system
- Types of WBCs
- Normal WBC count range
- Common conditions if WBC count is abnormal

Format the response nicely using markdown with headings and bullet points."""
    
    response = model.generate_content(prompt)
    return response.text.strip()

# Function to handle general user queries related to WBCs and blood health
def get_gemini_response_for_query(user_query):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""You are a helpful medical assistant specialized in hematology, particularly blood cells like White Blood Cells (WBCs), Red Blood Cells (RBCs), and platelets. 

You help users with:
- White Blood Cell(WBC) Information
- Blood cell functions
- Blood diseases and conditions
- Immune system responses
- Lab test interpretations (basic guidance)

If a user asks unrelated things like sports, technology, or travel, politely, or any other topic not related reply:
"I'm sorry, I can only assist with blood cells and hematology-related queries."

Respond with clear headings, bullet points, and markdown formatting.

**User's question:** {user_query}"""
    
    response = model.generate_content(prompt)
    return response.text.strip()

# Streamlit app setup
st.set_page_config(page_title="Advanced WBC Detection & Counting System in Blood Smear Image", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF0800;'>Advanced White Blood Cell (WBC) Detection & Counting System in Blood Smear Image</h1>", unsafe_allow_html=True)

# Initialize session state for chat visibility and history
if 'chat_visible' not in st.session_state:
    st.session_state.chat_visible = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Image upload and processing
st.subheader("Upload Blood Smear Image:")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and display the image
    image = Image.open(uploaded_image)
    image = image.convert("RGB")
    image = image.resize((640, 640))

    st.subheader("Uploaded Blood Smear Image:")
    st.image(image, caption='Uploaded Image (Resized to 640x640)', use_container_width=True)

    # Convert to numpy array
    image_np = np.array(image)

    # Load the YOLO model
    model_path = "wbc_yolo12_model.pt"  # Update your model path
    model = load_model(model_path)

    if model is not None:
        # Perform detection
        result_plot, wbc_count = detect_and_plot(image_np, model)

        # Show detection results
        st.subheader("Detection Results:")
        st.image(result_plot, caption='Detection Results', use_container_width=True)
        st.subheader(f"Total WBCs detected: **{wbc_count}**")

        # Additional WBC Information
        st.markdown("---")
        st.subheader("Learn About White Blood Cells (WBCs)")
        wbc_info = get_gemini_response()
        st.markdown(wbc_info)

    else:
        st.error("Failed to load the model. Please check your model path.")

# Chat assistant section
if st.session_state.chat_visible:
    st.title("WBC Chat Assistant")
    st.write("Ask me anything about blood cells, WBCs, or hematology!")

    # Show chat history
    for user_input, bot_response in st.session_state.chat_history:
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Bot:** {bot_response}")
        st.markdown("---")
    
    # Input field for new question
    user_input = st.text_input("Enter your question:", key="chat_input")

    if user_input:
        bot_response = get_gemini_response_for_query(user_input)

        # Add to chat history
        st.session_state.chat_history.append((user_input, bot_response))

        # Display immediately
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Bot:** {bot_response}")
        st.markdown("---")

    # Clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.chat_visible = False

else:
    # Button to start chat
    if st.button("Start Chat Assistant"):
        st.session_state.chat_visible = True

# Empty rerun to refresh UI
st.empty()
