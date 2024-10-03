# app.py

import streamlit as st
import cv2
import tempfile
import os
import clarifai_grpc
import clarifai
from PIL import Image
import requests
from clarifai import ClarifaiApp
from clarifai import Image as ClImage

# Replace 'YOUR_CLARIFAI_API_KEY' with your actual API key
CLARIFAI_API_KEY = '5c0df2fce4794ec1a47efe2bb5a01f9c'

def extract_frames(video_path, interval=5):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    success, image = cap.read()
    count = 0
    while success:
        if count % frame_interval == 0:
            frames.append(image)
        success, image = cap.read()
        count += 1
    cap.release()
    return frames

def identify_clothing_items(image):
    # Save image to a temporary file
    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp_image.name, image)

    # Initialize Clarifai app
    clarifai_app = ClarifaiApp(api_key=CLARIFAI_API_KEY)
    model = clarifai_app.models.get('e9576d86d2004ed1a38ba0cf39ecb4b1')  # 'fashion' model ID

    # Predict
    with open(temp_image.name, 'rb') as img_file:
        response = model.predict_by_bytes(img_file.read())

    os.unlink(temp_image.name)  # Delete the temp image file

    concepts = response['outputs'][0]['data'].get('concepts', [])
    items = [concept['name'] for concept in concepts if concept['value'] > 0.9]
    return items

def search_products(item):
    # This is a placeholder function.
    # You can integrate with real shopping APIs like Amazon, eBay, or others.
    # For now, we'll return a Google search link.
    search_url = f"https://www.google.com/search?q=buy+{item.replace(' ', '+')}"
    return search_url

def main():
    st.title("🛍️ Video Clothing Finder")
    st.write("Upload a video, and we'll find links to buy the clothes featured in it!")

    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(uploaded_video)

        st.info("Extracting frames from the video...")
        frames = extract_frames(tfile.name, interval=5)  # Extract a frame every 5 seconds
        st.success(f"Extracted {len(frames)} frames.")

        all_items = set()

        st.info("Identifying clothing items in frames...")
        progress_bar = st.progress(0)
        for idx, frame in enumerate(frames):
            items = identify_clothing_items(frame)
            all_items.update(items)
            progress_bar.progress((idx + 1) / len(frames))

        if all_items:
            st.success("Found the following clothing items:")
            for item in all_items:
                st.write(f"- {item}")
                link = search_products(item)
                st.write(f"[Buy {item}]({link})")
        else:
            st.warning("No clothing items were confidently identified.")

        os.unlink(tfile.name)  # Delete the temp video file

if __name__ == "__main__":
    main()
