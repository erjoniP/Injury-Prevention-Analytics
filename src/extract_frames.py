import os
import cv2
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Load the pre-trained image captioning model from Hugging Face
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_caption(image: Image.Image) -> str:
    """
    Generates a caption for a single image using the pre-trained model.
    """
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def process_video_for_caption(video_path: str, frame_interval: int = 20, output_frames_dir: str = "data/frames") -> list:
    """
    Extracts key frames from the video (every 'frame_interval' frames),
    generates a caption for each frame, and returns a list of captions.
    """
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    captions = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract a frame at the specified interval
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_frames_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            # Convert the frame to a PIL image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            caption = generate_caption(pil_image)
            captions.append(caption)
            print(f"Frame {frame_count}: {caption}")
        
        frame_count += 1
    
    cap.release()
    return captions

# Example usage:
video_path = "data/The Bench Press.mp4"  # Replace with your video file
captions = process_video_for_caption(video_path, frame_interval=30)

# Aggregate the captions into a single summary
summary_text = " ".join(captions)
print("Video Summary:")
print(summary_text)
