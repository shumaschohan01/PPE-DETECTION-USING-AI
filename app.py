import gradio as gr
from ultralytics import YOLO
import cv2
import tempfile

# Load model
model = YOLO("PEP-DETECTION.pt")

# Detection function
def detect_media(media):
    
    # IMAGE
    if media.endswith((".png", ".jpg", ".jpeg")):
        results = model(media)
        return results[0].plot()

    # VIDEO
    else:
        cap = cv2.VideoCapture(media)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        out = cv2.VideoWriter(
            tmp_file.name,
            fourcc,
            20.0,
            (int(cap.get(3)), int(cap.get(4)))
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            frame = results[0].plot()
            out.write(frame)

        cap.release()
        out.release()

        return tmp_file.name


# Gradio UI
interface = gr.Interface(
    fn=detect_media,
    inputs=gr.File(),
    outputs=gr.File(),
    title="PEP Detection System"
)

interface.launch()