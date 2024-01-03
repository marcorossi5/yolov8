import io

from PIL import Image
import streamlit as st
from ultralytics import YOLO


# config
options_to_yolo_checkpoint = {
    "pretrained": "yolo/models/yolov8n-seg.pt",
    "finetuned": "yolo/runs/detect/train/weights/best.pt"
}

st.set_page_config(layout="wide")

st.session_state["image"] = None

st.session_state["yolo_checkpoint"] = None
st.session_state["yolo_model"] = None

st.title("Object detection with YOLO v8")

device_selection = st.radio("Aquire image : from", ["file", "camera"])


def save_img_in_memory(img: io.BytesIO):
    if img is not None:
        st.session_state["image"] =  Image.open(img)


def select_from_file():
    with st.form("Load form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose a file", type="jpg")
        submitted = st.form_submit_button("Load")
        if submitted:
            save_img_in_memory(uploaded_file)
            uploaded_file = None


def select_from_device():
    screenshot = st.camera_input("camera input")
    save_img_in_memory(screenshot)


def load_yolo(force: bool = False):
    if st.session_state["yolo_model"] is None or force:
        st.session_state["yolo_model"] = YOLO(st.session_state["yolo_checkpoint"])


def detect_with_yolo(img):
    load_yolo()
    result = st.session_state["yolo_model"](img)
    im_array = result[0].plot()
    im = Image.fromarray(im_array[..., ::-1]) 
    st.image(im)


col1, col2 = st.columns(2)
with col1:
    selected_model = st.selectbox(
        "Select model", options_to_yolo_checkpoint.keys()
    )
    st.session_state["yolo_checkpoint"] = options_to_yolo_checkpoint[selected_model]

    match device_selection:
        case "file":
            select_from_file()
        case "camera":
            select_from_device()
        case _:
            raise ValueError("We should not be here")


with col2:
    if st.session_state["image"] is not None:
        with st.spinner("Detecting ..."):
            detect_with_yolo(st.session_state["image"])
