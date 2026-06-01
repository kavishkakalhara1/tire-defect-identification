import os

import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import models, transforms

MODEL_OPTIONS = {
    "TorchScript (scripted)": "tire_defect_resnet18_scripted.pt",
    "TorchScript (traced)": "tire_defect_resnet18_traced.pt",
    "PyTorch state_dict (.pth)": "tire_defect_resnet18.pth",
}
CLASS_NAMES = ["Defective", "Good"]
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


st.set_page_config(
    page_title="Tyre Defect Test App",
    page_icon="🛞",
    layout="wide",
)


@st.cache_resource
def load_exported_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model file: {model_path}")

    if model_path.endswith(".pt"):
        model = torch.jit.load(model_path, map_location=DEVICE)
        model.eval()
        return model

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


inference_transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(image: Image.Image, model):
    input_tensor = inference_transforms(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    return predicted_class, confidence.item() * 100, probabilities.detach().cpu().tolist()


st.title("Tyre Defect Classification Test Bench")
st.caption("Upload a tyre image and test the exported models directly in the browser.")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Model Selection")
    selected_model_name = st.radio(
        "Choose a model export",
        list(MODEL_OPTIONS.keys()),
        index=0,
        help="Use the exported TorchScript model first. The .pth weights are available as a fallback.",
    )
    selected_model_path = MODEL_OPTIONS[selected_model_name]
    st.write(f"Loading: {selected_model_path}")

    uploaded_file = st.file_uploader(
        "Upload a tyre image",
        type=["png", "jpg", "jpeg", "bmp"],
    )

    run_button = st.button("Run Prediction", type="primary", use_container_width=True)

with right_col:
    st.subheader("Prediction Output")

    if uploaded_file is None:
        st.info("Upload an image, then press Run Prediction.")
    elif run_button:
        try:
            model = load_exported_model(selected_model_path)
            image = Image.open(uploaded_file)
            predicted_class, confidence, probabilities = predict(image, model)

            display_col, result_col = st.columns([1, 1])
            with display_col:
                st.image(image, caption="Uploaded tyre image", use_container_width=True)

            with result_col:
                st.metric("Predicted class", predicted_class)
                st.metric("Confidence", f"{confidence:.2f}%")
                st.write("Class probabilities")
                for class_name, probability in zip(CLASS_NAMES, probabilities):
                    st.write(f"{class_name}: {probability * 100:.2f}%")

            if predicted_class == "Defective":
                st.error("This tyre was classified as defective.")
            else:
                st.success("This tyre was classified as good.")

        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    else:
        st.info("Press Run Prediction to evaluate the uploaded image with the selected export.")

st.divider()
st.write("Model files expected in the workspace:")
st.code("\n".join(MODEL_OPTIONS.values()), language="text")
st.caption("If you want the web app to use only TorchScript exports, keep the .pt files and remove the .pth option.")
