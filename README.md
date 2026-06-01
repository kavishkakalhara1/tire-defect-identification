# TyreNet: Tyre Defect Classification

This project trains a ResNet-18 model to classify tyre images as **Defective** or **Good**. It also includes exported deployment artifacts and a simple Streamlit app for testing the exported models in a browser.

## Project Files

- `notebok1.ipynb` - training, evaluation, and model export notebook
- `main.py` - OpenCV-based inspection script for live camera workflows
- `streamlit_app.py` - browser app for testing exported models on uploaded images
- `tire_defect_resnet18.pth` - trained PyTorch weights
- `tire_defect_resnet18_scripted.pt` - exported TorchScript scripted model
- `tire_defect_resnet18_traced.pt` - exported TorchScript traced model
- `dataset/` - raw dataset with `Defective/` and `Good/`
- `dataset_split/` - train/validation split used by the notebook

## Model Export Formats

The recommended deployment files are the exported TorchScript models:

- `tire_defect_resnet18_scripted.pt`
- `tire_defect_resnet18_traced.pt`

The Streamlit app can also fall back to the original `.pth` weights if needed.

## Requirements

Install the Python packages used by the notebook and the web app:

```bash
pip install torch torchvision matplotlib scikit-learn seaborn pillow streamlit
```

If you are on macOS and need to bypass system package restrictions, you may need:

```bash
pip install --break-system-packages torch torchvision matplotlib scikit-learn seaborn pillow streamlit
```

## How To Run The Web App

Start the Streamlit app from the project root:

```bash
streamlit run streamlit_app.py
```

Then upload a tyre image and choose one of the exported models.

## How To Use The Notebook

1. Open `notebok1.ipynb`.
2. Run the training cells if you want to retrain the model.
3. Run the export cell to regenerate the `.pt` and `.onnx` artifacts.
4. Use the final inference cell to test a single image locally.

## Notes

- The notebook expects the dataset folders to be named `dataset/Good` and `dataset/Defective`.
- The validation and inference transforms both resize images to `224 x 224` and normalize them with ImageNet statistics.
- If you add new model exports, keep them in the project root or update the paths in `streamlit_app.py`.
