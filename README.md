# TyreNet: Tyre Defect Classification

This project trains a ResNet-18 model to classify tyre images as **Defective** or **Good**. It also includes exported deployment artifacts, a Streamlit browser app, and an Expo mobile/web app that share the same prediction flow.

## Project Files

- `notebok1.ipynb` - training, evaluation, and model export notebook
- `main.py` - OpenCV-based inspection script for live camera workflows
- `streamlit_app.py` - browser app for testing exported models on uploaded images
- `api_server.py` - FastAPI inference service used by the Expo app
- `mobile-app/` - Expo app for Android and web
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

## How To Run The Expo App

The Expo app lives in `mobile-app/` and uses `api_server.py` for inference. Because the app is set up for an EAS development build, it should be opened with a development client, not Expo Go.

1. Install the backend dependencies:

```bash
pip install -r api_requirements.txt
```

2. Start the FastAPI server from the project root:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

3. Start the Expo app:

```bash
cd mobile-app
npm run start:dev
```

For Android:

```bash
cd mobile-app
npm run android:dev
```

If you want to use Expo Go instead of a development build, remove `expo-dev-client` and switch back to the plain `expo start` flow. The current project is configured for the dev-client path.

To build an Android APK or AAB with EAS:

```bash
cd mobile-app
npx eas login
npx eas build:configure
npx eas build --platform android --profile development
```

For a production Android bundle:

```bash
cd mobile-app
npx eas build --platform android --profile production
```

If you are running on a physical Android device, set `EXPO_PUBLIC_API_BASE_URL` to your machine's LAN IP, for example `http://192.168.1.20:8000`.

The app supports the same three model choices as the backend:

- `scripted`
- `traced`
- `pth`

## How To Use The Notebook

1. Open `notebok1.ipynb`.
2. Run the training cells if you want to retrain the model.
3. Run the export cell to regenerate the `.pt` and `.onnx` artifacts.
4. Use the final inference cell to test a single image locally.

## Notes

- The notebook expects the dataset folders to be named `dataset/Good` and `dataset/Defective`.
- The validation and inference transforms both resize images to `224 x 224` and normalize them with ImageNet statistics.
- If you add new model exports, keep them in the project root or update the paths in `streamlit_app.py`.
- The Expo app uses base64 image uploads so it can run on both Android and web with the same interface.
