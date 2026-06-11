import os
import cv2
import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# =====================================================================
# 1. HARDWARE & SYSTEM CONFIGURATION
# =====================================================================
MODEL_PATH = 'tire_defect_resnet18.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Defective', 'Good']  # Verified: ImageFolder maps alphabetically (d before g)

# Inspection Parameters
TOTAL_TIMESTEPS = 12   # Number of rotational steps required to cover 360 degrees
NUM_CAMERAS = 3        # Number of synchronized inspection angles per step

# =====================================================================
# 2. MODEL INITIALIZATION
# =====================================================================
def load_inspection_model(model_path):
    """Rebuilds the exact training topology and binds the saved weights."""
    model = models.resnet18()
    num_features = model.fc.in_features
    
    # CORRECTED: Multi-layer classification head matching training topology exactly
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),  
        nn.Linear(256, len(CLASS_NAMES))
    )
    
    # Load the trained math weights into the blueprint structure
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()  # CRITICAL: Disables dropout layers for stable inference
    return model

# Validation transform mappings matching training profile
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =====================================================================
# 3. CORE INFERENCE WORKER
# =====================================================================
def evaluate_frame_matrix(frame, model):
    """Processes a single raw camera matrix frame and returns status and confidence."""
    # Convert OpenCV BGR array to standard RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    # Preprocess and inject batch dimension: [Channels, H, W] -> [1, Channels, H, W]
    input_tensor = inference_transforms(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        # CORRECTED: Clean batch-dimension multi-class evaluation logic
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
    # Index 0 -> Defective, Index 1 -> Good
    status = CLASS_NAMES[predicted_idx.item()]
    return status, confidence.item() * 100

# =====================================================================
# 4. ROTATIONAL CYCLING INSPECTION ROUTINE
# =====================================================================
def run_automated_inspection():
    try:
        model = load_inspection_model(MODEL_PATH)
        print("Inspection model weights successfully bound.")
    except FileNotFoundError:
        print(f"Initialization Error: Model weights file '{MODEL_PATH}' not found.")
        return

    # Connect to the 3 distinct physical hardware cameras
    cam_L = cv2.VideoCapture(0)
    cam_T = cv2.VideoCapture(1)
    cam_R = cv2.VideoCapture(2)

    print("\n=== MACHINE VISION STATION ONLINE ===")
    print("System Status: IDLE | Awaiting Mechanical Trigger...")
    print("Instructions: Press 't' to trigger a tire rotation cycle. Press 'q' to exit system.")

    while True:
        # Display an idle tracking window monitoring the main tread camera feed
        ret, idle_frame = cam_T.read()
        if ret:
            cv2.putText(idle_frame, "SYSTEM IDLE - READY FOR TIRE", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Inspection Station Monitor', idle_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # 't' simulates a hardware signal from the PLC / motor controller initiating rotation
        elif key == ord('t'):
            print("\n--------------------------------------------------")
            print("▶ STARTING TIRE INSPECTION CYCLE (Rotation Initiated)...")
            print("--------------------------------------------------")
            
            # Temporary storage buffer for frames collected during this rotation
            inspection_buffer = []

            # Phase A: Mechanical Rotation & Image Acquisition
            for step in range(1, TOTAL_TIMESTEPS + 1):
                print(f"Capturing step {step}/{TOTAL_TIMESTEPS}...")
                
                # Fetch synchronized frames from the video hardware buffers
                success_L, frame_L = cam_L.read()
                success_T, frame_T = cam_T.read()
                success_R, frame_R = cam_R.read()
                
                if not (success_L and success_T and success_R):
                    print("🚨 Critical Error: Synchronized camera frame capture failure.")
                    break
                
                # Cache images in memory mapped to their acquisition sources
                inspection_buffer.append({"step": step, "camera": "Left Sidewall", "frame": frame_L.copy()})
                inspection_buffer.append({"step": step, "camera": "Center Tread", "frame": frame_T.copy()})
                inspection_buffer.append({"step": step, "camera": "Right Sidewall", "frame": frame_R.copy()})

                # Visualizing the progression on screen during rotation
                combined_live = cv2.hconcat([frame_L, frame_T, frame_R])
                cv2.putText(combined_live, f"SCANNING ROTATION: STEP {step}/{TOTAL_TIMESTEPS}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.imshow('Inspection Station Monitor', combined_live)
                cv2.waitKey(150) # Simulates time delay for physical indexing/stepping movement

            print("\n🔄 Rotation Complete. Processing batch analysis across all surface zones...")

            # Phase B: Batch Processing & Analytical Decision Matrix
            tire_is_defective = False
            defect_logs = []

            for entry in inspection_buffer:
                status, confidence = evaluate_frame_matrix(entry["frame"], model)
                
                if status == "Defective":
                    tire_is_defective = True
                    defect_logs.append(f"-> Defect found by [{entry['camera']}] at Timestep [{entry['step']}] (Confidence: {confidence:.1f}%)")

            # Phase C: Final Verdict Assertion
            print("\n================ FINAL REPORT ================")
            if tire_is_defective:
                print("❌ SYSTEM VERDICT: REJECT TIRE")
                print(f"Total Flaws Localized: {len(defect_logs)}")
                for log in defect_logs:
                    print(log)
                verdict_text = "VERDICT: REJECT TIRE"
                verdict_color = (0, 0, 255) # Red
            else:
                print("✅ SYSTEM VERDICT: PASS TIRE")
                print("No surface defects localized across the 360-degree matrix mapping.")
                verdict_text = "VERDICT: TIRE PASSED"
                verdict_color = (0, 255, 0) # Green
            print("==============================================")

            # Display final static result frame on screen until user continues
            final_display = cv2.hconcat([frame_L, frame_T, frame_R])
            cv2.rectangle(final_display, (0, 0), (final_display.shape[1], final_display.shape[0]), verdict_color, 8)
            cv2.putText(final_display, verdict_text, (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, verdict_color, 3, cv2.LINE_AA)
            cv2.imshow('Inspection Station Monitor', final_display)
            
            print("\nCycle finalized. Press any key on the video screen to reset station to IDLE state.")
            cv2.waitKey(0) # Hold window open until user acknowledgment
            print("System returned to IDLE state. Ready for next mechanical run.")

    # Disconnect sequences
    cam_L.release()
    cam_T.release()
    cam_R.release()
    cv2.destroyAllWindows()
    print("Machine Vision Hardware safely disconnected.")

if __name__ == '__main__':
    run_automated_inspection()