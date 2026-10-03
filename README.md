# AI Face Liveness + Access Control

Browser-camera access control using a layered computer-vision pipeline:

**Browser webcam → YuNet face detection + landmarks → MiniFASNetV2 anti-spoofing → SFace face recognition → multi-template matching → GRANTED/DENIED → audit log**

## What changed in this version

- **YuNet** remains the face detector and landmark provider.
- **SFace** remains the identity-recognition model with multiple enrollment templates.
- The old custom MobileNetV3 `.pth` liveness classifier has been removed from the runtime path.
- Liveness now uses **MiniFASNetV2 ONNX**, a lightweight face anti-spoofing model.
- The liveness implementation follows the published MiniFASNetV2 inference setup: 80×80 input, BGR/CHW float32 input, and a 2.7 face crop scale.
- PyTorch/Torchvision are no longer required for liveness, which makes the package smaller and better suited to serverless deployment.

The MiniFASNetV2 implementation used here is based on the public `yakhyo/face-anti-spoofing` inference project, which documents MiniFASNetV2 as a lightweight anti-spoofing model and provides PyTorch/ONNX weights. The upstream repository is Apache-2.0 licensed. Review the exact model and dependency licenses before any commercial deployment.

## Windows setup

Use a fresh virtual environment if the previous project environment contains the old Torch/OpenCV packages:

```cmd
cd /d D:\Downloads\face-liveness-access-control-vercel-minifasnet-final
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_models.py
python check_setup.py
python -m uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000`.

### Required model files

`python download_models.py` downloads:

- `models/face_detection_yunet_2023mar.onnx`
- `models/face_recognition_sface_2021dec_int8.onnx`
- `models/MiniFASNetV2.onnx`

The MiniFASNetV2 ONNX source is the model listed by the upstream face-anti-spoofing project. The published ONNX inference uses a 3-class output and treats class index 1 as Real; this project follows that behavior.

## Liveness threshold

`LIVENESS_THRESHOLD=0.50` is a starting value, not a universal security threshold. For the FYP, collect live and spoof samples from your target camera/environment and report FAR, FRR, APCER/BPCER or an appropriate PAD metric, along with recognition FAR/FRR.

Do not lower the threshold just to make registration pass. A lower threshold can change the security/usability trade-off.

## Recognition threshold

`FACE_RECOGNITION_THRESHOLD=0.42` is also only a starting value. Calibrate it with genuine and impostor SFace cosine scores. Re-enroll users after switching from the old ResNet embedding system because the new system stores SFace templates.

## Vercel

The browser-camera architecture is compatible with a FastAPI-style serverless deployment, but database persistence and model bundle size still need production treatment. SQLite under `/tmp` is demo-only because serverless filesystems are ephemeral. Use a managed database for production.

## Troubleshooting registration

The registration screen now shows the **exact** reason a capture was rejected. Common ones:

- *Face is cut off by the edge of the camera view* - sit in the centre so your whole face is inside the frame.
- *Liveness check said SPOOF* - use even front lighting (no window behind you), 40-80 cm from the camera, no photo/screen.
- *Model file(s) missing* - run `python download_models.py`, then restart `uvicorn`.
- Local data (users, logs) is stored in `./data/` (on Vercel: `/tmp`, demo only).
- `GET /api/health` lists any missing model files.

## Diagnosing without the server

```cmd
python diagnose.py --camera
```

Captures one webcam frame (saved as `diagnose_capture.jpg`) and prints each stage: face found, inside frame, liveness
probabilities under every input format, embedding. If a real face is SPOOF in the current format but LIVE in another,
set `LIVENESS_INPUT_MODE` (bgr255 | rgb255 | bgr01 | rgb01). `REGISTRATION_REQUIRE_LIVENESS=0` disables the liveness
check for enrollment only (debug use; never in production).
