import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Serverless (Vercel) only allows writing to /tmp. Locally, keep data next to the project
# so it works on Windows too (a hard-coded "/tmp" becomes C:\tmp there).
_DEFAULT_DATA_DIR = Path("/tmp") if os.getenv("VERCEL") else BASE_DIR / "data"
DATA_DIR = Path(os.getenv("DATA_DIR", str(_DEFAULT_DATA_DIR)))

LIVENESS_MODEL_PATH = Path(os.getenv("LIVENESS_MODEL_PATH", str(BASE_DIR / "models" / "MiniFASNetV2.onnx")))
YUNET_MODEL_PATH = Path(os.getenv("YUNET_MODEL_PATH", str(BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx")))
SFACE_MODEL_PATH = Path(os.getenv("SFACE_MODEL_PATH", str(BASE_DIR / "models" / "face_recognition_sface_2021dec_int8.onnx")))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "face_access_users.db")))
ACCESS_LOG_PATH = Path(os.getenv("ACCESS_LOG_PATH", str(DATA_DIR / "face_access_logs.db")))

MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "70"))
LIVENESS_THRESHOLD = float(os.getenv("LIVENESS_THRESHOLD", "0.50"))
FACE_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.42"))
FACE_DETECTION_THRESHOLD = float(os.getenv("FACE_DETECTION_THRESHOLD", "0.85"))
FACE_NMS_THRESHOLD = float(os.getenv("FACE_NMS_THRESHOLD", "0.30"))
FACE_TOP_K = int(os.getenv("FACE_TOP_K", "5000"))

# Enrollment: the browser captures several frames; at least MIN of them must pass
# every check (one face, fully inside the frame, LIVE, embedding OK).
ENROLLMENT_MIN_SAMPLES = int(os.getenv("ENROLLMENT_MIN_SAMPLES", "3"))
ENROLLMENT_MAX_SAMPLES = int(os.getenv("ENROLLMENT_MAX_SAMPLES", "8"))
# How much of the face box may stick out of the frame before a sample is rejected (fraction).
FACE_EDGE_TOLERANCE = float(os.getenv("FACE_EDGE_TOLERANCE", "0.03"))

# Liveness model input format. Default matches the reference MiniFASNetV2 ONNX (raw 0-255 BGR).
# If `python diagnose.py --camera` shows REAL faces scoring SPOOF in this mode but LIVE in another,
# set LIVENESS_INPUT_MODE to that one. Options: bgr255, rgb255, bgr01, rgb01.
LIVENESS_INPUT_MODE = os.getenv("LIVENESS_INPUT_MODE", "bgr255").lower()
# Debug switch for enrollment ONLY. Leave "1" in any real deployment: with "0" anyone (or a photo)
# can be enrolled. Daily access checks still always require LIVE.
REGISTRATION_REQUIRE_LIVENESS = os.getenv("REGISTRATION_REQUIRE_LIVENESS", "1") not in ("0", "false", "False")
