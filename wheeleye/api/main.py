import os
import json
import uuid
import time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from wheeleye.pipeline.verifier import WheelEyeVerifier

app = FastAPI(title="WheelEye API", description="Automated Visual Inspection API for WheelEye")

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow local React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument the app for Prometheus
Instrumentator().instrument(app).expose(app)


# Global instances of our models to keep them loaded in memory
verifier = None
START_TIME = time.time()
TOTAL_UNITS = 0
PASSED_UNITS = 0
TOTAL_LATENCY_MS = 0.0

@app.on_event("startup")
def load_models():
    global verifier
    # We try to use the ONNX weights by default for speed, falling back to PT if needed.
    detector_weights = "exports/wheeleye_detector.onnx" if os.path.exists("exports/wheeleye_detector.onnx") else "weights/best.pt"
    classifier_weights = "exports/wheeleye_classifier.onnx" if os.path.exists("exports/wheeleye_classifier.onnx") else "weights/classify_best.pt"
    
    # If weights aren't found at root, might need to adjust paths depending on where this is run,
    # but we will assume it's run from the project root.
    print(f"Initializing Verifier with Detector: {detector_weights}, Classifier: {classifier_weights}", flush=True)
    verifier = WheelEyeVerifier(
        detector_weights_path=detector_weights if os.path.exists(detector_weights) else None,
        classifier_weights_path=classifier_weights if os.path.exists(classifier_weights) else None,
    )

@app.get("/skus")
def get_skus():
    """Return available SKU options for the frontend dropdown menus."""
    return {
        "materials": ["Steel", "Alloy"],
        "tiers": [
            {"name": "Standard", "expected_fasteners": 4},
            {"name": "Premium", "expected_fasteners": 5},
            {"name": "Luxury", "expected_fasteners": 6},
        ],
        "sizes": ["17_inch", "18_inch", "19_inch"],
    }

@app.post("/inspect")
async def inspect_frame(
    file: UploadFile = File(...),
    manifest: str = Form("{}")
):
    global TOTAL_UNITS, PASSED_UNITS, TOTAL_LATENCY_MS
    
    start_inference = time.perf_counter()
    
    manifest_dict = json.loads(manifest)
    
    # Save uploaded file to temp location
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())
        
    try:
        # Run inference
        report = verifier.verify(temp_filename, manifest_dict)
    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
    end_inference = time.perf_counter()
    latency_ms = (end_inference - start_inference) * 1000
    
    # Update stats
    TOTAL_UNITS += 1
    if report["status"] == "PASS":
        PASSED_UNITS += 1
    TOTAL_LATENCY_MS += latency_ms
    
    # Attach stats to the report so frontend can read them
    uptime_seconds = int(time.time() - START_TIME)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    report["stats"] = {
        "station_id": "ST-42B",
        "uptime": f"{hours:03d}:{minutes:02d}:{seconds:02d}",
        "units_inspected": f"{TOTAL_UNITS:,}",
        "pass_rate": f"{(PASSED_UNITS / TOTAL_UNITS * 100):.1f}%" if TOTAL_UNITS > 0 else "100.0%",
        "avg_latency": f"{(TOTAL_LATENCY_MS / TOTAL_UNITS):.1f}ms"
    }
    
    # Optional: Attach a thumbnail (in a real system, you'd send base64 or a URL, 
    # but for simplicity we'll let frontend use a generic pass/fail placeholder unless we base64 the image)
    report["thumbnail"] = "https://placehold.co/100x100/242424/34C759/png?text=OK" if report["status"] == "PASS" else "https://placehold.co/100x100/242424/FF3B30/png?text=FAIL"
    
    return report
