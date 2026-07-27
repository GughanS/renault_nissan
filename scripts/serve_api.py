import uvicorn
import sys
import os

# Ensure the project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    # Run the FastAPI app via uvicorn
    # wheeleye.api.main:app is the target
    print("Starting WheelEye backend API...")
    uvicorn.run("wheeleye.api.main:app", host="0.0.0.0", port=8000, reload=True)
