import uvicorn

from api.api import app

# Run from root with: python service/api.py
uvicorn.run(app, host="0.0.0.0", port=8000)
