
import json
from pathlib import Path
from urllib.request import (
    Request,
    build_opener,
    ProxyHandler
)

BASE_DIR = Path(__file__).resolve().parent

payload_path = BASE_DIR / "test_payload.json"

with open(payload_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

print("Sending", len(payload), "features to Flask...")

data = json.dumps(payload).encode("utf-8")

request = Request(
    "http://127.0.0.1:5000/predict",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

# Make a direct connection without configured proxies.
opener = build_opener(ProxyHandler({}))

with opener.open(request, timeout=60) as response:
    print("HTTP Status:", response.status)
    print("Response:", response.read().decode("utf-8"))
