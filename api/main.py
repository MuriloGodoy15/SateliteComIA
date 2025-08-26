import os
import json
import shutil
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LAST_RESULT_PATH = "api/last_result.json"
IMAGE_PATH = "machine learning/modelo treinado/imgs Teste/captura.jpeg"
PREDICT_SCRIPT = "machine learning/modelo treinado/predictPC.py"

def run_prediction():
    # chama o predictPC.py FINALMENTE AHAHAHAHAHAHA 
    proc = subprocess.run(
        ["python", PREDICT_SCRIPT, IMAGE_PATH],
        capture_output=True,
        text=True
    )
    output = proc.stdout

    # pega as infos da saída
    label = ""
    confidence = 0.0
    for line in output.splitlines():
        if "Classe:" in line and "Probabilidade:" in line:
            try:
                parts = line.split("|")
                label = parts[0].replace("Classe:", "").strip()
                confidence = float(parts[1].replace("Probabilidade:", "").replace("%", "").strip())
            except Exception:
                pass

    # salva no JSON
    result = {
        "soilType": label,
        "confidence": confidence,
        "imageUrl": "/photo"
    }
    if os.path.exists(LAST_RESULT_PATH):
        os.remove(LAST_RESULT_PATH)
    with open(LAST_RESULT_PATH, "w") as f:
        json.dump(result, f)
    return result

@app.post("/capture")
async def capture(file: UploadFile = File(...)):
    # substitui a foto antiga
    with open(IMAGE_PATH, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # roda a predição
    return run_prediction()

@app.get("/latest")
def latest_data():
    if not os.path.exists(LAST_RESULT_PATH):
        return {"error": "Nenhum dado ainda."}
    with open(LAST_RESULT_PATH) as f:
        return json.load(f)

@app.get("/photo")
def get_photo():
    if not os.path.exists(IMAGE_PATH):
        return {"error": "Imagem não encontrada"}
    return FileResponse(IMAGE_PATH, media_type="image/jpeg")
