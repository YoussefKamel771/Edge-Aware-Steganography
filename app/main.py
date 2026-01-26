from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from app.dependencies import model, preprocess_image, postprocess_tensor, DEVICE
import torch

app = FastAPI(title="Edge-Aware Steganography API")

@app.post("/hide")
async def hide_secret(
    cover: UploadFile = File(...),
    secret: UploadFile = File(...),
    edge: UploadFile = File(...)
):
    try:
        # 1. Read bytes
        c_bytes = await cover.read()
        s_bytes = await secret.read()
        e_bytes = await edge.read()

        # 2. Preprocess
        c_tensor = preprocess_image(c_bytes)
        s_tensor = preprocess_image(s_bytes)
        e_tensor = preprocess_image(e_bytes, is_edge=True)

        # 3. Inference
        with torch.no_grad():
            stego, _, _ = model.hide_network(c_tensor, s_tensor, e_tensor)

        # 4. Convert to PNG bytes
        result_bytes = postprocess_tensor(stego)

        return Response(content=result_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ready", "device": str(DEVICE)}