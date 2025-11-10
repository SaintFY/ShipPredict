from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def page(request: Request):
    return templates.TemplateResponse("standard_mode.html", {"request": request})

@router.post("/predict")
async def predict(payload: dict):
    step = int(payload.get("step", 15))
    coords = [[31.2 + k*0.01, 121.5 + k*0.02] for k in range(step)]
    track = {"name": "Fusion", "color": "#00d2ff", "coords": coords}
    return JSONResponse({"tracks": [track]})

@router.get("/export/{fmt}")
async def export(fmt: str):
    return {"ok": True, "path": f"data/results/standard_result.{fmt}"}
