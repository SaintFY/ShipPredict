# ===============================
# main.py (Day3 优化完整版)
# ===============================
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# 引入两个模式模块
from routers.standard import router as standard_router
from routers.advanced import router as advanced_router

app = FastAPI(title="ShipTrack", docs_url="/docs", redoc_url="/redoc")

# ===== 静态文件挂载 =====
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/exports", StaticFiles(directory="exports"), name="exports")  # ✅ 新增导出目录访问
templates = Jinja2Templates(directory="templates")

# ===== 允许跨域 =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===== 登录 & 注册 =====
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == "admin" and password == "123456":
        return RedirectResponse(url="/mode", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "用户名或密码错误，请重试。"})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def register_submit(request: Request, username: str = Form(...), password: str = Form(...), confirm: str = Form(...)):
    if password != confirm:
        return templates.TemplateResponse("register.html", {"request": request, "error": "两次输入的密码不一致。"})
    return RedirectResponse(url="/login", status_code=303)

# ===== 模式选择与入口 =====
@app.get("/", response_class=HTMLResponse)
async def index():
    return RedirectResponse(url="/login")

@app.get("/mode", response_class=HTMLResponse)
async def mode_page(request: Request):
    return templates.TemplateResponse("mode.html", {"request": request})

@app.get("/standard", response_class=HTMLResponse)
async def standard_page(request: Request):
    return templates.TemplateResponse("standard_mode.html", {"request": request})

@app.get("/advanced", response_class=HTMLResponse)
async def advanced_page(request: Request):
    return templates.TemplateResponse("advanced_mode.html", {"request": request})

# ===== 注册后端接口路由 =====
app.include_router(standard_router, prefix="/standard", tags=["Standard Mode API"])
app.include_router(advanced_router, prefix="/advanced", tags=["Advanced Mode API"])

# ===== 启动入口 =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
