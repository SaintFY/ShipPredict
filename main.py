# ===============================
# main.py (Day3 修正版)
# ===============================
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 引入 Day3 的两个模式模块
from routers.standard import router as standard_router
from routers.advanced import router as advanced_router

app = FastAPI(title="ShipTrack", docs_url="/docs", redoc_url="/redoc")

# ========== 静态资源与模板路径 ==========
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")  # ✅ 新增：可访问 CSV / PDF
templates = Jinja2Templates(directory="templates")

# ========== 登录、注册部分 ==========
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    # TODO: 替换为真实验证逻辑
    if username == "admin" and password == "123456":
        return RedirectResponse(url="/mode", status_code=303)
    else:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "用户名或密码错误，请重试。"
        })

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def register_submit(request: Request, username: str = Form(...), password: str = Form(...), confirm: str = Form(...)):
    if password != confirm:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "两次输入的密码不一致。"
        })
    # TODO: 写入数据库注册逻辑
    return RedirectResponse(url="/login", status_code=303)

# ========== 模式选择 ==========
@app.get("/", response_class=HTMLResponse)
async def index():
    return RedirectResponse(url="/login")

@app.get("/mode", response_class=HTMLResponse)
async def mode_page(request: Request):
    return templates.TemplateResponse("mode.html", {"request": request})

# ========== 模式页面入口 ==========
@app.get("/standard", response_class=HTMLResponse)
async def standard_page(request: Request):
    """标准模式页面入口"""
    return templates.TemplateResponse("standard_mode.html", {"request": request})

@app.get("/advanced", response_class=HTMLResponse)
async def advanced_page(request: Request):
    """高级模式页面入口"""
    return templates.TemplateResponse("advanced_mode.html", {"request": request})

# ========== 注册后端接口路由 ==========
app.include_router(standard_router, prefix="/standard", tags=["standard"])
app.include_router(advanced_router, prefix="/advanced", tags=["advanced"])

# ========== 运行入口 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
