from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# 绑定静态文件和模板目录
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 模拟数据库（后续可改为SQLite）
users = {"admin": "123456"}


@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username in users and users[username] == password:
        return RedirectResponse(url="/mode", status_code=303)  # ✅ 登录成功后跳到模式选择页
    return templates.TemplateResponse("login.html", {"request": request, "error": "用户名或密码错误"})



@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@app.post("/register", response_class=HTMLResponse)
async def register(request: Request, username: str = Form(...), password: str = Form(...)):
    if username in users:
        return templates.TemplateResponse("register.html", {"request": request, "error": "用户名已存在"})
    users[username] = password
    return RedirectResponse(url="/", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return HTMLResponse("<h1>欢迎进入船舶轨迹预测系统主页！</h1>")

@app.get("/mode", response_class=HTMLResponse)
async def mode_page(request: Request):
    return templates.TemplateResponse("mode.html", {"request": request})
