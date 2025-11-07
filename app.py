# app.py
import dash
from dash import html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("船舶轨迹预测系统"),
    html.P("欢迎，请选择模式进入系统。"),
    dbc.Button("标准模式", color="primary", className="me-2"),
    dbc.Button("高级模式", color="secondary"),
], fluid=True)

if __name__ == '__main__':
    app.run(debug=True)
