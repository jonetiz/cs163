import plotly.express as px
import dash
from dash import html, dcc
import data
import pandas as pd

dash.register_page(__name__, path='/')

layout = html.Div([
    html.P("This project is an exploration into growth of income across the lower, middle, and upper classes between 2014 and 2024 to gain insight into what factors affect increases in income the most as natural inflation takes course."),
    # dcc.Graph(figure=fig),
    # html.I("This graph shows that the upper class sees remarkably more wage growth than the lower and middle classes, and the wage growth for the top 10% of the upper class is more than double that of the bottom 10% of the upper class.")
])