import plotly.express as px
import dash
from dash import html, dcc
import data

dash.register_page(__name__, path='/findings')

layout = html.Div([
    html.H2("Q1: Does wage growth have any correlation to level of income?"),
    html.H4("BLUF: Yes. But many other factors also apply."),
    html.H2("Q2: Do wages and income grow uniformly throughout different economic classes?"),
    html.H4("BLUF: Yes, roughly."),
    html.H2("Q3: How do asset ownership, education, and other factors not directly related to wages impact income growth?"),
    html.H4("BLUF: The impact is substantial."),
    html.P("Lorem ipsum")
    
])