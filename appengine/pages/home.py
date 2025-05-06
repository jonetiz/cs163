import dash
from dash import html, callback, Input, Output, dcc, State, ctx

dash.register_page(__name__, path="/", name="Home")


#home page layout with search tab and view posted reviews tab
layout = html.Div(
    className="glass-style",
    children=[
        dcc.Store(id="session-id", storage_type="session"),
        dcc.Store(id="game-title", storage_type="session"),
        dcc.Location(id="url", refresh=True),
        html.Button("Logout", id="logout-button", n_clicks=0),
        html.Div(id="logout-output", style={"margin-top": "10px"}),
        html.Div(id="debug-output"),
        html.H1(id='welcome', style={"margin-bottom": "20px"}),
        dcc.Tabs(
            id="tabs-home",
            value="search",
            children=[
                dcc.Tab(label="Search Games", value="search"),
                dcc.Tab(label="Your Reviews", value="get_reviews")
            ]
        ),
        html.Div(id="content", style={"margin-top": "20px"})
    ]
)
