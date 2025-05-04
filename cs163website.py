import dash
from dash import Dash, html, dcc, Input, Output, ALL

app = Dash(__name__, use_pages=True)

server = app.server

class NavBlock(html.Div):
    def __init__(self, link, title):
        super().__init__(className="nav_block", id={"type":"link-navbar", "index": link})

        self.children = (dcc.Link(title, href=link))

navblocks = [
    NavBlock("/", "Home"),
    NavBlock("/objectives", "Objectives"),
    NavBlock("/methodology", "Analytical Methods"),
    NavBlock("/findings", "Major Findings")
]

app.layout = html.Div([
    html.Div([
        dcc.Location(id="url"),
        html.Span('Socioeconomic Classes Over Time'),
        html.Div(navblocks)
    ], id="navbar"),
    dash.page_container
], id="page_wrapper")

# animate current page
# from StackOverflow https://stackoverflow.com/questions/73650919/how-to-highlight-active-page-with-a-classname-plotly-dash
@app.callback(Output({"type":"link-navbar", "index":ALL}, "className"), 
[Input("url", "pathname"), Input({"type":"link-navbar", "index":ALL}, "id")])
def callback_func(pathname, link_elements):
    return ["nav_block active" if val["index"] == pathname else "nav_block" for val in link_elements]

if __name__ == '__main__':
    app.run(debug=True)