from dash import Dash, html, dcc, page_container, callback, Input, Output

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

server = app.server


server.config['SECRET_KEY'] = 'veryhardtoguesskey' 
server.config['SESSION_TYPE'] = 'session'


app.layout = html.Div([
    dcc.Store(id="session-id", storage_type="session"),  
    dcc.Location(id="url", refresh=True),  
    page_container
])

if __name__ == "__main__":
    app.run_server(debug=True)
