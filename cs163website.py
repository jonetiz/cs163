import dash
from dash import Dash, html, dcc, Input, Output, ALL

import analysis, data
import graph_cache

# handlle all analysis/caching in main file so it only runs once to save time
asec_data = data.asec_data
fam_data = data.fam_data

poverty_fig = analysis.PovertyAnalysis(asec_data).visualize()
graph_cache.cache(poverty_fig, 'poverty_fig')

income_dollar_fig, income_pct_fig = analysis.IncomeGrowth(asec_data).visualize()
graph_cache.cache(income_dollar_fig, 'income_dollar_fig')
graph_cache.cache(income_pct_fig, 'income_pct_fig')


education_change, occupation_change = analysis.PersonalIncomeGrowth(asec_data).visualize()
graph_cache.cache(education_change, 'education_change')
graph_cache.cache(occupation_change, 'occupation_change')

quantile = analysis.Quantile(fam_data)
graph_cache.cache(quantile.visualize(percentage=False), 'quantile')
graph_cache.cache(quantile.visualize(percentage=True), 'quantile_pct')

permutation_importance = analysis.PermutationImportance(asec_data)
graph_cache.cache(permutation_importance.visualize(), 'permutation_importance')

cross_sectional_regression = analysis.CrossSectionalRegression(asec_data)
graph_cache.cache(cross_sectional_regression.visualize(), 'cross_sectional_regression')

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