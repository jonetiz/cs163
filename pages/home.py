import dash
from dash import html, dcc

import analysis, data

dollar_change, pct_change = analysis.IncomeGrowth(data.asec_data).visualize()

dash.register_page(__name__, path='/')

layout = html.Div([
    html.H1("Understanding Changes in Socioeconomic Classes Over Time"),
    html.H3("Henry Chau & Jonathan Etiz", style={'text-align': 'center'}),
    html.P( """
           This project is an exploration into earnings data to see the effect of natural inflation on socioeconomic classes, particularly how the lower, middle,
           and upper classes (as defined by Pew Research Center) respond in different ways.
            """),
    html.H2("Change in Income Over Time, by Economic Class"),
    dcc.Tabs([
        dcc.Tab(dcc.Graph(figure=dollar_change), label='Dollar-Value Increase'),
        dcc.Tab(dcc.Graph(figure=pct_change), label='Percentage Increase')
    ]),
    html.I("The above charts show dollar-value and percentage increases in income over time across the different classes." \
    "Notably, percentage increases track similarly across all classes, with the middle class consistently seeing the least percentage increase."),
    html.P("For further information, this site showcases the sum of our work, with the following pages:"),
    html.Ul([
        html.Li([dcc.Link("Objectives", href="/objectives"), " - the project outline, data sources, and hypotheses"]),
        html.Li([dcc.Link("Analytical Methods", href="/methodology"), " - our methodology and analysis strategies"]),
        html.Li([dcc.Link("Major Findings", href="/findings"), " - the major findings of our project"]),
    ])
])