import plotly.express as px
import dash
from dash import html, dcc
import pandas as pd
import data

dash.register_page(__name__, path='/')

income_data = []
for year in range(2014, 2025):
    df_year = data.asec_data[data.asec_data['YEAR'] == year]
    for income_class in ['Lower', 'Middle', 'Upper']:
        df_group = df_year[df_year['INCOME_CLASS'] == income_class]
        if not df_group.empty:
            income_data.append({'year': year, 'class': income_class, 'median_income': df_group['ADJUSTED_INC'].median()})

income_data = pd.DataFrame(income_data).set_index(['year', 'class'])
income_data['change'] = income_data['median_income'].diff(3)
income_data['pct_change'] = income_data['median_income'].pct_change(3) * 100

income_data['total_change'] = income_data['change'].groupby('class').cumsum().fillna(0)
income_data['total_change_pct'] = ((income_data['total_change'] / (income_data['median_income'] - income_data['total_change'])) * 100).fillna(0)

income_data

raw_increase = px.line(income_data, x=income_data.index.get_level_values(0), y='total_change',
color=income_data.index.get_level_values(1), markers=True, title="Dollar Change in Income Over Time (2014-2024)",
labels={
    'x': 'Year',
    'total_change': 'Increase since 2014 ($)',
    'color': 'Income Class'
})

pct_increase = px.line(income_data, x=income_data.index.get_level_values(0), y='total_change_pct',
color=income_data.index.get_level_values(1), markers=True, title="Percent Change in Income Over Time (2014-2024)",
labels={
    'x': 'Year',
    'total_change_pct': 'Increase since 2014 (%)',
    'color': 'Income Class'
})

layout = html.Div([
    html.H1("Changes in Socioeconomic Classes Over Time"),
    html.H3("Henry Chau & Jonathan Etiz", style={'text-align': 'center'}),
    html.H2("Change in Income Over Time, by Economic Class"),
    dcc.Tabs([
        dcc.Tab(dcc.Graph(figure=raw_increase), label='Dollar-Value Increase'),
        dcc.Tab(dcc.Graph(figure=pct_increase), label='Percentage Increase')
    ]),
    html.P( """
           This project is an exploration into earnings data to see the effect of natural inflation on socioeconomic classes, particularly how the lower, middle,
           and upper classes (as defined by Pew Research Center) respond in different ways. This site showcases the sum of our work, with the following pages:
            """),
    html.Ul([
        html.Li([dcc.Link("Objectives", href="/objectives"), " - the project outline, data sources, and hypotheses"]),
        html.Li([dcc.Link("Analytical Methods", href="/methodology"), " - our methodology and analysis strategies"]),
        html.Li([dcc.Link("Major Findings", href="/findings"), " - the major findings of our project"]),
    ])
])