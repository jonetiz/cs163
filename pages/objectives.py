import dash
from dash import html, dcc

dash.register_page(__name__, path='/objectives')

layout = html.Div([
    html.H1("Understanding the Effects of Inflation Across Socioeconomic Classes"),
    html.I("A project by Henry Chau and Jonathan Etiz for CS 163, Spring 2025"),
    html.H3("Project Objectives"),
    html.P("This project will delve into inflation and earnings data to determine if inflation and other economic factors affect socioeconomic classes evenly. " \
    "We will explore the differences between pure income based classification and a more inclusive definition for classification. " \
    "Using various statistical strategies, we will determine the following questions: "),
    html.Ol([
        html.Li("Does wage growth have any correlation to level of income?"),
        html.Li("Do wages grow uniformly throughout different economic classes (defined purely based on income) resultant to inflation?"),
        html.Li("Does including other features (such as wealth/assets, occupation, and education levels) in economic classification give any more intuition or information than a classification based purely on income? Do any of these other variables, mentioned above, exhibit any influence on wage growth?")
    ]),
    html.P("Scope: This project focuses on data from 2014 to 2024."),
    html.H3("Broader Impacts"),
    html.Ul([
        html.Li("The findings of the analysis can be useful for government agencies to determine regulations necessary to keep lower income earners afloat."),
        html.Li("This analysis can help wage earners understand the factors that affect their livelihood and inform them on what kind of economic policies they should support."),
        html.Li("This project may result in findings that contradict our hypotheses, and in fact show that inflation affects classes uniformly - if this is the case, it may provide some unification across different socioeconomic classes")
    ]),
    html.H3("Data Sources"),
    html.Ol([
        html.Li([
            dcc.Link("U.S. Bureau of Labor Statistics - Labor Force Statistics from the Current Population Survey", href="https://www.bls.gov/cps/earnings.htm", target="_blank"),
            html.Ul([html.Li("Summarized data showing overall trends through the years.")])
        ]),
        html.Li([
            dcc.Link("U.S. Census Bureau - Current Population Survey, Annual Social and Economic Supplements", href="https://www.census.gov/programs-surveys/cps/data/datasets.html ", target="_blank"),
            html.Ul([html.Li("Large datasets containing raw Census data on income and socioeconomic factors.")])
        ])
    ]),
    html.H3("Expected Major Findings (Hypotheses)"),
    html.Ul([
        html.Li("Lower income wage earners do not experience proportional wage growth to higher income earners."),
        html.Li("Wage growth is correlated to other factors commonly associated with socio-economic class division (e.g. assets owned, education level, occupation). A more holistic classification (as opposed to one purely based on income) would provide more information."),
        html.Li("Extremely high income households/families may benefit from inflation in ways other than wages (ie. investments, real estate value)."),
        html.Li("People with more assets are less affected by inflation than people without, regardless of income.")
    ])
])