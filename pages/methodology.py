import dash
from dash import html, dcc
import data, analysis

dash.register_page(__name__, path='/methodology')

asec_data = data.asec_data
fam_data = data.fam_data

# analysis code found in analysis.py
permutation_importance = analysis.PermutationImportance(asec_data)
cross_sectional_regression = analysis.CrossSectionalRegression(asec_data)
quantile = analysis.Quantile(fam_data)

layout = html.Div([
    html.H3("Permutation Importance Analysis"),
    dcc.Graph(figure=permutation_importance.visualize()),
    html.H4("What it means:"),
    html.P("This plot shows the feature importance (calculated by using a Random Trees Regressor) of each of the listed factors with respect to wage growth. In other words, the greater the importance value, the greater the influence the respective feature has on wage growth."),
    html.H4("What it shows:"),
    html.P("The features with the greatest influence on wage growth, by a large margin, are education level, and job type. The only aspect of age that really influences wage growth is whether or not the individual is elderly. This is inferred from the observation that most of the age bins have a negligible importance value (0.2 and under), whereas the bin for ages 80 and over has a much higher importance value (nearing 0.7)."),
    html.I(f"RMSE: {permutation_importance.rmse}"),

    html.H3("Cross-Sectional Regression Analysis"),
    dcc.Graph(figure=cross_sectional_regression.visualize()),
    html.H4("What it means:"),
    html.P("These plots track the change in regression coefficients for each of the listed features with respect to total earnings between the years 2014 and 2024. This gives insight on each features' correlation with wage growth by showing how their impact changes over time."),
    html.H4("What it shows:"),
    html.P("The plot for education level shows the most consistent upward trend which suggests a potentially strong positive correlation between that feature and wage growth. The plot for age however, shows a moderate downward trend, indicating that age, while not being impactful at all to wages early on, becomes more and more negatively correlated with total earnings. dividends and retirement income have very little impact on earnings across all years. Family size and military service show wide fluctuations across the years, and therefore do not indicate any correlation with wage growth. Occupation type shows a generally positive trend, but also features a strange dip between 2020 and 2022, which could potentially be due to the COVID - 19 pandemic."),

    html.H3("Percentile Analysis of Wage Growth"),
    dcc.Tabs([
        dcc.Tab(label='Wage Growth in Dollars', children=[
            dcc.Graph(figure=quantile.visualize(percentage=False))
        ]),
        dcc.Tab(label='Wage Growth Percentage', children=[
            dcc.Graph(figure=quantile.visualize(percentage=True))
        ])
    ]),
    html.H4("What it means:"),
    html.P("These plots show the percentile wage growths and growth percentages for each class. This effectively allows us to compare wage growth at each percentile for each class, which helps us determine whether there is a trend between income level, and income class to wage growth."),
    html.H4("What it shows:"),
    html.P("This plot shows that generally, wage growth is fairly even across each class, for the most part, disproving our initial hypothesis regarding the relationship between income level and wage growth. The chart shows that the majority of the population regardless of class exhibit wage growth that is between 45% and 50% over 10 years. Looking closer at the plot shows that, if anything, the middle class consistently features a lower growth rate compared to the other two classes. There is a small portion of the population observed that is an exception: the 80th percentile and onwards for the upper class show a spike in wage growth that is uncharacteristic of the rest of the population, peaking nearly at 60% growth at the 90th percentile of the upper class.")
])