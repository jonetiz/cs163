import dash
from dash import html, dcc
import graph_cache

dash.register_page(__name__, path='/findings')

layout = html.Div([
    html.H2("Overview"),
    html.P("This project sought to gain insight on how income grows across different socioeconomic classes, and what factors affect said growth. This page is intended " \
    "to give a layman a better idea of our project with more of a focus on findings, than the technical details presented in the Analytical Methods page."),
    html.P(["To first clarify some terminology, ", html.B("wage growth"), " refers to actual increase in wages/salary earned from work, whereas ",
            html.B("income growth"), " refers to wage growth and growth in other sources of income such as investments.", "The term ", html.B("adjusted income"),
            " throughout this page refers to income adjusted for a family in accordance with the Organization for Economic Cooperation and Development (OECD) equivalence"
            " scale. This scale calculates adjusted income by dividing the family's total income by the square root of the number of persons."]),
    html.P("In understanding our study, it is crucial to understand what is defined as 'Lower', 'Middle', and 'Upper' classes." \
    " We followed the Pew Research Center's classification:"),
    html.Ul([
        html.Li("Lower Class - families whose adjusted incomes are less than two-thirds of the median income."),
        html.Li("Middle Class - families whose adjusted incomes are between two-thirds and two times the median income."),
        html.Li("Upper Class - families whose adjusted incomes exceed two times the median income.")
    ]),

    html.H2("Claim: Higher earners experience disproportionately higher wage growth."),
    html.H4("BLUF: Our findings find no correlation between higher earnings and higher wage growth, but other factors do apply."),
    html.P("Many people have the conception that lower income earners (particularly those near the poverty line) do not experience proportionate wage growth to higher " \
    "earners, or that they do not see enough increases to outpace poverty. Our analysis found this to be mostly untrue."),
    dcc.Graph(figure=graph_cache.get('poverty_fig')),
    html.I("As shown in the chart above, the lower class's income in general outpaces the poverty line."),
    dcc.Graph(figure=graph_cache.get('income_pct_fig')),
    html.I("The above graph shows the percentage increase of all classes; most earners between 2014 and 2024 saw similar income growth across the lower, middle, " \
    "and upper classes. Note the jump in 2020; this may be due to COVID-19 stimulus checks."),

    html.H4("There does appear to be some disproportionate increase in the 90th percentile of the upper class."),
    dcc.Graph(figure=graph_cache.get('quantile_pct')),
    html.P("This chart shows that the lower, middle, and upper classes generally see an increase between 43 and 50 percent, however the 90th percentile of the" \
    " upper class saw an increase closer to 60 percent. An interesting note from this chart as well, is that upper middle class earners seem to have the least" \
    " percentage increases across the board."),
    html.I("The NaN% for the lower class's 10th percentile is intended, as these are families that have 0 income."),
    
    html.H2("If affluence doesn't equate to higher wage increases, what does?"),
    html.H4("BLUF: Education level and occupation are among the most important factors."),
    html.P("After running permutation importance and cross-sectional regression analyses on our data, we found that the factors that most correlate to higher" \
    " income growth are education level and occupation."),
    dcc.Tabs([
        dcc.Tab(dcc.Graph(figure=graph_cache.get('permutation_importance')), label="Important Features"),
        dcc.Tab(dcc.Graph(figure=graph_cache.get('cross_sectional_regression')), label="Importance of Features Over Time")
    ]),
    html.P("According to the cross-sectional regression, there does seem to be a correlation with family size, and military service had a strong correlation prior to 2021." \
    "Age has a weak negative correlation, likely due to entry level positions paying higher and higher. Having income from investments and retirement had nearly " \
    "no correlation, contradicting one of our hypotheses that people with investments would have much higher income increases. This does make sense, however " \
    "when given the context that the vast majority of people make significantly more money from wages/salary than investments."),

    html.H2("Conclusion"),
    html.P("In general, most families see a similar upward trend of income growth across the socioeconomic spectrum, with the exception being the high-upper class. " \
    "This does, however, disprove our first hypothesis, that lower income earners do not experience proportional wage growth to higher income earners, since those in the " \
    "high-upper class constitute a very small subset of the population; lower income earners, in fact, might experience higher proportional wage growth than the upper " \
    "middle class."),
    html.P("Our study did confirm our second hypothesis, since we found education level and occupation to be the single two strongest factors in wage growth."),
    html.P("Our third hypothesis was inconclusive, as the data regarding investments and asset ownership showed very little correlation to income growth; one shortcoming " \
    "of the dataset is that it didn't include net worth information; if this was the case, we could have normalized that dataset and potentially found different " \
    "results by contrasting wage/salary earners with people who's primary source of income is investments."),
    html.P("Our fourth hypothesis was disproved, as there was no evidence of correlation between asset ownership and income.")
    
])