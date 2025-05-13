from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.colors as pc
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from time import perf_counter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import plotly.graph_objects as go

class Analysis(ABC):
    """Analysis class to help with logging and organization"""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        print(f"Initializing a new Analysis of type {self.__class__.__name__}")
        print(f"Performing {self.__class__.__name__} Analysis")
        start_time = perf_counter()
        self.do_analysis()
        time = perf_counter() - start_time
        print(f"Completed {self.__class__.__name__} Analysis ({time})")
        
    @abstractmethod
    def do_analysis(self):
        """Perform the analysis"""

        pass

    @abstractmethod
    def visualize(self) -> go.Figure:
        """Return appropriate visualization after analysis"""

        pass

class PermutationImportance(Analysis):
    def do_analysis(self):

        # bin age groups
        def bin_age(row):
            if row < 20:
                return '0-19'
            elif row < 30:
                return '20-29'
            elif row < 40:
                return '30-39'
            elif row < 50:
                return '40-49'
            elif row < 60:
                return '50-59'
            elif row < 70:
                return '60-69'
            elif row < 80:
                return '70-79'
            else:
                return '80+'
            
        self.data['AGE_GROUP'] = self.data['A_AGE'].map(bin_age)
        agg_asec = self.data.groupby(['YEAR', 'AGE_GROUP', 'A_MJOCC', 'A_HGA', 'PEAFEVER'])[['PEARNVAL', 'WSAL_VAL', 'DIV_VAL', 'RTM_VAL']].mean().reset_index()

        pivot = agg_asec.pivot_table(index=['AGE_GROUP', 'A_MJOCC', 'A_HGA', 'PEAFEVER'], columns='YEAR', values='PEARNVAL')
        # Ensure 2014 and 2024 exist in the columns
        if 2014 in pivot.columns and 2024 in pivot.columns:
            # Calculate wage growth between 2024 and 2014
            pivot['WAGE_GROWTH'] = pivot[2024] - pivot[2014]
            # Wage growth in percentage terms
            pivot['WAGE_GROWTH_PCT'] = (pivot['WAGE_GROWTH'] / pivot[2014]) * 100
        else:
            raise ValueError("2014 and/or 2024 not present in the dataset.")

        # Reset index to make it a flat DataFrame
        pivot = pivot.reset_index()

        # Drop the original FTOTVAL columns for clarity
        pivot = pivot[['AGE_GROUP', 'A_MJOCC', 'A_HGA', 'PEAFEVER', 'WAGE_GROWTH', 'WAGE_GROWTH_PCT']]
        pivot.dropna(inplace = True)

        # Step 1: Prepare features and target
        features = [col for col in pivot.columns if col not in ['WAGE_GROWTH', 'WAGE_GROWTH_PCT']]
        X = pivot[features]
        y = pivot['WAGE_GROWTH_PCT']

        # Step 2: One-hot encode and mask invalid values
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Filter out invalid values
        mask = y.replace([np.inf, -np.inf], np.nan).notna()
        X_clean = X_encoded[mask]
        y_clean = y[mask]

        # Clip to avoid log(0) or negative issues
        y_clean = y_clean.clip(lower=-99.9)

        # THEN log1p transform (after scaling percent down)
        y_log = np.log1p(y_clean / 100)

        # Fit model
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_clean, y_log)


        # Step 5: Predict and compute RMSE on log-transformed scale
        y_pred_log = model.predict(X_clean)
        self.rmse = np.sqrt(mean_squared_error(y_log, y_pred_log))
        print(f"RMSE on log-transformed target: {self.rmse:.4f}")

        # Step 6: Compute permutation importance
        result = permutation_importance(model, X_clean, y_log, n_repeats=10, random_state=0, scoring='r2')

        # Step 7: Build importance dataframe
        self.importance_df = pd.DataFrame({
            'Feature': X_clean.columns,
            'Importance': result.importances_mean,
            'Std': result.importances_std
        })

    def visualize(self):
        importance_df = self.importance_df

        age_order = [
            'AGE_GROUP_0-19',
            'AGE_GROUP_20-29',
            'AGE_GROUP_30-39',
            'AGE_GROUP_40-49',
            'AGE_GROUP_50-59',
            'AGE_GROUP_60-69',
            'AGE_GROUP_70-79',
            'AGE_GROUP_80+'
        ]

        # Split into age-related and other features
        age_df = importance_df[importance_df['Feature'].isin(age_order)].copy()
        other_df = importance_df[~importance_df['Feature'].isin(age_order)].copy()

        # Sort age features by the desired order
        age_df['Feature'] = pd.Categorical(age_df['Feature'], categories=age_order, ordered=True)
        age_df = age_df.sort_values('Feature')

        # Concatenate back together
        ordered_df = pd.concat([age_df, other_df])

        feature_labels = {
            'A_HGA': 'Education Level',
            'A_MJOCC': 'Occupation Code',
            'PEAFEVER': 'Military Service',
            'AGE_GROUP_0-19': 'Ages 0-19',
            'AGE_GROUP_20-29': 'Ages 20-29',
            'AGE_GROUP_30-39': 'Ages 30-39',
            'AGE_GROUP_40-49': 'Ages 40-49',
            'AGE_GROUP_50-59': 'Ages 50-59',
            'AGE_GROUP_60-69': 'Ages 60-69',
            'AGE_GROUP_70-79': 'Ages 70-79',
            'AGE_GROUP_80+': 'Ages 80+'
        }

        original_features = ordered_df['Feature'].tolist()
        custom_labels = [feature_labels.get(f, f) for f in original_features]
        unique_features = list(dict.fromkeys(original_features))  # Preserve order
        num_features = len(unique_features)

        start, end = 0.4, 1.0  # <-- adjust this range to get darker colors
        sample_points = [start + (end - start) * i / (num_features - 1) for i in range(num_features)]
        dark_purples = pc.sample_colorscale("Purples", sample_points)

        color_map = {feat: dark_purples[i] for i, feat in enumerate(unique_features)}

        fig = px.bar(
            ordered_df,
            x='Importance',
            y='Feature',
            error_x='Std',
            orientation='h',
            color='Feature',
            color_discrete_map=color_map,
            title='Permutation Importance of Features'
        )
        fig.update_layout(
            showlegend=False,
            yaxis=dict(
                categoryorder='array',
                categoryarray=original_features,
                tickvals=original_features,
                ticktext=custom_labels
            )
        )

        return fig

class CrossSectionalRegression(Analysis):
    def do_analysis(self):
        def recat(row):
            if row == -1 or row == 2:
                return 'NO'
            elif row == 1:
                return 'YES'

        self.data['PEAFEVER'] = self.data['PEAFEVER'].map(recat)

        categorical_features = ['PEAFEVER']
        numerical_features = ['A_AGE', 'A_HGA', 'DIV_VAL', 'A_MJOCC', 'RTM_VAL', 'FPERSONS']

        coefficients_by_year = {}

        # Loop over each unique year in the merged dataframe
        for year in self.data['YEAR'].unique():
            # Filter the data for the current year
            df_year = self.data[self.data['YEAR'] == year]

            # Prepare the features (X) and target (y) for regression
            X = df_year[categorical_features + numerical_features]
            y = df_year['PEARNVAL']  # Or 'wage_growth_percent', depending on the target variable

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create a pipeline with preprocessing and regression model
            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ], remainder='passthrough')

            model = make_pipeline(preprocessor, LinearRegression())

            # Fit the model on the current year's data
            model.fit(X_train, y_train)

            # Get the coefficients for the model
            coefficients = model.named_steps['linearregression'].coef_
            features = categorical_features + numerical_features

            # Store the results in the dictionary with year as the key
            coefficients_by_year[year] = dict(zip(features, coefficients))
        
        rows = []
        for year, features in coefficients_by_year.items():
            for feature, coeff in features.items():
                rows.append({'Year': year, 'Feature': feature, 'Coefficient': coeff})

        self.df = pd.DataFrame(rows)
        
    def visualize(self):
        feature_labels = {
            'PEAFEVER': 'Former or Current Military',
            'A_AGE': 'Age',
            'A_HGA': 'Education Level',
            'A_MJOCC': 'Occupation Type',
            'DIV_VAL': 'Has Dividend Income',
            'RTM_VAL': 'Has Retirement Income',
            'FPERSONS': 'Family Size'
        }

        fig = px.line(
            self.df,
            x='Year',
            y='Coefficient',
            color='Feature',
            markers=True,
            title='Change in Cross-Sectional Regression Coefficients Over Time',
            labels={'Year': 'Year', 'Coefficient': 'Regression Coefficient'}
        )

        # set feature labels
        fig.for_each_trace(lambda t: t.update(name = feature_labels[t.name],
                                              legendgroup = feature_labels[t.name],
                                              hovertemplate = t.hovertemplate.replace(t.name, feature_labels[t.name])))

        fig.update_layout(height=550, width=900)

        return fig

class Quantile(Analysis):
    def do_analysis(self):
        self.data['ADJUSTED_INC'] = self.data['FTOTVAL'] / (self.data['FPERSONS'])**.5

        # Quantile wage growth analysis: compare 2014 vs 2024
        quantiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        summary_list = []

        for year in [2014, 2024]:
            df_year = self.data[self.data['YEAR'] == year]
            for income_class in ['Lower', 'Middle', 'Upper']:
                df_group = df_year[df_year['INCOME_CLASS'] == income_class]
                if not df_group.empty:
                    q_values = df_group['ADJUSTED_INC'].quantile(quantiles).reset_index()
                    q_values.columns = ['Quantile', 'AdjustedIncome']
                    q_values['YEAR'] = year
                    q_values['INCOME_CLASS'] = income_class
                    summary_list.append(q_values)

        # Combine results
        quantile_income_summary = pd.concat(summary_list, ignore_index=True)

        # Pivot for comparison
        self.pivoted = quantile_income_summary.pivot_table(
            index=['Quantile', 'INCOME_CLASS'],
            columns='YEAR',
            values='AdjustedIncome'
        ).reset_index()

        # Calculate absolute and percentage wage growth
        self.pivoted['wage_growth'] = self.pivoted[2024] - self.pivoted[2014]
        self.pivoted['wage_growth_pct'] = 100 * (self.pivoted['wage_growth'] / self.pivoted[2014])
    
    def visualize(self, percentage):
        if percentage:
            # Define desired order
            desired_order = ['Lower', 'Middle', 'Upper']
            quantile_order = [str(i) for i in [.1, .2, .3, .4, .5, .6, .7, .8, .9]]

            # Prepare the data
            heat_data = self.pivoted.iloc[1:].copy()
            heat_data['Quantile'] = heat_data['Quantile'].astype(str)

            # Pivot with specified order
            pivot_table = heat_data.pivot(index='INCOME_CLASS', columns='Quantile', values='wage_growth_pct')
            pivot_table = pivot_table.loc[desired_order, quantile_order]  # enforce order

            # Create z and text values
            z = pivot_table.values
            text = [[f"{val:.1f}%" for val in row] for row in z]

            # Plot
            fig = go.Figure(data=go.Heatmap(
                z=z,
                x=quantile_order,
                y=desired_order,
                text=text,
                texttemplate="%{text}",
                colorscale='Viridis',
                colorbar_title='Wage Growth (%)'
            ))

            fig.update_layout(
                title='Wage Growth Percentage Across Income Classes and Their Percentiles',
                xaxis_title='Income Percentile',
                yaxis_title='Income Class',
                height=500,
                width=800
            )

            return fig
        else:
            fig = px.line(
                self.pivoted.iloc[1:],
                x='Quantile',
                y='wage_growth',
                color='INCOME_CLASS',
                markers=True,
                title='Wage Growth (2014-2024) by Income Class Across Percentiles',
                labels={'wage_growth': 'Wage Growth ($)', 'Quantile': 'Income Percentile', 'INCOME_CLASS': 'Income Class'}
            )

            fig.update_layout(
                height=500,
                width=800,
                xaxis=dict(
                    tickmode='array',
                    tickvals=[.1, .2, .3, .4, .5, .6, .7, .8, .9],
                    ticktext=['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
                )
            )
            return fig

class IncomeGrowth(Analysis):
    def do_analysis(self):
        # calculate change in income over each year for each income bracket
        self.income_data = []
        for year in range(2014, 2025):
            df_year = self.data[self.data['YEAR'] == year]
            for income_class in ['Lower', 'Middle', 'Upper']:
                df_group = df_year[df_year['INCOME_CLASS'] == income_class]
                if not df_group.empty:
                    self.income_data.append({'year': year, 'class': income_class, 'median_income': df_group['ADJUSTED_INC'].median()})

        # calculate change per year
        self.income_data = pd.DataFrame(self.income_data).set_index(['year', 'class'])
        self.income_data['change'] = self.income_data['median_income'].diff(3)
        self.income_data['pct_change'] = self.income_data['median_income'].pct_change(3) * 100

        # calculate total change since 2014
        self.income_data['total_change'] = self.income_data['change'].groupby('class').cumsum().fillna(0)
        self.income_data['total_change_pct'] = ((self.income_data['total_change'] / (self.income_data['median_income'] - self.income_data['total_change'])) * 100).fillna(0)
    
    def visualize(self) -> tuple[go.Figure]:
         # create plotly line chart showing dollar increase
        raw_increase = px.line(self.income_data, x=self.income_data.index.get_level_values(0), y='total_change',
        color=self.income_data.index.get_level_values(1), markers=True, title="Dollar Change in Income Over Time (2014-2024)",
        labels={
            'x': 'Year',
            'total_change': 'Increase since 2014 ($)',
            'color': 'Income Class'
        })

        # create plotly line chart showing percentage increase
        pct_increase = px.line(self.income_data, x=self.income_data.index.get_level_values(0), y='total_change_pct',
        color=self.income_data.index.get_level_values(1), markers=True, title="Percent Change in Income Over Time (2014-2024)",
        labels={
            'x': 'Year',
            'total_change_pct': 'Increase since 2014 (%)',
            'color': 'Income Class'
        })

        return raw_increase, pct_increase
    
class PovertyAnalysis(Analysis):
    def do_analysis(self):
        self.income_data = []
        for year in range(2014, 2025):
            df_year = self.data[(self.data['YEAR'] == year) & (self.data['INCOME_CLASS'] == 'Lower') & (self.data['FPOVCUT'] != -1)]
            self.income_data.append({'year': year, 'median_income': df_year['ADJUSTED_INC'].median(), 'poverty_line': df_year['FPOVCUT'].min()})

        self.income_data = pd.DataFrame(self.income_data).set_index('year')

    def visualize(self):
        fig = px.line(self.income_data, y=['median_income', 'poverty_line'], markers=True, title="Increase in Lower Class Income Relative to Poverty Line",
        labels={
            'year': 'Year',
            'value': 'Dollar Value'
        })

        return fig
    
class PersonalIncomeGrowth(Analysis):
    def do_analysis(self):

        # calculate change in income over each year for individuals based on class, education, and occupation
        income_data = []
        for year in range(2014, 2025):
            df_year = self.data[(self.data['YEAR'] == year) * (self.data['A_AGE'] >= 18)]
            df_year.loc[:, 'EDUCATION'] = df_year['A_HGA'].map({
                0: 'No Diploma',
                31: 'No Diploma',
                32: 'No Diploma',
                33: 'No Diploma',
                34: 'No Diploma',
                35: 'No Diploma',
                36: 'No Diploma',
                37: 'No Diploma',
                38: 'No Diploma',
                39: 'HS Diploma or GED',
                40: 'Some College, no degree',
                41: 'Trade / Vocational School',
                42: 'Associate\'s Degree',
                43: 'Bachelor\'s Degree',
                44: 'Master\'s Degree',
                45: 'Professional Doctorate',
                46: 'Academic Doctorate',

            })
            df_year.loc[:, 'OCCUPATION'] = df_year['A_MJOCC'].map({
                0: 'Unemployed',
                1: 'Management / Business / Financial',
                2: 'Professional',
                3: 'Service Occupations',
                4: 'Sales',
                5: 'Office / Administrative',
                6: 'Farming / Fishing / Agriculture',
                7: 'Construction / Extraction',
                8: 'Installation / Maintenance / Repair',
                9: 'Production / Factory',
                10: 'Transportation'
            }, na_action='ignore')
            income_data.append(df_year)

        group = ['YEAR', 'EDUCATION', 'OCCUPATION']
        filtered_data = pd.concat(income_data)[['YEAR', 'OCCUPATION', 'EDUCATION', 'PEARNVAL']].set_index(group)

        # education and occupation data need to be split because we are looking at individual effects of both on income growth

        # educational data; calculate median income for each year and education level
        self.education_data = filtered_data.groupby(['YEAR', 'EDUCATION']).median()
        self.education_data['change'] = self.education_data.groupby(level=1)['PEARNVAL'].diff().fillna(0)
        self.education_data['pct_change'] = self.education_data.groupby(level=1)['PEARNVAL'].pct_change().fillna(0) * 100

        self.education_data['total_change'] = self.education_data['change'].groupby(level=1).cumsum().fillna(0)
        self.education_data['total_change_pct'] = ((self.education_data['total_change'] / (self.education_data['PEARNVAL'] - self.education_data['total_change'])) * 100).fillna(0)


        # occupational data; calculate median income for each year and occupation
        self.occupation_data = filtered_data.groupby(['YEAR', 'OCCUPATION']).median()
        self.occupation_data['change'] = self.occupation_data.groupby(level=1)['PEARNVAL'].diff().fillna(0)
        self.occupation_data['pct_change'] = self.occupation_data.groupby(level=1)['PEARNVAL'].pct_change().fillna(0) * 100

        self.occupation_data['total_change'] = self.occupation_data['change'].groupby(level=1).cumsum().fillna(0)
        self.occupation_data['total_change_pct'] = ((self.occupation_data['total_change'] / (self.occupation_data['PEARNVAL'] - self.occupation_data['total_change'])) * 100).fillna(0)

        self.education_data = self.education_data.reset_index(level=1)
        self.occupation_data = self.occupation_data.reset_index(level=1)
    def visualize(self) -> tuple[go.Figure]:
        # education data plot
        education_levels = ['No Diploma',
                            'HS Diploma or GED',
                            'Some College, no degree',
                            'Trade / Vocational School',
                            'Associate\'s Degree',
                            'Bachelor\'s Degree',
                            'Master\'s Degree',
                            'Professional Doctorate',
                            'Academic Doctorate']

        education_fig = px.line(self.education_data,
                                y='total_change',
                                color='EDUCATION',
                                markers=True,
                                title="Change in Income Over Time by Education Level (2014-2024)",
                                labels={
                                    'YEAR': 'Year',
                                    'total_change': 'Change since 2014',
                                    'EDUCATION': 'Education Level'
                                },
                                category_orders={
                                    'EDUCATION': education_levels
                                })

        education_fig.update_layout(
            updatemenus = [{
                'buttons':[
                    {
                        'args': [{'y': [self.education_data.loc[self.education_data['EDUCATION'] == i].loc[:, 'total_change'] for i in education_levels]},
                                 {'yaxis.title.text': 'Change since 2014 ($)'}],
                        'label': 'Dollar Value',
                        'method': 'update'
                    },
                    {
                        'args': [{'y': [self.education_data.loc[self.education_data['EDUCATION'] == i].loc[:, 'total_change_pct'] for i in education_levels]},
                                 {'yaxis.title.text': 'Change since 2014 (%)'}],
                        'label': 'Percent Value',
                        'method': 'update'
                    },
                ],
                'direction': 'down',
                'showactive': True,
                'yanchor': 'top',
                'y': 1.125,
                'x': 0.37
            }]
        )

        # occupation plot
        occupations = [
            'Unemployed',
            'Management / Business / Financial',
            'Professional',
            'Service Occupations',
            'Sales',
            'Office / Administrative',
            'Farming / Fishing / Agriculture',
            'Construction / Extraction',
            'Installation / Maintenance / Repair',
            'Production / Factory',
            'Transportation',
            'Armed Forces'] # armed forces is excluded because the data is very inconsistent and ruins the graph

        occupation_fig = px.line(self.occupation_data,
                                y='total_change',
                                color='OCCUPATION',
                                markers=True,
                                title="Change in Income Over Time by Occupation (2014-2024)",
                                labels={
                                    'YEAR': 'Year',
                                    'total_change': 'Change since 2014',
                                    'OCCUPATION': 'Occupation Type'
                                },
                                category_orders={
                                    'OCCUPATION': occupations
                                })

        occupation_fig.update_layout(
            updatemenus = [{
                'buttons':[
                    {
                        'args': [{'y': [self.occupation_data.loc[self.occupation_data['OCCUPATION'] == i].loc[:, 'total_change'] for i in occupations]},
                                 {'yaxis.title.text': 'Change since 2014 ($)'}],
                        'label': 'Dollar Value',
                        'method': 'update'
                    },
                    {
                        'args': [{'y': [self.occupation_data.loc[self.occupation_data['OCCUPATION'] == i].loc[:, 'total_change_pct'] for i in occupations]},
                                 {'yaxis.title.text': 'Change since 2014 (%)'}],
                        'label': 'Percent Value',
                        'method': 'update'
                    },
                ],
                'direction': 'down',
                'showactive': True,
                'yanchor': 'top',
                'y': 1.125,
                'x': 0.37
            }]
        )

        return education_fig, occupation_fig