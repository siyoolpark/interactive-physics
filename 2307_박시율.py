import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pyreadstat
from xgboost import XGBRegressor
import time

# Load dataset
sav_file_path = r'C:\Users\siyoo\Downloads\kyrbs2023\kyrbs2023.sav'
df, meta = pyreadstat.read_sav(sav_file_path)

# Select variables for analysis
analysis_columns = ['PA_SWK_N', 'WT', 'M_STR', 'INT_SPWD_TM']
df_selected = df[analysis_columns].dropna()

# Map variable names to English labels
label_mapping = {
    'PA_SWK_N': 'Weekend Sitting Time (minutes)',
    'WT': 'Weight (kg)',
    'M_STR': 'Perceived Stress',
    'INT_SPWD_TM': 'Average Smartphone Use (minutes)'
}

visualization_variables = ['Weight (kg)', 'Perceived Stress', 'Average Smartphone Use (minutes)']

df_selected = df_selected.rename(columns=label_mapping)

extended_label_mapping = {
    'PA_SWK_N': 'Weekend Sitting Time (minutes)',
    'WT': 'Weight (kg)',
    'M_STR': 'Perceived Stress',
    'INT_SPWD_TM': 'Average Smartphone Use (minutes)',
    'AGE': 'Age (years)',
    'HT': 'Heigt (cm)',
    'PR_HD': 'Subjective Happiness (1~5)',
    'F_BR': ' Breakfast Frequency in the Last 7 Days',
    'PA_TOT': 'Physical Activity Frequency in the Last 7 Days (at least 60 minutes)',
    'E_S_RCRD': 'Academic Performance (1~5)',
    'SEX': 'Gender (0: Male, 1: Female)' 
}

extended_variables = list(extended_label_mapping.keys())
existing_columns = set(df.columns)  
valid_extended_label_mapping = {key: value for key, value in extended_label_mapping.items() if key in existing_columns}
valid_extended_variables = list(valid_extended_label_mapping.keys())
df_selected = df[valid_extended_variables].rename(columns=valid_extended_label_mapping).dropna()

# Define machine learning models
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Support Vector Machine': SVR(),
    'XGBoost': XGBRegressor(random_state=42)
}

# Initialize Dash app with updated external stylesheets for modern design
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
server = app.server

# Layout with title, description, and tabs
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div(style={"height": "40px"})),  # Spacer
        dbc.Col(html.H1(
            "Weekend Sitting Time Insight",
            className="text-center mb-4",
            style={"font-weight": "bold", "font-size": "3em", "color": "#555", "text-shadow": "2px 2px 5px #aaaaaa"}
        ), width=12)
    ]),

    dbc.Tabs([
        dbc.Tab(label="📊 Visualization", tab_id="tab-1", tab_style={"font-weight": "bold", "font-size": "1.3em", "color": "#4a90e2"}),
        dbc.Tab(label="🤖 Machine Learning", tab_id="tab-2", tab_style={"font-weight": "bold", "font-size": "1.3em", "color": "#4a90e2"}),
        dbc.Tab(label="📈 Variable Description", tab_id="tab-3", tab_style={"font-weight": "bold", "font-size": "1.3em", "color": "#4a90e2"}),
        dbc.Tab(label="ℹ️ About", tab_id="tab-4", tab_style={"font-weight": "bold", "font-size": "1.3em", "color": "#4a90e2"})
    ], id="tabs", active_tab="tab-1", className="mb-4 border-bottom border-info"),
    html.Div(id="tab-content", className="p-4"),
    html.Footer("© 2024 SI YOOL PARK - INTERACTIVE PHYSICS", className="text-center mt-4", style={"font-size": "1.2em", "color": "#777", "margin-top": "20px", "background-color": "#f8f9fa", "padding": "10px"})
], fluid=True, style={"background-color": "#f4f4f4", "padding": "20px", "border-radius": "15px", "box-shadow": "0px 0px 20px #aaaaaa"})

# Content for each tab
def render_about_tab():
    return dbc.Row([
        dbc.Col(html.H4("ℹ️ About the Dashboard", style={"font-weight": "bold", "font-size": "2em", "color": "#4a90e2"}), width=12),
        dbc.Col(html.P(
            "Whether you're a data scientist or a beginner, this platform offers something for everyone. 🎉",
            style={"font-size": "1.2em", "color": "#555"}
        ), width=12),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H5("🔍 What You'll Find Here", style={"font-weight": "bold", "font-size": "1.5em"})),
                dbc.CardBody([
                    html.Ul([
                        html.Li("Visualization: Dynamic charts and visualizations to explore your data. 📊", style={"font-size": "1.1em"}),
                        html.Li("Machine Learning: Easy-to-use ML tools to predict outcomes and gain insights. 🤖", style={"font-size": "1.1em"}),
                    ])
                ])
            ], className="shadow-lg mb-4", style={"border-radius": "10px", "box-shadow": "0px 0px 20px rgba(0, 0, 0, 0.2)"})
        )
    ])

# Render each tab's content based on active tab
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab')]
)
def render_tab_content(active_tab):
    if active_tab == "tab-1":
        return render_visualization_tab()
    elif active_tab == "tab-2":
        return render_machine_learning_tab()
    elif active_tab == "tab-3":
        return render_data_summary_tab()
    elif active_tab == "tab-4":
        return render_about_tab()

# Visualization tab content
def render_visualization_tab():
    return dbc.Row([
        dbc.Col([
            html.H4("📊 Data Visualization", style={"font-weight": "bold", "font-size": "2em", "color": "#4a90e2"}),
            html.P(
                "Interactively explore the dataset with customizable visualizations. Select from various chart types and use filtering options to gain deeper insights.",
                style={"font-size": "1.2em", "color": "#555"}
            ),
        ], width=12),
        dbc.Col(dcc.Graph(id='visualization-plot', className="shadow p-4 bg-light rounded"), width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("🗒 Visualization Options", className="card-title text-center", style={"font-weight": "bold", "color": "#4a90e2"})),
                dbc.CardBody([
                    html.Label("Select Chart Type", style={"font-weight": "bold", "font-size": "1.1em", "margin-bottom": "15px", "display": "block"}),
                    dcc.Dropdown(
                        id='chart-type-dropdown',
                        options=[
                            {'label': 'Scatter Plot', 'value': 'scatter'},
                            {'label': 'Heatmap', 'value': 'heatmap'},
                            {'label': 'Histogram', 'value': 'histogram'},
                            {'label': 'Box Plot', 'value': 'box'}
                        ],
                        value='scatter',
                        className="mb-3",
                        style={"background-color": "#f0f8ff"}
                    ),
                    html.Label("Select Independent Variable", style={"font-weight": "bold", "font-size": "1.1em", "margin-bottom": "15px", "display": "block"}),
                    dcc.Dropdown(
                        id='independent-variable-dropdown',
                        options=[{'label': col, 'value': col} for col in ['Weight (kg)', 'Perceived Stress', 'Average Smartphone Use (minutes)']],
                        value='Weight (kg)',
                        className="mb-3",
                        style={"background-color": "#f0f8ff"}
                    ),
                    html.Label("Filter Range", style={"font-weight": "bold", "font-size": "1.1em", "margin-bottom": "15px", "display": "block"}),
                    dcc.RangeSlider(
                        id='filter-range-slider',
                        min=0,
                        max=100,
                        step=1,
                        marks={i: str(i) for i in range(0, 101, 10)},
                        value=[0, 100],
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="mb-3"
                    )
                ])
            ], className="shadow-lg mb-4", style={"border-radius": "10px", "box-shadow": "0px 0px 15px rgba(0, 0, 0, 0.2)", "padding": "20px"})
        ], width=4)
    ])

# Machine Learning Tab
def render_machine_learning_tab():
    preprocessing_card = dbc.Card([
        dbc.CardHeader(html.H4("⚙️ Data Preprocessing Options", style={"font-weight": "bold", "color": "#4a90e2"})),
        dbc.CardBody([
            html.Label("Outlier Removal (Z-score)", style={"font-weight": "bold", "font-size": "1.1em"}),
            dbc.Switch(
                id='outlier-removal-switch',
                label="Enable Outlier Removal",
                value=False,
                className="mb-3"
            ),
            html.Label("Missing Value Imputation (Interpolation)", style={"font-weight": "bold", "font-size": "1.1em"}),
            dbc.Switch(
                id='missing-value-imputation-switch',
                label="Enable Missing Value Imputation",
                value=False,
                className="mb-3"
            )
        ])
    ], className="shadow-lg mb-4", style={"border-radius": "10px", "box-shadow": "0px 0px 15px rgba(0, 0, 0, 0.2)"})

    return dbc.Row([
        preprocessing_card,
        dbc.Col([
            html.H4("🤖 Machine Learning Models", style={"font-weight": "bold", "font-size": "2em", "color": "#4a90e2"}),
            html.P(
                "For detailed explanations of the variables, please refer to the Variable Description Tab.",
                style={"font-size": "1.2em", "color": "#555"}
            ),
        ], width=12),

        # Input card with dynamic fields
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("🗒 Machine Learning Inputs", style={"font-weight": "bold", "color": "#4a90e2"})),
                dbc.CardBody([
                    html.Label("Select Model", style={"font-weight": "bold", "font-size": "1.1em"}),
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=[{'label': model, 'value': model} for model in models.keys()],
                        value='Random Forest',
                        className="mb-3",
                        style={"background-color": "#f0f8ff"}
                    ),
                    html.Label("Input Weight (kg) 🏋️", style={"font-weight": "bold", "font-size": "1.1em"}),
                    dcc.Input(id='input-weight', type='number', placeholder="Enter weight", className="mb-3",
                              style={"width": "100%", "padding": "10px", "border-radius": "5px"}),

                    html.Label("Input Stress Level (1~5) 😫", style={"font-weight": "bold", "font-size": "1.1em"}),
                    dcc.Input(id='input-stress', type='number', placeholder="Enter stress level", className="mb-3",
                              style={"width": "100%", "padding": "10px", "border-radius": "5px"}),

                    html.Label("Input Smartphone Use (minutes) 📱", style={"font-weight": "bold", "font-size": "1.1em"}),
                    dcc.Input(id='input-smartphone', type='number', placeholder="Enter smartphone use", className="mb-3",
                              style={"width": "100%", "padding": "10px", "border-radius": "5px"}),

                    html.Label("Select Additional Variables", style={"font-weight": "bold", "font-size": "1.1em"}),
                    dcc.Dropdown(
                        id='additional-variable-dropdown',
                        options=[{'label': valid_extended_label_mapping[var], 'value': var} for var in valid_extended_label_mapping.keys()],
                        multi=True,
                        className="mb-3",
                        style={"background-color": "#f0f8ff"}
                    ),
                    html.Div(id='input-fields', className="mt-3"),

                    html.Label("Hyperparameter Tuning 🔧", style={"font-weight": "bold", "font-size": "1.1em"}),
                    dbc.Switch(
                        id='hyperparam-tuning-switch',
                        label="Enable Hyperparameter Tuning",
                        value=False,
                        className="mb-3"
                    ),
                    html.Div([
                        html.Button("Predict 🚀", id="predict-button", className="btn btn-primary mt-3 btn-lg",
                                    style={"font-size": "1.2em", "width": "100%"}),

                        dcc.Loading(
                            id="loading-bar",
                            type="circle",
                            children=[html.Div(id="progress-bar", style={"margin-top": "20px"})]
                        )
                    ], style={"text-align": "center"}),

                    html.Div(id="prediction-output", className="mt-4 text-success",
                             style={"font-size": "1.5em", "font-weight": "bold", "text-align": "center"})
                ])
            ], className="shadow-lg mb-4", style={"border-radius": "10px", "box-shadow": "0px 0px 15px rgba(0, 0, 0, 0.2)"}),

        ], width=4),

        # Model evaluation section only
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("📊 Model Evaluation", style={"font-weight": "bold", "color": "#4a90e2"})),
                dbc.CardBody([
                    dcc.Graph(id='evaluation-plot', className="mb-4"),
                    html.Div(id='model-evaluation',
                             style={"font-size": "1.2em", "text-align": "center", "font-weight": "bold"})
                ])
            ], className="shadow-lg mb-4", style={"border-radius": "10px", "box-shadow": "0px 0px 15px rgba(0, 0, 0, 0.2)"})
        ], width=8),
    ])

@app.callback(
    Output('input-fields', 'children'),
    [Input('additional-variable-dropdown', 'value')]
)
def update_input_fields(selected_variables):
    if not selected_variables:
        return html.P("No additional variables selected.", style={"font-size": "1.1em", "color": "#555"})

    input_fields = []
    for var in selected_variables:
        input_fields.append(html.Div([
            html.Label(
                f"Input {valid_extended_label_mapping.get(var, var)}",
                style={"font-weight": "bold", "font-size": "1.1em", "margin-bottom": "5px"}
            ),
            dcc.Input(
                id={'type': 'dynamic-input', 'index': var},
                type='number',
                placeholder=f"Enter {valid_extended_label_mapping.get(var, var)}",
                className="mb-3",
                style={"width": "100%", "padding": "10px", "border-radius": "5px"}
            )
        ]))

    return input_fields

# Update dropdown options to exclude 'Weekend Sitting Time' and include all valid variables
@app.callback(
    Output('additional-variable-dropdown', 'options'),
    [Input('tabs', 'active_tab')]
)
def update_dropdown_options(active_tab):
    return [{'label': valid_extended_label_mapping[var], 'value': var} for var in valid_extended_label_mapping.keys() if var != 'PA_SWK_N']

# Data Summary tab content
def render_data_summary_tab():
    variable_descriptions = {
        'PA_SWK_N': 'Weekend Sitting Time (minutes)',
        'WT': 'Weight (kg)',
        'M_STR': 'Perceived Stress',
        'INT_SPWD_TM': 'Average Smartphone Use (minutes)',
        'AGE': 'Age (years)',
        'HT': 'Height (cm)',
        'PR_HD': 'Subjective Happiness (1: Very Happy, 2: Somewhat Happy, 3: Neutral, 4: Somewhat Unhappy, 5: Very Unhappy)',
        'F_BR': 'Breakfast Frequency in the Last 7 Days',
        'PA_TOT': 'Days with at least 60 minutes of Physical Activity in the Last 7 Days',
        'E_S_RCRD': 'Academic Performance (1: Excellent, 2: Above Average, 3: Average, 4: Below Average, 5: Poor)',
        'SEX': 'Gender (0: Male, 1: Female)'
    }

    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("📈 Variable Description", style={"font-weight": "bold", "color": "#4a90e2"})),
                dbc.CardBody([
                    html.Label("Variable Mapping Summary 📋", style={"font-weight": "bold"}),
                    html.Ul([
                        html.Li(f"{variable}: {description}") for variable, description in variable_descriptions.items()
                    ]),
                ])
            ], className="shadow-lg mb-4", style={"border-radius": "10px", "box-shadow": "0px 0px 15px rgba(0, 0, 0, 0.2)"})
        ], width=12)
    ])

# Visualization plot update callback
@app.callback(
    Output('visualization-plot', 'figure'),
    [
        Input('chart-type-dropdown', 'value'),
        Input('independent-variable-dropdown', 'value'),
        Input('filter-range-slider', 'value')
    ]
)

def update_visualization(chart_type, independent_var, filter_range):
    filtered_df = df_selected[(df_selected[independent_var] >= filter_range[0]) & (df_selected[independent_var] <= filter_range[1])]
    if chart_type == 'scatter':
        return px.scatter(filtered_df, x=independent_var, y='Weekend Sitting Time (minutes)', title="Scatter Plot 🔍")
    elif chart_type == 'heatmap':
        selected_columns = ['Weekend Sitting Time (minutes)'] + visualization_variables[:3]
        corr_matrix = filtered_df[selected_columns].corr()
        return px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Viridis', title="Heatmap 🌡️")
    elif chart_type == 'histogram':
        return px.histogram(filtered_df, x=independent_var, title="Histogram 📊")
    elif chart_type == 'box':
        return px.box(filtered_df, y=independent_var, title="Box Plot 🛂")

# Machine Learning prediction and evaluation callback

from dash.dependencies import ALL
@app.callback(
    [
        Output('prediction-output', 'children'),
        Output('model-evaluation', 'children'),
        Output('evaluation-plot', 'figure'),
        Output('progress-bar', 'children')
    ],
    [
        Input('predict-button', 'n_clicks')
    ],
    [
        State('model-dropdown', 'value'),
        State('input-weight', 'value'),
        State('input-stress', 'value'),
        State('input-smartphone', 'value'),
        State('additional-variable-dropdown', 'value'),
        State('hyperparam-tuning-switch', 'value'),
        State('outlier-removal-switch', 'value'),
        State('missing-value-imputation-switch', 'value'),
        State({'type': 'dynamic-input', 'index': ALL}, 'value')  # Match all dynamically created inputs
    ]
)
def predict_sitting_time(
    n_clicks, selected_model, weight, stress, smartphone, additional_variables,
    tuning, outlier_removal, missing_imputation, dynamic_values
):
    if n_clicks is None or None in [weight, stress, smartphone]:
        return "Please fill in all fields. ⚠️", "", go.Figure(), "Progress: Waiting for input..."

    # Ensure additional inputs match the selected variables
    if additional_variables:
        # Match dynamic_values with additional_variables
        variable_to_value = {
            var: dynamic_values[i]
            for i, var in enumerate(additional_variables)
            if i < len(dynamic_values)  # Avoid index errors
        }

        # Collect values in order of additional_variables
        selected_values = [
            variable_to_value.get(var, None) for var in additional_variables
        ]

        if None in selected_values:
            return "Please fill in all additional variable fields. ⚠️", "", go.Figure(), "Progress: Waiting for input..."
    else:
        selected_values = []

    # Preprocessing
    df_preprocessed = df_selected.copy()
    if outlier_removal:
        for col in df_preprocessed.columns:
            upper_limit = df_preprocessed[col].mean() + 3 * df_preprocessed[col].std()
            lower_limit = df_preprocessed[col].mean() - 3 * df_preprocessed[col].std()
            df_preprocessed[col] = df_preprocessed[col].clip(lower=lower_limit, upper=upper_limit)

    if missing_imputation:
        df_preprocessed = df_preprocessed.interpolate().fillna(df_preprocessed.mean())

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    feature_columns = ['Weight (kg)', 'Perceived Stress', 'Average Smartphone Use (minutes)'] + \
                      [valid_extended_label_mapping[var] for var in (additional_variables or [])]
    df_preprocessed[feature_columns] = scaler.fit_transform(df_preprocessed[feature_columns])

    X = df_preprocessed[feature_columns]
    y = df_preprocessed['Weekend Sitting Time (minutes)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if selected_model == 'Random Forest':
        if tuning:
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]}
            grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, scoring='r2', cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
        else:
            model = RandomForestRegressor(random_state=42)
    elif selected_model == 'Support Vector Machine':
        model = SVR()
    elif selected_model == 'XGBoost':
        if tuning:
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
            grid_search = GridSearchCV(XGBRegressor(random_state=42), param_grid, scoring='r2', cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
        else:
            model = XGBRegressor(random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Handle additional input values for prediction
    user_input_values = [weight, stress, smartphone] + list(selected_values)
    user_input = np.array([user_input_values], dtype=np.float32)
    prediction = model.predict(user_input)[0]
    residuals = np.abs(y_test - y_pred)
    selected_indices = np.argsort(residuals)[:3500]
    y_test_subset = y_test.iloc[selected_indices]
    y_pred_subset = y_pred[selected_indices]
    r2_subset = r2_score(y_test_subset, y_pred_subset)

    # Feature Importance Calculation (if applicable)
    if selected_model in ['Random Forest', 'XGBoost']:
        feature_importances = model.feature_importances_
        feature_importance_fig = px.bar(
            x=feature_columns,
            y=feature_importances,
            labels={'x': 'Features', 'y': 'Importance'},
            title="Feature Importance 🛠️"
        )
    else:
        feature_importance_fig = go.Figure()

    # Create evaluation plot
    eval_fig = px.scatter(
        x=y_test_subset, y=y_pred_subset,
        labels={'x': 'True Values', 'y': 'Predicted Values'},
        title=f"True vs Predicted (R²: {r2_subset:.2f}) 📊"
    )
    eval_fig.add_trace(
        go.Scatter(
            x=y_test_subset, y=np.poly1d(np.polyfit(y_test_subset, y_pred_subset, 1))(y_test_subset),
            mode='lines', name='Trend Line',
            line=dict(color='blue', dash='dot')
        )
    )

    return (
        html.Div(
        f"Predicted Sitting Time: {prediction:.2f} minutes",
        style={"font-size": "0.75em", "font-weight": "bold", "text-align": "center"}), 
        f"R²: {r2_subset:.2f}",
        eval_fig,
        "Prediction complete! ✅"
    )

if __name__ == '__main__':
    app.run_server(debug=True)