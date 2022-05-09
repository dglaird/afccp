import dash
from dash.dependencies import Input, Output
import dash_design_kit as ddk
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px

from app import app

df = px.data.iris()  # sample dataset

layout = html.Div([

    ddk.Card(
        width=100,
        children=[
            ddk.CardHeader(
                title='Summary',
                children=dcc.Dropdown(
                    id='filter-summary',
                    options=[
                        {'label': i, 'value': i}
                        for i in df['species_id'].unique()
                    ],
                    value=df['species_id'].unique()[0]
                ),
                fullscreen=True
            ),
            ddk.Graph(id='historical-summary')
        ]
    ),

    ddk.SectionTitle('Raw Data'),

    ddk.Card(
        width=100,
        children=[
            ddk.DataTable(
                data=df.to_dict('records'),
                columns=[
                    {'name': i, 'id': i}
                    for i in df.columns
                ],
                style_table={
                    'maxHeight': '500px',
                    'overflowY': 'scroll'
                }
            )
        ]
    )

])


@app.callback(Output('historical-summary', 'figure'), [Input('filter-summary', 'value')])
def update_graph(value):
    dff = df[df['species_id'] == value]
    return px.parallel_coordinates(
        dff,
        labels={
            "sepal_width": "Sepal Width",
            "sepal_length": "Sepal Length",
            "petal_width": "Petal Width",
            "petal_length": "Petal Length"
        }
    )
