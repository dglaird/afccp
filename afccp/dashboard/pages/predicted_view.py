import dash
from dash.dependencies import Input, Output
import dash_design_kit as ddk
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px

from app import app

df = px.data.tips()  # sample dataset

layout = html.Div([

    ddk.Card(
        width=100,
        children=[
            ddk.CardHeader(
                title='Summary',
                children=dcc.Dropdown(
                    id='smoker',
                    options=[
                        {'label': 'Smoker', 'value': 'Yes'},
                        {'label': 'Non-Smoker', 'value': 'No'}
                    ],
                    value=df['smoker'].unique()[0]
                ),
                fullscreen=True
            ),
            ddk.Graph(id='summary')
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


@app.callback(Output('summary', 'figure'), [Input('smoker', 'value')])
def update_graph(value):
    dff = df[df['smoker'] == value]
    return px.scatter(
        dff, x="total_bill", y="tip", facet_row="time", facet_col="day",
        category_orders={
            "day": ["Thur", "Fri", "Sat", "Sun"],
            "time": ["Lunch", "Dinner"]
        }
    )
