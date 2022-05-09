from dash.dependencies import Input, Output
import dash_design_kit as ddk
import dash_html_components as html
import dash_core_components as dcc
from app import app
import plotly.express as px

df = px.data.iris()  # sample dataset

layout = html.Div([

    ddk.Block(
        width=30,
        children=ddk.ControlCard(
            width=100,
            children=[
                ddk.ControlItem(
                    label='Species',
                    children=dcc.Dropdown(
                        id='species',
                        options=[
                            {'label': i, 'value': i}
                            for i in df['species_id'].unique()
                        ],
                        value=df['species_id'].unique()[0]
                    )
                ),
            ]

        )
    ),

    ddk.Block(
        width=70,
        children=[
            ddk.Card(
                width=100,
                children=ddk.Graph(id='splom', style={'height': '800px'})
            ),
            ddk.Card(
                width=50,
                children=ddk.Graph(id='scatter')
            ),
            ddk.Card(
                width=50,
                children=ddk.Graph(id='density')
            ),
        ]
    )

])



@app.callback(
    [Output('splom', 'figure'), Output('scatter', 'figure'), Output('density', 'figure')],
    [Input('species', 'value')])
def update_graph(value):
    dff = df[df['species_id'] == value]
    splom = px.scatter_matrix(
        df,
        dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"]
    )

    scatter = px.scatter(
        dff, x="sepal_width", y="sepal_length",
        marginal_y="violin", marginal_x="violin")

    density = px.density_contour(dff, x="sepal_width", y="sepal_length")

    return [splom, scatter, density]
