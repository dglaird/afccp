import dash
from dash.dependencies import Input, Output
import dash_design_kit as ddk
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from app import app
import pages
server = app.server

app.layout = ddk.App([
    ddk.Header([
        ddk.Logo(src=app.get_relative_path('/assets/logo.png')),
        ddk.Title('Analytics'),
        ddk.Menu([
            dcc.Link(
                href=app.get_relative_path('/'),
                children='Home'
            ),
            dcc.Link(
                href=app.get_relative_path('/historical-view'),
                children='Historical View'
            ),
            dcc.Link(
                href=app.get_relative_path('/predicted-view'),
                children='Predicted View'
            ),
        ])
    ]),

    dcc.Location(id='url'),
    html.Div(id='content')
])


@app.callback(Output('content', 'children'), [Input('url', 'pathname')])
def display_content(pathname):
    page_name = app.strip_relative_path(pathname)
    if not page_name:  # None or ''
        return pages.home.layout
    elif page_name == 'historical-view':
        return pages.historical_view.layout
    elif page_name == 'predicted-view':
        return pages.predicted_view.layout


if __name__ == '__main__':
    app.run_server(debug=True)
