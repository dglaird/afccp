import base64
import io
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash.dependencies import Input, Output

from dash import dcc, html

import pandas as pd
import data_builder

# create dashboard
app = dash.Dash(__name__)
server = app.server

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

# create dashboard layout
app.layout = html.Div([
    dcc.ConfirmDialog(
        id='confirm-warning',
        message='WARNING: This solution has assigned at least one cadet to an AFSC that they are Ineligible to hold.',
    ),
    dcc.ConfirmDialog(
        id='confirm-nomatch',
        message='Warning: This solution has not assigned any cadets to at least one AFSC.',
    ),
    dcc.ConfirmDialog(
        id='file-warning',
        message='WARNING: This dashboard only excepts .csv and .xlsx files please upload a file of this type.',
    ),
    dcc.ConfirmDialog(
        id='content-warning',
        message=(
            'WARNING: An error occured when processing this file. Please double check the format and contents'
            ' of your file against the provided example file.')
    ),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },

        multiple=True
    ),
    html.H3('Select Solution set'),
    dcc.Dropdown(id='Solution set'),
    html.Hr(),
    dcc.Tabs([
        # layout for first tab
        dcc.Tab(label='Solution Overview', children=[
            html.Div([
                dcc.Graph(id='overview_stats1'),
                dcc.Graph(id='overview_stats2')
            ])
        ]),

        # layout for second tab
        dcc.Tab(label='AFSC Comparison', children=[
            dcc.Graph(id='Total_cadet_numbers'),
            dcc.Graph(id='Merit_perAFSC'),
            dcc.Graph(id='Commissioning_source_balance'),
            dcc.Graph(id='Vol_nonVol'),
        ]),

        # layout for third tab
        dcc.Tab(label='Individual AFSC Stats', children=[
            html.Hr(),
            html.H3('Select AFSC'),
            dcc.Dropdown(id='AFSC dropdown'),
            html.Div(children=[
                dcc.Graph(id="AFSC stats", style={'display': 'inline-block'}),
                dcc.Graph(id="AFSC target", style={'display': 'inline-block'})
            ]),
            html.Div(children=[
                dcc.Graph(id='AFSC stats2', style={'display': 'inline-block'}),
                dcc.Graph(id='AFSC Vol/nonVol', style={'display': 'inline-block'})
            ])
        ])
    ]),
    html.Div(id='output-data-upload'),

    # storage for uploaded/calculated data
    dcc.Store(id='dataset'),
    dcc.Store(id='cadets_fixed_df'),
    dcc.Store(id='solutions_df'),
    dcc.Store(id='afsc_fixed_df'),
    dcc.Store(id='combined_df')
])


# parse the data in the base64 format the upload component creates
def parse_data(contents, filename):
    """Reads the data that is uploaded by the user.

    Parameters
    ----------
    contents : base64
        base64 formated data that was uploaded via the upload component
    filename : base64
        base64 formated filename that was uploaded via the upload component

    Returns
    -------
    pandas dataframe
        cadets fixed sheet of the input file
    pandas dataframe
        solutions sheet of the input file
    pandas dataframe
        afsc fixed sheet of the input file
    pandas dataframe
        combined dataframe of the cadets_fixed and solutions sheets
    bool
        True if file-warning is needed
    bool
        True if content-warning is needed

    """

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    # blank dataframe to use if there is an error parsing data
    blank = pd.DataFrame()
    try:
        if 'csv' in filename:
            # If .csv file
            cadets_fixed_df = pd.read_excel(io.StringIO(decoded.decode('utf-8')),
                                            sheet_name='Cadets Fixed')
            solutions_df = pd.read_excel(io.StringIO(decoded.decode('utf-8')),
                                         sheet_name='Solutions')
            afsc_fixed_df = pd.read_excel(io.StringIO(decoded.decode('utf-8')),
                                          sheet_name='AFSCs Fixed')
            combined_df = pd.merge(cadets_fixed_df,
                                   solutions_df,
                                   on='Cadet')

        elif 'xlsx' in filename:
            # If excel file
            cadets_fixed_df = pd.read_excel(io.BytesIO(decoded),
                                            sheet_name='Cadets Fixed')
            solutions_df = pd.read_excel(io.BytesIO(decoded),
                                         sheet_name='Solutions')
            afsc_fixed_df = pd.read_excel(io.BytesIO(decoded),
                                          sheet_name='AFSCs Fixed')
            combined_df = pd.merge(cadets_fixed_df,
                                   solutions_df,
                                   on='Cadet')

        else:

            # if wrong file type return file-warning message
            return blank, blank, blank, blank, True, False
    except Exception as e:
        print(e)

        # if a read error occurred return content-warning message
        return blank, blank, blank, blank, False, True

    # if no warnings return data
    return cadets_fixed_df, solutions_df, afsc_fixed_df, combined_df, False, False


# ineligible assignment warning popup
@app.callback(Output('confirm-warning', 'displayed'),
              Input('dataset', 'data')
              )
def display_warning(json_data):
    """Determines if the ineligible assignment warning is needed.

    Parameters
    ----------
    json_data : json
        stored data from the data_builder code

    Returns
    -------
    bool
        True if solution has assigned a cadet to an AFSC they are ineligible to hold.

    """
    df = pd.read_json(json_data, orient='split')
    if df['Ineligible'].sum() > 0:
        return True
    return False


# AFSC has received no cadets warning popup
@app.callback(Output('confirm-nomatch', 'displayed'),
              Input('dataset', 'data')
              )
def display_nomatch_warning(json_data):
    """Determines if there is at least one AFSC where the solution has not
    assigned any cadets.

    Parameters
    ----------
    json_data : json
        calculated data from the data_builder function

    Returns
    -------
    bool
        True if solution has assigned no cadets to at least one AFSC

    """
    df = pd.read_json(json_data, orient='split')
    for afsc in df['Total Cadets Assigned']:
        if afsc == 0:
            return True
    return False


# store the dataset for future callbacks to save processing time
@app.callback(Output('cadets_fixed_df', 'data'),  # data storage update
              Output('solutions_df', 'data'),
              Output('afsc_fixed_df', 'data'),
              Output('combined_df', 'data'),

              Output('Solution set', 'options'),  # update solution set dropdown
              Output('Solution set', 'value'),

              Output('AFSC dropdown', 'options'),  # update AFSC selection dropdown
              Output('AFSC dropdown', 'value'),

              Output('file-warning', 'displayed'),  # display warnings if needed
              Output('content-warning', 'displayed'),
              [
                  Input('upload-data', 'contents'),
                  Input('upload-data', 'filename')
              ]
              )
def update_storage(contents, filename):
    """When ever a file is uploaded store the data if correct and update the dropdown
    boxes.

    Parameters
    ----------
    contents : base64
        base64 contents of uploaded file.
    filename : base64
        base64 filename of uploaded file.

    Returns
    -------
    json
        json formated data of the cadets_fixed_df
    json
        json formated data of the solutions_df
    json
        json formated data of the afsc_fixed_df
    json
        json formated data of the combined_df
    Pandas dataframe
        list of solutions for the solutions dropdown component
    str
        default value for the dropdown component
    Pandas dataframe
        list of AFSCs in data for the AFSCs dropdown component
    str
        default value for the dropdown component
    file_warning : bool
        True if file type is not excepted
    content_warning : bool
        True if there was an error trying to read data from the file

    """
    if contents:
        contents = contents[0]
        filename = filename[0]

    # parse data
    cadets_fixed_df, solutions_df, afsc_fixed_df, combined_df, file_warning, content_warning = parse_data(contents, filename)
    print(afsc_fixed_df['AFSC'])

    try:  # normal behaviour
        return (cadets_fixed_df.to_json(date_format='iso', orient='split'),
                solutions_df.to_json(date_format='iso', orient='split'),
                afsc_fixed_df.to_json(date_format='iso', orient='split'),
                combined_df.to_json(date_format='iso', orient='split'),
                solutions_df.columns[1:],
                solutions_df.columns[1],
                afsc_fixed_df['AFSC'],
                afsc_fixed_df['AFSC'][0],
                file_warning,
                content_warning)

    except:  # returned if file-warning or content-warning is needed, as solutions_df and afsc_fixed_df
        # are blank and will return errors in normal behaviour
        return (cadets_fixed_df.to_json(date_format='iso', orient='split'),
                solutions_df.to_json(date_format='iso', orient='split'),
                afsc_fixed_df.to_json(date_format='iso', orient='split'),
                combined_df.to_json(date_format='iso', orient='split'),
                [],
                [],
                [],
                [],
                file_warning,
                content_warning)


# update the dataframe uses for information when the solution or data changes
@app.callback(Output('dataset', 'data'),
              Input('Solution set', 'value'),
              Input('cadets_fixed_df', 'data'),
              Input('solutions_df', 'data'),
              Input('afsc_fixed_df', 'data'),
              Input('combined_df', 'data')
              )
def update_data(solution_name, cadets_fixed_json, solutions_json, afsc_fixed_json, combined_json):
    """Create/Update the data to be used for the dashboard form the uploaded data.

    Parameters
    ----------
    solution_name : str
        solution chosen by user, from the solution set dropdown
    cadets_fixed_json : json
        stored cadets information
    solutions_json : json
        stored solutions information
    afsc_fixed_json : json
        stored AFSCs information
    combined_json : json
        stored combined infromation of the cadets_fixed_df and soltuions_df

    Returns
    -------
    json
        The final calculated information to be used by the infomration display portions of
        the dashboard.

    """

    # create dataframes
    cadets_fixed_df = pd.read_json(cadets_fixed_json, orient='split')
    solutions_df = pd.read_json(solutions_json, orient='split')
    afsc_fixed_df = pd.read_json(afsc_fixed_json, orient='split')
    combined_df = pd.read_json(combined_json, orient='split')

    # run the builder file
    df = data_builder.main(cadets_fixed_df, solutions_df, afsc_fixed_df, combined_df, solution_name)

    return df.to_json(date_format='iso', orient='split')  # store in json format


# create/update the overview page
@app.callback(Output('overview_stats1', 'figure'),
              Output('overview_stats2', 'figure'),
              Input('dataset', 'data')
              )
def update_overview_tab(json_data):
    """Create/update the overview tab when the dataset if updated

    Parameters
    ----------
    json_data : json
        data calculated from the data_builder imported funciton for the selected solution set

    Returns
    -------
    fig : plotly figure
        plotly figure of four indicators describing the overall solutions voluntary and
        USAFA/ROTC distributions
    fig2 : plotly figure
        plotly figure of four indicators describing the overall solutions average merit and
        the number of over/under classified AFSCs

    """
    # read in the data
    df = pd.read_json(json_data, orient='split')

    # ------------------------------------------------------------------
    # create the first graphic
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="number",
        title='Voluntary Cadet Percentage',
        value=df['Voluntary'].sum() / df['Total Cadets Assigned'].sum() * 100,
        domain={'row': 0, 'column': 0}
    ))

    fig.add_trace(go.Indicator(
        mode="number",
        title='Non-Voluntary Cadet Percentage',
        value=100 - df['Voluntary'].sum() / df['Total Cadets Assigned'].sum() * 100,
        domain={'row': 0, 'column': 1}
    ))

    fig.add_trace(go.Indicator(
        mode="number",
        title='Percent of USAFA',
        value=df['Number of USAFA Cadets'].sum() / df['Total Cadets Assigned'].sum() * 100,
        domain={'row': 1, 'column': 0}
    ))

    fig.add_trace(go.Indicator(
        mode="number",
        title='Percent of ROTC',
        value=100 - df['Number of USAFA Cadets'].sum() / df['Total Cadets Assigned'].sum() * 100,
        domain={'row': 1, 'column': 1}
    ))

    fig.update_layout(
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        template={'data': {'indicator': [{
            'mode': "number+delta+gauge",
            'delta': {'reference': 90}}]
        }})

    # ------------------------------------------------------------------
    # create the second graphic
    fig2 = go.Figure()

    fig2.add_trace(go.Indicator(
        mode="number",
        title='Average AFSC Cadet Merit',
        value=df['Cadet Merit Per AFSC'].mean(),
        domain={'row': 0, 'column': 0}
    ))

    fig2.add_trace(go.Indicator(
        mode="number",
        title='Median AFSC Cadet Merit',
        value=df['Cadet Merit Per AFSC'].median(),
        domain={'row': 0, 'column': 1}
    ))

    number_overclassified_afscs = 0
    for num in df['Total Cadets Assigned'] / df['AFSC Max Capacities'] * 100:
        if num > 100:
            number_overclassified_afscs += 1

    fig2.add_trace(go.Indicator(
        mode="number",
        title='Number of Overclassified AFSCs',
        value=number_overclassified_afscs,
        domain={'row': 1, 'column': 0}
    ))

    number_underclassified_afscs = 0
    for num in df['Total Cadets Assigned'] / df['AFSC Min Quotas'] * 100:
        if num < 100:
            number_underclassified_afscs += 1

    fig2.add_trace(go.Indicator(
        mode="number",
        title='Number of Underclassified AFSCs',
        value=number_underclassified_afscs,
        domain={'row': 1, 'column': 1}
    ))

    fig2.update_layout(
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        template={'data': {'indicator': [{
            'mode': "number+delta+gauge",
            'delta': {'reference': 90}}]
        }})

    return fig, fig2


# create/update the comparisson tab
@app.callback(Output('Total_cadet_numbers', 'figure'),
              Output('Merit_perAFSC', 'figure'),
              Output('Commissioning_source_balance', 'figure'),
              Output('Vol_nonVol', 'figure'),
              Input('dataset', 'data')
              )
def update_comparisson_tab(json_data):
    """Creates/updates the comparisson tab when the underlying data is updated.

    Parameters
    ----------
    json_data : json
        underlying stored json data

    Returns
    -------
    fig : plotly figure
        shows the total number of cadets assigned to an AFSC as well as that AFSC's
        requested min and max requested cadets
    fig2 : plotly figure
        shows the average merit of cadets assigned to an AFSC
    fig3 : plotly figure
        Displays the distribution of USAFA cadets to ROTC cadets in an AFSC
    fig4 : plotly figure
        Displays the number of cadets in an AFSC that were non-voluntary, seperated
        by the AFSC's degree requirements

    """

    # read in the data
    df = pd.read_json(json_data, orient='split')

    # ------------------------------------------------------------------
    # create/update the total cadets assigned graph
    fig = px.bar(df, x="AFSCs", y='Total Cadets Assigned')

    # create the min/max requested cadet lines
    ply_shapes = {}
    for i in range(1, len(df) * 2 + 1):
        # min cadet lines
        if i <= len(df):
            ply_shapes['shape_' + str(i)] = go.layout.Shape(type="line",
                                                            x0=df.index[i - 1] - 0.4,
                                                            y0=df['AFSC Min Quotas'].iloc[i - 1],
                                                            x1=df.index[i - 1] + 0.4,
                                                            y1=df['AFSC Min Quotas'].iloc[i - 1],
                                                            line=dict(
                                                                color="Red",
                                                                width=2),
                                                            layer="above",
                                                            name='Min'
                                                            )
        # create max cadet lines
        else:
            ply_shapes['shape_' + str(i)] = go.layout.Shape(type="line",
                                                            x0=df.index[i - len(df) - 1] - 0.4,
                                                            y0=df['AFSC Max Capacities'].iloc[i - len(df) - 1],
                                                            x1=df.index[i - len(df) - 1] + 0.4,
                                                            y1=df['AFSC Max Capacities'].iloc[i - len(df) - 1],
                                                            line=dict(
                                                                color="Green",
                                                                width=2),
                                                            layer="above",
                                                            name='Max'
                                                            )

    # add shapes to figure and format text
    lst_shapes = list(ply_shapes.values())
    fig.update_layout(shapes=lst_shapes,
                      yaxis_title='Total Cadets Assigned',
                      xaxis_title='AFSC',
                      title_text='Total Number of Cadets Assigned to Each AFSC',
                      title_x=0.5)

    # add shapes to legend
    dummy = df['AFSCs'][0]
    fig.add_trace(
        go.Scatter(
            x=[dummy, dummy],
            y=[1, 1],
            mode="lines",
            name="Maximum Capacity",
            line=go.scatter.Line(color="green"),
            showlegend=True)
    )

    fig.add_trace(
        go.Scatter(
            x=[dummy, dummy],
            y=[1, 1],
            mode="lines",
            name="Minimum Quota",
            line=go.scatter.Line(color="red"),
            showlegend=True)
    )

    # ------------------------------------------------------------------
    # create/update the afsc merit graph
    fig2 = px.scatter(df, x='AFSCs', y='Cadet Merit Per AFSC')
    fig2.update_layout(yaxis_title='Average Cadet Merit',
                       xaxis_title='AFSC',
                       title_text='Average Cadet Merit within each AFSC',
                       title_x=0.5)

    # ------------------------------------------------------------------
    # create/update the cadet balance graph
    AFSCs = df['AFSCs']
    fig3 = go.Figure(data=[
        go.Bar(name='ROTC Cadets', x=AFSCs, y=df['Percent of ROTC Cadets in AFSC']),
        go.Bar(name='USAFA Cadets', x=AFSCs, y=df['Percent of USAFA Cadets in AFSC'])
    ])
    fig3.update_layout(barmode='stack',
                       yaxis_title='Percent',
                       xaxis_title='AFSC',
                       title_text='Cadet USAFA/ROTC balance per AFSC',
                       title_x=0.5)

    # ------------------------------------------------------------------
    # create/update the volutary/nonvolutary graph
    AFSCs = df['AFSCs']
    fig4 = go.Figure(data=[
        go.Bar(name='Mandatory Non-Vol', x=AFSCs, y=df['Mandatory Non-Vol'] / df['Total Cadets Assigned'] * 100),
        go.Bar(name='Permitted Non-Vol', x=AFSCs, y=df['Permitted Non-Vol'] / df['Total Cadets Assigned'] * 100),
        go.Bar(name='Desired Non-Vol', x=AFSCs, y=df['Desired Non-Vol'] / df['Total Cadets Assigned'] * 100),
        go.Bar(name='Ineligible', x=AFSCs, y=df['Ineligible'] / df['Total Cadets Assigned'] * 100),
        go.Bar(name='Voluntary', x=AFSCs, y=df['Voluntary'] / df['Total Cadets Assigned'] * 100)
    ])
    fig4.update_layout(barmode='stack',
                       yaxis_title='Percent of Cadets Assigned',
                       xaxis_title='AFSC',
                       title_text='Voluntary/Non-Voluntary Partitioning per AFSC',
                       title_x=0.5)

    return fig, fig2, fig3, fig4


# create/update the single AFSC tab
@app.callback(Output('AFSC stats', 'figure'),
              Output('AFSC target', 'figure'),
              Output('AFSC stats2', 'figure'),
              Output('AFSC Vol/nonVol', 'figure'),
              Input('dataset', 'data'),
              Input('AFSC dropdown', 'value')
              )
def update_afsc_tab(json_data, afsc):
    """Creates/Updates the AFSC tab when the underlying data is changed or a
    different AFSC is selected from the dropdown.

    Parameters
    ----------
    json_data : json
        stored underlying data
    afsc : str
        selected AFSC the user wishes to see the stats for

    Returns
    -------
    fig1 : plotly figure
        Indicator statistics. Voluntary/non-voluntary percentages and USAFA/ROTC balance
    fig2 : plotly figure
        Displayed the selcted AFSC's min/max cadets requested and how much the solution
        assigned to them.
    fig3 : plotly figure
        Indicator statistics. Cadet merit and how much the AFSC is over/under classified by
    fig4 : plotly figure
        Displayed the number of non-voluntary cadets assgined to the AFSC selected; broken out
        by that AFSC's degree requirements.

    """

    # read in the data
    df = pd.read_json(json_data, orient='split')

    # create the first statistics block
    fig1 = go.Figure()

    fig1.add_trace(go.Indicator(
        mode="number",
        title='Voluntary Percentage',
        value=(df.loc[df['AFSCs'] == afsc]['Voluntary'].sum() /
               df.loc[df['AFSCs'] == afsc]['Total Cadets Assigned'].sum() * 100),
        domain={'row': 0, 'column': 0}
    ))

    fig1.add_trace(go.Indicator(
        mode="number",
        title='Non-Voluntary Percentage',
        value=(100 - df.loc[df['AFSCs'] == afsc]['Voluntary'].sum() /
               df.loc[df['AFSCs'] == afsc]['Total Cadets Assigned'].sum() * 100),
        domain={'row': 0, 'column': 1}
    ))

    fig1.add_trace(go.Indicator(
        mode="number",
        title='Percent of USAFA',
        value=(df.loc[df['AFSCs'] == afsc]['Number of USAFA Cadets'].sum() /
               df.loc[df['AFSCs'] == afsc]['Total Cadets Assigned'].sum() * 100),
        domain={'row': 1, 'column': 0}
    ))

    fig1.add_trace(go.Indicator(
        mode="number",
        title='Percent of ROTC',
        value=(100 - df.loc[df['AFSCs'] == afsc]['Number of USAFA Cadets'].sum() /
               df.loc[df['AFSCs'] == afsc]['Total Cadets Assigned'].sum() * 100),
        domain={'row': 1, 'column': 1}
    ))

    fig1.update_layout(
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        template={'data': {'indicator': [{
            'mode': "number+delta+gauge",
            'delta': {'reference': 90}}]
        }})

    # ------------------------------------------------------------------
    # create/update the target graph for the afsc
    AFSCs = df.loc[df['AFSCs'] == afsc]
    fig2 = px.histogram(df, x=AFSCs["AFSCs"], y=AFSCs['Total Cadets Assigned'])
    ply_shapes = {}
    for i in range(0, len(AFSCs) + 1):
        print(i)
        if i < len(AFSCs):
            ply_shapes['shape_' + str(i)] = go.layout.Shape(type="line",
                                                            x0=-0.4,
                                                            y0=AFSCs['AFSC Min Quotas'].iloc[i - 1],
                                                            x1=0.4,
                                                            y1=AFSCs['AFSC Min Quotas'].iloc[i - 1],
                                                            line=dict(
                                                                color="Red",
                                                                width=2),
                                                            layer="above"
                                                            )
        else:
            ply_shapes['shape_' + str(i)] = go.layout.Shape(type="line",
                                                            x0=-0.4,
                                                            y0=AFSCs['AFSC Max Capacities'].iloc[i - len(AFSCs) - 1],
                                                            x1=0.4,
                                                            y1=AFSCs['AFSC Max Capacities'].iloc[i - len(AFSCs) - 1],
                                                            line=dict(
                                                                color="Green",
                                                                width=2),
                                                            layer="above"
                                                            )

    lst_shapes = list(ply_shapes.values())
    fig2.update_layout(shapes=lst_shapes,
                       yaxis_title='Number of Cadets Assigned',
                       xaxis_title='AFSC',
                       title_text='Total Number of Cadets Assigned',
                       title_x=0.5)

    # add shapes to legend
    fig2.add_trace(
        go.Scatter(
            x=[afsc, afsc],
            y=[1, 1],
            mode="lines",
            name="Maximum Capacity",
            line=go.scatter.Line(color="green"),
            showlegend=True)
    )

    fig2.add_trace(
        go.Scatter(
            x=[afsc, afsc],
            y=[1, 1],
            mode="lines",
            name="Minimum Quota",
            line=go.scatter.Line(color="red"),
            showlegend=True)
    )

    # ------------------------------------------------------------------
    # create/update stats2
    fig3 = go.Figure()

    fig3.add_trace(go.Indicator(
        mode="number",
        title='Average Cadet Merit',
        value=df.loc[df['AFSCs'] == afsc]['Cadet Merit Per AFSC'].mean(),
        domain={'row': 0, 'column': 0}
    ))

    fig3.add_trace(go.Indicator(
        mode="number",
        title='Median Cadet Merit',
        value=df.loc[df['AFSCs'] == afsc]['Cadet Merit Per AFSC'].median(),
        domain={'row': 0, 'column': 1}
    ))

    number_overclassified_cadets = 0
    if (df.loc[df['AFSCs'] == afsc]['Total Cadets Assigned'] -
        df.loc[df['AFSCs'] == afsc]['AFSC Max Capacities']).iloc[0] > 0:
        number_overclassified_cadets = (df.loc[df['AFSCs'] == afsc]['Total Cadets Assigned'] -
                                        df.loc[df['AFSCs'] == afsc]['AFSC Max Capacities']
                                        ).iloc[0]

    fig3.add_trace(go.Indicator(
        mode="number",
        title='Cadets Overclassified',
        value=number_overclassified_cadets,
        domain={'row': 1, 'column': 0}
    ))

    number_underclassified_cadets = 0
    if (df.loc[df['AFSCs'] == afsc]['Total Cadets Assigned'] -
        df.loc[df['AFSCs'] == afsc]['AFSC Min Quotas']).iloc[0] < 0:
        number_underclassified_cadets = (df.loc[df['AFSCs'] == afsc]['AFSC Min Quotas'] -
                                         df.loc[df['AFSCs'] == afsc]['Total Cadets Assigned']
                                         ).iloc[0]

    fig3.add_trace(go.Indicator(
        mode="number",
        title='Cadets Underclassified',
        value=number_underclassified_cadets,
        domain={'row': 1, 'column': 1}
    ))

    fig3.update_layout(
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        template={'data': {'indicator': [{
            'mode': "number+delta+gauge",
            'delta': {'reference': 90}}]
        }})

    # ------------------------------------------------------------------
    # create/update the vol/nonvol graph for an afsc
    AFSCs = df.loc[df['AFSCs'] == afsc]
    fig4 = go.Figure(data=[
        go.Bar(name='Mandatory Non-Vol', x=AFSCs['AFSCs'], y=AFSCs['Mandatory Non-Vol']),
        go.Bar(name='Permitted Non-Vol', x=AFSCs['AFSCs'], y=AFSCs['Permitted Non-Vol']),
        go.Bar(name='Desired Non-Vol', x=AFSCs['AFSCs'], y=AFSCs['Desired Non-Vol']),
        go.Bar(name='Ineligible', x=AFSCs['AFSCs'], y=AFSCs['Ineligible']),
        go.Bar(name='Voluntary', x=AFSCs['AFSCs'], y=AFSCs['Voluntary'])
    ])
    fig4.update_layout(barmode='stack',
                       yaxis_title='Number of Cadets Assigned',
                       xaxis_title='AFSC',
                       title_text='Voluntary/Non-Voluntary Partitioning',
                       title_x=0.5)

    return fig1, fig2, fig3, fig4


if __name__ == '__main__':
    app.run_server(debug=False)