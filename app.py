# import psycopg2 as psg
import dash, json, argparse
import dash_bootstrap_components as dbc
import plotly.express as px, pandas as pd, numpy as np
import plotly.graph_objects as go, plotly.io as pio
from dash import dcc, html
from dash.dependencies import Input, Output
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
from train import get_data

AGG_DATA_POINTS = 100
# conn = psg.connect(host='13.53.45.83', database='postgres', user='postgres', password='postgres')


def generate_dump_data(length=100000):
    arr = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35] * (length // 800))
    np.random.shuffle(arr)
    arr = np.repeat(arr.reshape(-1, 1), 100, axis=-1)
    esp = np.random.rand(*arr.shape) / 5
    arr = np.sin(np.arange(0, esp.reshape(-1).shape[0] / 100, 0.01)).reshape(*esp.shape)
    arr = (arr + esp).reshape(-1)
    # return pd.DataFrame({
    #     'prediction': arr,
    #     'temperature': np.sin(np.arange(0, arr.shape[0] / 10, 0.1)),
    #     'pressure': np.sin(np.arange(0, arr.shape[0] / 5, 0.2)),
    #     'idx': np.arange(arr.shape[0]),
    # })
    _, _, _, _, MEAN, STD = get_data('train_new.h5',
                                     True,
                                     True)
    train_inputs, train_cls_label, train_deposit_thickness, train_inner_diameter, _, _ = get_data('val_new.h5',
                                                                                                  True,
                                                                                                  True,
                                                                                                  MEAN,
                                                                                                  STD)
    print(f'MEAN = {MEAN}\nSTD = {STD}')
    train_deposit_thickness = np.sort(train_deposit_thickness.flatten())  # Sort deposit A-Z
    return pd.DataFrame({
        'prediction': train_deposit_thickness + np.random.normal(0, .02, train_deposit_thickness.shape[0]),
        'temperature': np.sin(np.arange(0, train_deposit_thickness.shape[0] / 10, 0.1)),
        'pressure': np.sin(np.arange(0, train_deposit_thickness.shape[0] / 5, 0.2)),
        'idx': np.arange(train_deposit_thickness.shape[0]),
    })

### START of main Dataframe

# df = generate_dump_data() # Uncomment for the real simulator data

# [SOB-ReadDataFromFile]
df = pd.read_csv('values.txt')
df = df.transpose()
df.index = df.index.astype(int) # format indexes as number
df = df.rename(columns={0: "prediction", 1: "temperature", 2: "pressure"})
df['idx'] = df.index
# [EOB-ReadDataFromFile]

### End of main Dataframe


def map_line(path='pipLine.csv'):
    df_map = pd.read_csv(path)
    equinor_df = df_map[(df_map.cmpLongName == 'Equinor Energy AS') & (df_map.pplMedium == 'Oil')].dropna().reset_index(drop=True)
    f = lambda v: np.array([float(j) for i in v[18:-3].split(',') for j in i.strip().split(' ')]).reshape(-1, 2)
    equinor_df['position'] = equinor_df.pipGeometryWKT.apply(f)
    pv = lambda row: [[long, lat, row[0]] for long, lat in row[1]]
    geo_longlat = np.concatenate(equinor_df[['pplBelongsToName', 'position']].apply(pv, axis=1).values)
    longs, lats, names = geo_longlat.transpose()

    fig = go.Figure()

    colors = ['red', 'green', 'blue', 'orange', 'yellow', 'black']
    scl = [0, "rgb(10,255,0)"], [0.4, "rgb(255, 10, 0)"]

    for i in range(len(equinor_df)):
        long = equinor_df.iloc[i].position[:, 0]
        lat = equinor_df.iloc[i].position[:, 1]
        values = np.random.rand(long.shape[0]) / 2.5
        name = equinor_df.iloc[i].pplBelongsToName
        # print(values)

        if i == 3:
            # print(long.shape)
            # fig.add_trace(go.Scattergeo(
            #     locationmode='country names',
            #     lon=long,
            #     lat=lat,
            #     hoverinfo='text',
            #     text=name,
            #     mode='lines',
            #     line=dict(width=2, color=colors[i]),
            # ))
            fig.add_trace(go.Scattergeo(
                locationmode='country names',
                lon=long,
                lat=lat,
                hoverinfo='text',
                text=values,
                marker=dict(
                    color=values,
                    colorscale=scl,
                    #reversescale=True,
                    opacity=0.5,
                    size=5,
                    colorbar=dict(
                        titleside="right",
                        outlinecolor="rgba(68, 68, 68, 0)",
                        ticks="outside",
                        showticksuffix="last",
                        dtick=0.1
                    )
                )
            ))

    fig.update_layout(
        title_text='Equinor Pipeline',
        showlegend=False,
        geo=dict(
            scope='europe',
            projection_type='azimuthal equal area',
            showland=True,
            showlakes=True,
            showcountries=True,
            resolution=50,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
        ),
        # autosize=True,
        # hovermode='closest',
        # margin=dict(t=0, b=0, l=0, r=0),
        # mapbox_zoom=4,
        # mapbox_center_lat=5,
        # template='plotly_dark',
    )

    return fig


def get_app_layout():
    # SIDEBAR_STYLE = {
    #     "padding": "2rem 1rem 2rem 2rem",
    # }
    #
    # sidebar = html.Div(
    #     [
    #         html.H2("ROCSOLE"),
    #         html.Hr(),
    #         dbc.Nav(
    #             [
    #                 dbc.NavLink("Home", href="/", active="exact"),
    #                 dbc.NavLink("Page 1", href="/page-1", active="exact"),
    #                 dbc.NavLink("Page 2", href="/page-2", active="exact"),
    #             ],
    #             vertical=True,
    #             pills=True,
    #         ),
    #     ],
    #     style=SIDEBAR_STYLE,
    # )

    return html.Div([
        # sidebar,
        dbc.Row(dcc.Store('df_store')),

        dbc.Row([
            dbc.Col(
                children=[
                    dbc.Row(html.H1('PIG Dashboard')),
                    dbc.Row(dbc.Col(
                        dcc.RangeSlider(
                            id='datetime-slider',
                            min=df.index[0],
                            max=df.index[-1],
                            value=[df.index[-1] - 1000, df.index[-1]],
                            step=None
                        ),
                    )),
                ],
            ),
            dbc.Col(
                dbc.Row(
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
                        # Allow multiple files to be uploaded
                        multiple=True
                    )
                ),
                width=4,
            ),
        ]),

        # dbc.Row([
        #     dbc.Col(
        #         [
        #             dcc.Graph(id='live-update-graph'),
        #             dcc.Interval(
        #                 id='interval-component',
        #                 interval=1 * 1000,
        #                 n_intervals=0
        #             )
        #         ]  # real time predict graph
        #     )
        # ]),

        dbc.Row([
            dbc.Col(
                dbc.Row(dcc.Graph(id='dt-graph')), style={"height": "50%"}, # handle large amount of data
            )
        ]),

        dbc.Row([
            # dbc.Col(sidebar),
            dbc.Col(
                dbc.Row(dcc.Graph(id='map', figure=map_line()), style={"height": "100%"}),
                width=5,
            ),
            dbc.Col(
                children=[
                    # dbc.Row(html.Div(dcc.Graph(id='map', figure=map_line()))),
                    # dbc.Row(dcc.Graph(id='dt-graph')),
                    dbc.Row(dcc.Graph(id='t-graph'), style={"height": "50%"}),
                    dbc.Row(dcc.Graph(id='p-graph'), style={"height": "50%"}),
                ],
                width=7,
                # style={'margin-left': '7px', 'margin-top': '7px'},
            )
        ]),
    ])


load_figure_template('LUX')
app = dash.Dash(__name__, update_title=None, external_stylesheets=[dbc.themes.LUX])
app.layout = get_app_layout
server = app.server


def generate_chart(df, field, title="", x_title="", y_title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.idx,
                             y=df[field],
                             name="prediction",
                             line_shape='spline',
                             ))
    fig.update_layout(
        title={
            'text': title,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        # xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


# SUBSET DATA FROM SLIDER
@app.callback(Output('df_store', 'data'),
              Input('datetime-slider', 'value'))
def load_subdata(selected_value):
    filtered_df = df.iloc[selected_value[0]:selected_value[1], :]

    # [SOB] Aggregate data if needed
    if filtered_df.shape[0] > AGG_DATA_POINTS:
        n = int(filtered_df.shape[0] / AGG_DATA_POINTS)
        new_d = int(filtered_df.shape[0] / n)
        filtered_df = filtered_df.iloc[: (n * new_d), :]  # prevent overload
        temp_data = filtered_df.values[:, :-1].reshape(new_d, n, -1).mean(1)  # average to the same data point

        filtered_df = pd.DataFrame({
            'prediction': temp_data[:, 0],
            'temperature': temp_data[:, 1],
            'pressure': temp_data[:, 2],
            'idx': np.arange(new_d),
        })
        # print(selected_value[1] - selected_value[0])
        # print(n, new_d)
    # [EOB] Aggregate data if needed

    # print(filtered_df.shape)
    # Format as JSON to return
    filtered_df = filtered_df.to_json()
    return json.dumps(filtered_df)


# UPDATE FIGURES FROM SELECTED DATA
@app.callback(Output('dt-graph', 'figure'),
              Output('t-graph', 'figure'),
              Output('p-graph', 'figure'),
              Input('df_store', 'data'))
def function_square(df_store):
    if df_store is None:
        raise PreventUpdate
    temp_df = pd.read_json(json.loads(df_store))

    return generate_chart(temp_df, "prediction", "DEPOSIT THICKNESS"), \
           generate_chart(temp_df, "temperature", "TEMPERATURE"), \
           generate_chart(temp_df, "pressure", "PRESSURE")


# UPDATE LIVE GRAPH NEW (WITH LIVE PREDICTION)
# @app.callback(Output('live-update-graph', 'figure'),
#               Input('interval-component', 'n_intervals'))
# def function_square(n):
#     cur = conn.cursor()
#     cur.execute('SELECT * FROM public."pig-predictions" ORDER BY "time" DESC LIMIT 60')
#     df_plot = pd.DataFrame(cur.fetchall(), columns=['target', 'predicted', 'date_time', 'request_id'])
#     df_plot = df_plot.iloc[::-1] # Reverse order to match time series
#     cur.close()
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df_plot.date_time,
#                              y=df_plot.target,
#                              name="target",
#                              line_shape='spline',
#                              ))
#     fig.add_trace(go.Scatter(x=df_plot.date_time,
#                              y=df_plot.predicted,
#                              name="predict",
#                              line_shape='spline',
#                              fill='tonexty'))
#     return fig


if __name__ == '__main__':
    # app.run_server(debug=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='host', default='localhost', type=str, help='IP of host')
    parser.add_argument('-p', dest='port', default='5000', type=str, help="Port of Dash")
    parser.add_argument('-d', dest='debug', default=False, type=bool, help="Port of Dash")
    args = parser.parse_args()

    server.run(debug=args.debug, host=args.host, port=args.port)
    # conn.close()
