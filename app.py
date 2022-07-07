import psycopg2 as psg
import dash, json
import dash_bootstrap_components as dbc
import plotly.express as px, pandas as pd, numpy as np
import plotly.graph_objects as go, plotly.io as pio
from dash import dcc, html
from dash.dependencies import Input, Output
from dash_bootstrap_templates import load_figure_template


# conn = psg.connect(host='10.8.8.105', database='postgres', user='postgres', password='postgres')
# cur = conn.cursor()
# cur.execute('SELECT * FROM public."pig-predictions"')
# df = pd.DataFrame(cur.fetchall(), columns=['target', 'predicted', 'date_time', 'request_id'])
# cur.close()
# conn.close()


def generate_dump_data(length=100000):
    arr = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35] * (length // 800))
    np.random.shuffle(arr)
    arr = np.repeat(arr.reshape(-1, 1), 100, axis=-1)
    esp = np.random.rand(*arr.shape) / 5
    arr = np.sin(np.arange(0, esp.reshape(-1).shape[0] / 100, 0.01)).reshape(*esp.shape)
    arr = (arr + esp).reshape(-1)
    return pd.DataFrame({
        'prediction': arr,
        'temperature': np.sin(np.arange(0, arr.shape[0] / 10, 0.1)),
        'pressure': np.sin(np.arange(0, arr.shape[0] / 5, 0.2)),
        'idx': np.arange(arr.shape[0]),
    })


df = generate_dump_data()


def map_line(path='C:/Users/BruceNguyen/Downloads/pipLine.csv'):
    df = pd.read_csv(path)
    equinor_df = df[(df.cmpLongName == 'Equinor Energy AS') & (df.pplMedium == 'Oil')].dropna().reset_index(drop=True)
    f = lambda v: np.array([float(j) for i in v[18:-3].split(',') for j in i.strip().split(' ')]).reshape(-1, 2)
    equinor_df['position'] = equinor_df.pipGeometryWKT.apply(f)
    pv = lambda row: [[long, lat, row[0]] for long, lat in row[1]]
    geo_longlat = np.concatenate(equinor_df[['pplBelongsToName', 'position']].apply(pv, axis=1).values)
    longs, lats, names = geo_longlat.transpose()

    fig = go.Figure()

    colors = ['red', 'green', 'blue', 'orange', 'yellow', 'black']
    for i in range(len(equinor_df)):
        long = equinor_df.iloc[i].position[:, 0]
        lat = equinor_df.iloc[i].position[:, 1]
        name = equinor_df.iloc[i].pplBelongsToName

        fig.add_trace(go.Scattergeo(
            locationmode='country names',
            lon=long,
            lat=lat,
            hoverinfo='text',
            text=name,
            mode='lines',
            line=dict(width=1, color=colors[i]),
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
        autosize=True,
        hovermode='closest',
        margin=dict(t=0, b=0, l=0, r=0),
        mapbox_zoom=4,
        mapbox_center_lat=5,
        # template='plotly_dark',
    )

    return fig


def get_app_layout():
    SIDEBAR_STYLE = {
        "padding": "2rem 1rem 2rem 2rem",
    }

    sidebar = html.Div(
        [
            html.H2("ROCSOLE"),
            html.Hr(),
            dbc.Nav(
                [
                    dbc.NavLink("Home", href="/", active="exact"),
                    dbc.NavLink("Page 1", href="/page-1", active="exact"),
                    dbc.NavLink("Page 2", href="/page-2", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    return html.Div([
        # sidebar,
        dbc.Row(
            dbc.Col(
                children=[
                    html.H1('PIG Dashboard'),
                    dbc.Row(
                        dcc.RangeSlider(
                            id='datetime-slider',
                            min=df.index[0],
                            max=df.index[-1],
                            value=[df.index[-1] - 1000, df.index[-1]],
                            step=None
                        ),
                    ),
                ],
            ),
        ),
        dbc.Row(
            dbc.Col(dbc.Row(dcc.Graph(id='dt-graph')), style={"height": "50%"},)
        ),
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


def generate_chart(df, field):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.idx,
                             y=df[field],
                             name="prediction",
                             line_shape='spline',
                             ))
    # fig.update_layout(template='plotly_dark')
    return fig


@app.callback(Output('dt-graph', 'figure'),
              Output('t-graph', 'figure'),
              Output('p-graph', 'figure'),
              Input('datetime-slider', 'value'))
def function_square(selected_value):
    filterd_df = df.iloc[selected_value[0]:selected_value[1], :]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filterd_df.index,
                             y=filterd_df.prediction,
                             name="prediction",
                             line_shape='spline',
                             ))
    return generate_chart(filterd_df, "prediction"), generate_chart(filterd_df, "temperature"), generate_chart(
        filterd_df, "pressure")


if __name__ == '__main__':
    # app.run_server(debug=True)
    server.run(debug=True)
