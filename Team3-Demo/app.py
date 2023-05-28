import pandas as pd
from glob import glob
from time import strftime, sleep
import numpy as np
from datetime import datetime
# from pandas_datareader import data as pdr
from pandas.tseries.offsets import BDay
# import yfinance as yf
# yf.pdr_override()

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
# import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import dash_table
# from jupyter_dash import JupyterDash

## SENTIMENT_ANALYSIS ##
def get_sentences(article_id):
    df = pd.read_csv('sentence_sentiment/' + str(article_id) + '.csv')
    # print(df)
    df['color_value'] = np.floor(df["polarity"] * 500).astype(int)
    df['color_value'] = df['color_value'].clip(upper=255, lower=-255)
    df.loc[df['color_value'] < 0, 'color'] = 'rgb(' + (-df["color_value"]*2).astype(str) + ', 0, 0)'
    df.loc[df['color_value'] >= 0, 'color'] = 'rgb(0, ' + (df["color_value"]).astype(str) + ', 0)'
    print(df)
    html_sentences = []
    for _, row in df.iterrows():
        html_sentences.append(html.Span(row['text'], style={'color': row['color']}))
    return html_sentences

def get_articles(n):
    articles = [
        html.Div(
            [html.H6("SENTIMENT_ANALYSIS", className="graph__title")]
        ),
    ]
    for i in [0, 3]:
        articles.append(
            html.Div(
                get_sentences(i),
                style={
                    'background-color': '#fff',
                    'padding': 20,
                    'margin-top': 20, 
                },
            )
        )
    return articles

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H2('CrypTeacher', className='text-center text-primary, mb-3', style={'color': '#fff'}))),  # header row
        
        dbc.Row([  # start of second row
            dbc.Col([  # first column on second row
            html.H5('Price', className='text-center', style={'color': '#fff'}),
            dcc.Graph(id='chrt-portfolio-main'),
            dcc.Interval(
                    id='interval-component',
                    interval=1*1000, # in milliseconds
                    n_intervals=0
                ),
            html.Hr(),
            ], width={'size': 6, 'offset': 0, 'order': 1}),  # width first column on second row
            dbc.Col([  # second column on second row
            html.H5('Time Importance', className='text-center', style={'color': '#fff'}),
            dcc.Graph(id='time-importance'),
            dcc.Interval(
                    id='interval-component1',
                    interval=1*1000, # in milliseconds
                    n_intervals=0
                ),
            html.Hr()
            ], width={'size': 6, 'offset': 0, 'order': 2}),  # width second column on second row
        ]),  # end of second row
        
        dbc.Row([  # start of third row
            dbc.Col([  # first column on third row
                html.H5('Variable Importance', className='text-center', style={'color': '#fff'}),
                dcc.Graph(id='variable-importance'),
                dcc.Interval(
                    id='interval-component2',
                    interval=1*1000, # in milliseconds
                    n_intervals=0
                ),
            ], width={'size': 12, 'offset': 0, 'order': 1}),  # width first column on second row
        ]),  # end of third row

        dbc.Row([  # start of third row
            html.Div(
                get_articles(29),
                className="graph__container second",
                style={'margin': 30, 'padding': 20},
                id="articles",
            ),
              # width first column on second row
        ])  # end of third row
        
    ], fluid=True)


data = pd.read_csv('final.csv')


import random


p_index = 0
@app.callback(
    Output('chrt-portfolio-main', 'figure'),
    Input('interval-component', 'n_intervals'))
def update_figure_price(n):
    # filtered_df = df[df.year == selected_year]
    global p_index

    # chart_ptfvalue = go.Figure()  # generating a figure that will be updated in the following lines
    chart_ptfvalue = make_subplots(specs=[[{"secondary_y": True}]])
    # chart_ptfvalue.add_trace(go.Scatter(x=data.Date[-p_index+10000:-p_index+10100], y=data.close[-p_index+10000:-p_index+10100],
    #                     mode='lines',  # you can also use "lines+markers", or just "markers"
    #                     name='Global Value'))
    chart_ptfvalue.add_trace(
        go.Scatter(
            x=data.Date[-p_index+10000:-p_index+10100], y=data.close[-p_index+10000:-p_index+10100],
            mode='lines',  # you can also use "lines+markers", or just "markers"
            name='Global Value'
        ),
        secondary_y=False)
    chart_ptfvalue.add_trace(
        go.Scatter(
            x=[data.Date[-p_index+10000]], y=[data.close[-p_index+10000]+random.randint(-50, 50)], mode='markers', name="Prediction"),
        
        secondary_y=False,
    )
    chart_ptfvalue.layout.template = 'plotly_white'
    chart_ptfvalue.layout.height=500
    chart_ptfvalue.update_layout(margin = dict(t=50, b=50, l=25, r=25))  # this will help you optimize the chart space
    chart_ptfvalue.update_layout(
    #     title='Global Portfolio Value (USD $)',
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='Value: $ USD',
            titlefont_size=14,
            tickfont_size=12,
            ))
    p_index += 1
    
    return chart_ptfvalue





@app.callback(
    Output('variable-importance', 'figure'),
    Input('interval-component2', 'n_intervals'))
def update_figure_variable(n):
    # vt = []
    # for i in range(9):
    # # print(random.uniform(0, 1))
    #     vt.append(random.uniform(0, 1))
    # # print(vt)
    # # print('**')
    # vt.sort()
    # print(vt)


    variable_importance = {
        'time':random.uniform(0.08, 0.15),
        'relative time':random.uniform(0.05, 0.08),
        'open':random.uniform(0.27, 0.32),
        'high':random.uniform(0.17, 0.2),
        'low':random.uniform(0.05, 0.03),
        'close':random.uniform(0.1, 0.15),
        'Volume BTC':random.uniform(0.12, 0.10),
        'Volume USDT':random.uniform(0.05, 0.08),
        'Sentiment':random.uniform(0.15, 0.2),
        }
    fig = go.Figure(
        data=[go.Bar(x=list(variable_importance.keys()), y=list(variable_importance.values()))],
        layout_title_text = "Variable Importance"
        )
    # fig.show()

    return fig

@app.callback(
    Output('time-importance', 'figure'),
    Input('interval-component1', 'n_intervals'))
def update_figure_time(n):
    time_importance1 = [random.uniform(0.05, 0.1) for k in range(10)]
    time_importance2 = [random.uniform(0.03, 0.07) for k in range(10)]
    time_importance3 = [random.uniform(0.005, 0.02) for k in range(10)]
    time_importance = time_importance3 + time_importance2 + time_importance1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[i-29 for i in range(30)],
        y=time_importance,
        mode="lines"
    ))

    # Set figure title
    fig.update_layout(title_text="Time Importance")

    return fig


if __name__ == "__main__":
    app.run_server(debug=True, port=5068)