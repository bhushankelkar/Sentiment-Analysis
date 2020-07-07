#from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify
import atexit
import os
import re
import json
import dash
import plotly
import dash_core_components as dcc
import dash_html_components as html 
import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import sys
#from app import app
#from tabs import sidepanel, tab1, tab2
#from database import transforms

external_stylesheets = ['https://codepen.io/anon/pen/mardKv.css']
theme =  {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}

app = dash.Dash(__name__,external_stylesheets = [dbc.themes.BOOTSTRAP])
server = app.server
app.config.suppress_callback_exceptions = True
'''
db_name = 'mydb'
client = None
db = None

if 'VCAP_SERVICES' in os.environ:
    vcap = json.loads(os.getenv('VCAP_SERVICES'))
    print('Found VCAP_SERVICES')
    if 'cloudantNoSQLDB' in vcap:
        creds = vcap['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)
elif "CLOUDANT_URL" in os.environ:
    client = Cloudant(os.environ['CLOUDANT_USERNAME'], os.environ['CLOUDANT_PASSWORD'], url=os.environ['CLOUDANT_URL'], connect=True)
    db = client.create_database(db_name, throw_on_exists=False)
elif os.path.isfile('vcap-local.json'):
    with open('vcap-local.json') as f:
        vcap = json.load(f)
        print('Found local VCAP_SERVICES')
        creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)'''

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))

#@app.route('/')
#def root():
#    return app.send_static_file('index.html')

#4 Dataframes for 4 lockdown periods
df1 = pd.read_csv('lock1.csv',encoding='latin')
df2 = pd.read_csv('lock2.csv',encoding='latin')
df3 = pd.read_csv('lock3.csv',encoding='latin')
df4 = pd.read_csv('lock4.csv',encoding='latin')

# Use this for hashtag extract
'''
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

HT_regular = hashtag_extract(df['text'][df['labels'] == 1])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(df['text'][df['labels'] == 0])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])
print(HT_regular,file=sys.stderr)
print(HT_negative,file=sys.stderr)


#positive hashtags
#a = nltk.FreqDist(HT_regular)
#d = pd.DataFrame({'Hashtag': list(a.keys()),
#                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
#d = d.nlargest(columns="Count", n = 10) 
#plt.figure(figsize=(22,10))
#ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
#ax.set(ylabel = 'Count')
#plt.show()

#negative hastags funtion will come over here
#b = nltk.FreqDist(HT_negative)
#e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
#e = e.nlargest(columns="Count", n = 10)   
#plt.figure(figsize=(16,5))
#ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
#ax.set(ylabel = 'Count')
#plt.show()'''

#Use this for wordcount
'''
def word_count(sentence):
    return len(sentence.split())


df['word count'] = df['text'].apply(word_count)
x = df['word count'][df.labels == 1]
y = df['word count'][df.labels == 0]
print(x,file=sys.stderr)
print(y,file=sys.stderr)
#plt.figure(figsize=(12,6))
#plt.xlim(0,45)
#plt.xlabel('word count')
#plt.ylabel('frequency')
#g = plt.hist([x, y], color=['r','b'], alpha=0.5, label=['positive','negative'])
#plt.legend(loc='upper right')
#Till here'''

app.layout =  html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Live Tweets', value='tab-1'),
        dcc.Tab(label='Lockdown Analysis', value='tab-2'),
        dcc.Tab(label='Lockdown 1.0', value='tab-3'),
        dcc.Tab(label='Lockdown 2.0', value='tab-4'),
        dcc.Tab(label='Lockdown 3.0', value='tab-5'),
        dcc.Tab(label='Lockdown 4.0', value='tab-6'),
    ]),
    html.Div(id='tabs-content')
])


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
       return html.Div([
        html.Div([
        dcc.Graph(
            id="scatter_chart",
            figure={
            'data':[
            go.Scatter(
                x=df1.text,
                y=df1.labels,
                mode='markers'
                

                )

            ],
            'layout':go.Layout(
                title ="Scatterplot",
                xaxis = {'title': 'Tweet'},
                yaxis = {'title': 'label'},
                
        

                )
            }


            )


        ],style={'width':'33.33%','display':'inline-block','padding':'0 0 0 20'}),
        html.Div([
        dcc.Graph(
            id="pie_chart",
            figure={
            'data':[
            go.Pie(
                labels=['positives','negatives','neutrals'],
                values=[500,460,620],
                name="Sentiment Analysis",
                hoverinfo='label+percent',
                textinfo='value'
                

                )

            ],
            'layout':go.Layout(
                title ="Pie Chart",
                
                
        

                )
            }


            )


        ],style={'width':'33.33%','display':'inline-block','padding':'0 0 0 20'}),

        html.Div([
        dcc.Graph(
            id="pie_chart",
            figure={
            'data':[
            go.Pie(
                labels=['positives','negatives','neutrals'],
                values=[500,460,620],
                name="Sentiment Analysis",
                hoverinfo='label+percent',
                textinfo='value'
                

                )

            ],
            'layout':go.Layout(
                title ="Pie Chart",
                
                
        

                )
            }


            )


        ],style={'width':'33.33%','display':'inline-block','padding':'0 0 0 20'}),

        html.Div([
        dcc.Graph(
            id="pie_chart",
            figure={
            'data':[
            go.Pie(
                labels=['positives','negatives','neutrals'],
                values=[500,460,620],
                name="Sentiment Analysis",
                hoverinfo='label+percent',
                textinfo='value'
                

                )

            ],
            'layout':go.Layout(
                title ="Pie Chart",
                
                
        

                )
            }


            )


        ],style={'width':'33.33%','display':'inline-block','padding':'0 0 0 20'}),

        html.Div([
        dcc.Graph(
            id="pie_chart",
            figure={
            'data':[
            go.Pie(
                labels=['positives','negatives','neutrals'],
                values=[500,460,620],
                name="Sentiment Analysis",
                hoverinfo='label+percent',
                textinfo='value'
                

                )

            ],
            'layout':go.Layout(
                title ="Pie Chart",
                
                
        

                )
            }


            )


        ],style={'width':'33.33%','display':'inline-block','padding':'0 0 0 20'}),

        ])

        
    elif tab == 'tab-2':
       return html.Div([
        dcc.Graph(
            id="scatter_chart",
            figure={
            'data':[
            go.Scatter(
                x=df1.text,
                y=df1.labels,
                mode='markers'
                )

            ],
            'layout':go.Layout(
                title ="Scatterplot",
                xaxis = {'title': 'Tweet'},
                yaxis = {'title': 'label'}

                )
            }


            )


        ])
    elif tab == 'tab-3':
       return html.Div([
        dcc.Graph(
            id="scatter_chart",
            figure={
            'data':[
            go.Scatter(
                x=df1.text,
                y=df1.labels,
                mode='markers'
                )

            ],
            'layout':go.Layout(
                title ="Scatterplot",
                xaxis = {'title': 'Tweet'},
                yaxis = {'title': 'label'}

                )
            }


            )


        ])
    elif tab == 'tab-4':
       return html.Div([
        dcc.Graph(
            id="scatter_chart",
            figure={
            'data':[
            go.Scatter(
                x=df2.text,
                y=df2.labels,
                mode='markers'
                )

            ],
            'layout':go.Layout(
                title ="Scatterplot",
                xaxis = {'title': 'Tweet'},
                yaxis = {'title': 'label'}

                )
            }


            )


        ])
    elif tab == 'tab-5':
       return html.Div([
        dcc.Graph(
            id="scatter_chart",
            figure={
            'data':[
            go.Scatter(
                x=df3.text,
                y=df3.labels,
                mode='markers'
                )

            ],
            'layout':go.Layout(
                title ="Scatterplot",
                xaxis = {'title': 'Tweet'},
                yaxis = {'title': 'label'}

                )
            }


            )


        ])
    elif tab == 'tab-6':
       return html.Div([
        dcc.Graph(
            id="scatter_chart",
            figure={
            'data':[
            go.Scatter(
                x=df4.text,
                y=df4.labels,
                mode='markers'
                )

            ],
            'layout':go.Layout(
                title ="Scatterplot",
                xaxis = {'title': 'Tweet'},
                yaxis = {'title': 'label'}

                )
            }


            )


        ])
# /* Endpoint to greet and add a new visitor to database.
# * Send a POST request to localhost:8000/api/visitors with body
# * {
# *     "name": "Bob"
# * }
# */
'''
@app.route('/api/visitors', methods=['GET'])
def get_visitor():
    if client:
        return jsonify(list(map(lambda doc: doc['name'], db)))
    else:
        print('No database')
        return jsonify([])'''

# /**
#  * Endpoint to get a JSON array of all the visitors in the database
#  * REST API example:
#  * <code>
#  * GET http://localhost:8000/api/visitors
#  * </code>
#  *
#  * Response:
#  * [ "Bob", "Jane" ]
#  * @return An array of all the visitor names
#  */
'''
@app.route('/api/visitors', methods=['POST'])
def put_visitor():
    user = request.json['name']
    data = {'name':user}
    if client:
        my_document = db.create_document(data)
        data['_id'] = my_document['_id']
        return jsonify(data)
    else:
        print('No database')
        return jsonify(data)

@atexit.register
def shutdown():
    if client:
        client.disconnect()'''
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=port, debug=True)
