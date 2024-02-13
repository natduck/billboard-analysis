# import statements
from collections import Counter
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash.dependencies import Input, Output
from sklearn.neighbors import NearestNeighbors
import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as ps
import sankey as sk

# start running dash app with a stylesheet
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
load_figure_template('LUX')

# import full dataset
full_df = pd.read_csv('final_billboard.csv')

# add danceability score categories to df
full_df['Danceability Score'] = 0

# create range list and add
range_list = []
for num in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    num1 = num
    num2 = num + .2
    range_list.append((num1, num2))

# add to the decade column based on begindate
for idx in range(len(full_df)):
    for num_range in range_list:
        score = full_df.loc[idx, 'danceability']
        if score == '[None]':
            full_df.loc[idx, 'Danceability Score'] = 2
        elif num_range[0] <= float(score) <= num_range[1]:
            full_df.loc[idx, 'Danceability Score'] = num_range[0]

# map these range values to descriptors
dance_dict = {0: 'Least Danceable', 0.2: 'Less Danceable', 0.4: 'Danceable',
              0.6: 'More Danceable', 0.8: 'Most Danceable', 2: 'Unknown'}
full_df['Danceability Score'] = full_df['Danceability Score'].map(dance_dict)

# create a means to narrow the set to interactive timeline
# this will be called in all viz things to ensure they use the correct timeline
def df_time_adjust(df, start_year, end_year):
    """
    takes in a dataframe and filters out the years not intended to be included in visuals

    args:
        df (dataframe): full dataframe with everything needed for dashboard analysis
        start_year (int): year the data showing will start at
        end_year (int): year that data showing will end at

    returns:
        final_df(dataframe): an adjusted dataframe with only data pertaining to the selected years
    """
    # creating booleans to sort out appropriate years
    start_bool = df.loc[:, 'Year'] >= int(start_year)
    end_bool = df.loc[:, 'Year'] <= int(end_year)

    # smaller dataframe based on start
    start_df = df.loc[start_bool, :]

    # trim off the end
    final_df = start_df.loc[end_bool, :]

    # add a column that counts the number of times the artist appears in the new time frame
    val_counts = dict(final_df['Artist'].value_counts())
    final_df['artist_count'] = final_df['Artist'].map(val_counts)

    return final_df

def img_html(path):
    """
    takes in an image link and returns it as an html representation of it

    args:
        path(str): a link to an image

    returns:
        an html embedded version of this image"""
    return html.Img(src=path, style= {'width':'100%'})

def play_html(path):
    """
    takes in an audio link and returns it as a playable html element

    args:
        path(str): a link to an audio file
    returns:
        html audio player of the file
    """
    return html.Audio(src=path,controls=True, n_clicks=2, preload="none",style= {'width':'100%'}),

def html_header(text):
    """
    takes in a string and wraps it in the H1 html format

    args:
        text(str): a string of text
    returns:
        an html H1 wrapped version of this text
    """
    return html.H4(text, style= {'width':'100%','font-weight': 'bold'}),

def html_h3(text):
    """
    takes in a string and wraps it in the H3 html format

    args:
        text(str): a string of text

    returns:
        an html H3 wrapped version of this text
    """
    return html.H5(text, style= {'width':'100%'}),

def snapshot_df(df, top_x):
    """
    takes in a dataframe of the top 100 songs and returns the top x
    ranked songs for the time frame in a df with strings formatted
    like html elements

    args:
        df (dataframe): billboard top 100 df for the time frame
        top_x (int): number of top songs to be included in the dataframe

    return:
        return_df(dataframe): a dataframe formatted to include the song data in
            html-style elements to include song images, preview urls, and the like
    """
    # smaller dataframe based on how many of the top songs to include
    first_bool = df.loc[:, 'Rank'].astype(int) <= int(top_x)
    first_df = df.loc[first_bool, :]

    # change the columns into html elements
    first_df['image_url'] = first_df['image_url'].apply(lambda x: img_html(x))
    first_df['preview_url'] = first_df['preview_url'].apply(lambda x: play_html(x))
    first_df['Year/Song'] = first_df['Year'].astype(str) + ' - ' + first_df['Song']
    first_df['Year/Song'] = first_df['Year/Song'].apply(lambda x: html_header(x))
    first_df['Artist'] = first_df['Artist'].apply(lambda x: html_h3(x))
    first_df['ranktext'] = 'Rank: ' + first_df['Rank'].astype(str)
    first_df['rankshow'] = first_df['ranktext'].apply(lambda x: html_h3(x))
    # combining rank, title, preview and artist into one cell
    first_df['show_elements'] = first_df['Year/Song']+first_df['Artist']+first_df['rankshow']+first_df['preview_url']

    # pull out the columns with just the columns of proper format
    return_df = first_df[['image_url', 'show_elements']]
    return return_df

def create_ml_df(df):
    """
    takes in a dataframe and removes the rows that are missing data/cannot be utilized in the model

    args:
        df(dataframe): original dataframe with missing values

    returns:
        ml_df(dataframe): an updated dataframe with all columns needed to perform ML, normalized values
        norm_feat_list(list): a list of the normalized features we will perform ML on later
    """
    # list of all of the columns we will use for data analysis later
    ml_feat_list = ['sentiment_score', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence',
                    'tempo', 'duration_ms', 'time_signature']

    # dropping columns that are missing ml features
    ml_df = df.dropna(thresh=14)

    # dropping one weird row in there
    weird_bool = ml_df.loc[:, 'danceability'] != '[None]'
    ml_df = ml_df.loc[weird_bool, :]

    # adding columns for normalized features
    norm_feat_list = []
    for feat in ml_feat_list:
        col_name = feat + '_norm'
        norm_feat_list.append(col_name)
        ml_df[col_name] = ml_df[feat].astype(float) / ml_df[feat].astype(float).std()

    # resetting the dataframe index
    ml_df = ml_df.reset_index()

    # create a column to select songs from with artist and song name for easier dropdown option
    ml_df['song/artist'] = ml_df['Song'] + ' - ' + ml_df['Artist']

    return ml_df, norm_feat_list

# pulling out the vars from the established
ml_df, norm_feat_list = create_ml_df(full_df)

def generate_table(df):
    """
    creates table from a dataframe that displays html elements as they are

    args:
        df (dataframe): narrowed dataframe we hope to present as a table in the dashboard

    returns:
        Table(dash bootstrap component): table displaying song info with responsive html elements
    """
    return dbc.Table.from_dataframe(df, responsive="sm", borderless=True, header=False, style={'table-layout': 'fixed'})


# the title of the dashboard
dash_title = html.H1("Billboard Top 100 in the 21st century", className="bg-black p-2 mb-2 text-white text-center")

# selecting the years that the user wants to view
start_text = html.P('Select a Start Year')
end_text = html.P('Select an End Year')
start_yr_drop = dcc.Dropdown(full_df['Year'].unique(), id='start_year', value= '2012')
end_yr_drop = dcc.Dropdown(full_df['Year'].unique(), id='end_year', value= '2022')

# top song title and table
table_title = html.H1("Top Songs", className="bg-black p-2 mb-2 text-white text-center")
slider_label = html.H5('select number of top songs per year', className="text-center")
top_x_slider = dcc.Slider(1,5,step=1, marks=None, id='top_x',
                           tooltip={"placement": "bottom", "always_visible": True}, value=3)
display_table = html.Div(id='display_table')


# tabs and elements for the artist analysis portion of the dashboard
# remove unknown from df for the purpose of the dropdown for artist by genre bar chart
unk_bool = full_df.loc[:,'primary_genre'] != 'unknown'
unk_df = full_df.loc[unk_bool, :]
# toggles for bar chart
genre_drop_label= html.P('Select a genre:', className="text-center")
genre_list = list(unk_df['primary_genre'].unique())
genre_drop = dcc.Dropdown(genre_list, id='prime_genre', value='pop')

# bar chart format
artist_tab_div = html.Div([
    dbc.Row([
        dbc.Col([
            html.P(),
            genre_drop_label
        ]),
        dbc.Col([
            html.P(),
            genre_drop
        ]),
    ]),
    dbc.Row([
        dcc.Graph(id = 'artists-genre')
    ])
])

# artist sankey toggle
slider_sankey_label=html.H5('select min artist appearances to be in diagram:', className="text-center")
x_appearances = dcc.Slider(5,15,step=1, marks=None, id='x_appearances',
                           tooltip={"placement": "bottom", "always_visible": True}, value=12)
# artist sankey tab
artist_sankey_div = html.Div([
    html.P(),
    slider_sankey_label,
    x_appearances,
    dcc.Graph(id='artist_sankey')
])
# tabs
artist_tab1 = dbc.Tab(dcc.Graph(id = 'artist_bar'), label = 'Top 10 Artists')
artist_tab2 = dbc.Tab(artist_tab_div, label = 'Top Artists by Genre')
artist_tab3 = dbc.Tab(artist_sankey_div, label = 'Artist Sankey')

artist_tabs = dbc.Card(dbc.Tabs([artist_tab1, artist_tab2, artist_tab3]))

artist_collapse = html.Div(
    [
        dbc.Button(
            "Artist Insights (click to expand)",
            id="artist_collapse-button",
            className="mb-3",
            color="secondary",
            n_clicks=0,
        ),
        dbc.Collapse(
            artist_tabs,
            id="artist_collapse",
            is_open=False,
        ),
    ]
)

# lyrics-related elements of the dashboard
# tabs for the lyric analysis portion of the dashboard
lyric_tab1 = dbc.Tab(dcc.Graph(id='senti_over_time'), label = 'Song Sentiment Over Time')
lyric_tab2 = dbc.Tab(dcc.Graph(id='senti_palette'), label='#1 Songs Sentiment Palette')
lyric_tabs = dbc.Card(dbc.Tabs([lyric_tab1, lyric_tab2]))

lyric_collapse = html.Div(
    [
        dbc.Button(
            "Lyric Insights (click to expand)",
            id="lyric_collapse-button",
            className="mb-3",
            color="secondary",
            n_clicks=0,
        ),
        dbc.Collapse(
            lyric_tabs,
            id="lyric_collapse",
            is_open=False,
        ),
    ]
)

# genre sankey toggle
genre_sankey_label=html.H5('select min artist appearances to be in diagram:', className="text-center")
min_appearances = dcc.Slider(5,15,step=1, marks=None, id='min_appearances',
                           tooltip={"placement": "bottom", "always_visible": True}, value=12)
# genre sankey tab info
genre_sankey_div = html.Div([
    html.P(),
    genre_sankey_label,
    min_appearances,
    dcc.Graph(id='genre_sankey')
])

# genre elements in the dashboard
# tabs for the genre analysis portion of the dashboard
genre_tab1 = dbc.Tab(genre_sankey_div, label = 'Genre Sankey')
genre_tab2 = dbc.Tab(dcc.Graph(id='genre_line'), label = 'Genres Observed over Time')
genre_tab3 = dbc.Tab(dcc.Graph(id='genre_rank'), label = 'Rank by Genre')
genre_tabs = dbc.Card(dbc.Tabs([genre_tab1, genre_tab2, genre_tab3]))

genre_collapse = html.Div(
    [
        dbc.Button(
            "Genre Insights (click to expand)",
            id="genre_collapse-button",
            className="mb-3",
            color="secondary",
            n_clicks=0,
        ),
        dbc.Collapse(
            genre_tabs,
            id="genre_collapse",
            is_open=False,
        ),
    ]
)

# elements for the recommendation system
song_picker = dcc.Dropdown(list(ml_df['song/artist'].unique()), id='song_picker', value= 'Heat Waves - Glass Animals')
match_table = html.Div(id='match_table')

# create the contents to fit inside the ml/recommendation dropdown
ml_section = html.Div([
    dbc.Row([
        dbc.Col([
            html.P('Please select a song to recommend from:', className="text-center"),
            song_picker
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            match_table
        ])
    ]),
])

# create collapsable section for this recommendation with default being open
ml_collapse = html.Div(
    [
        dbc.Button(
            "Song Recommendations",
            id="ml_collapse-button",
            className="mb-3",
            color="secondary",
            n_clicks=0,
        ),
        dbc.Collapse(
            ml_section,
            id="ml_collapse",
            is_open=False,
        ),
    ]
)

# define the layout of the dashboard using dash bootstrap components
app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.P(),
            dash_title,
            dbc.Row([
                dbc.Col([
                    html.P(),
                    start_text,
                    start_yr_drop,
                    html.P()
                ], width=4),
                dbc.Col([
                    html.P(),
                    end_text,
                    end_yr_drop,
                    html.P()
                ], width=4)
            ], style={"margin-left": "3px", "margin-right": "3px", "border":"5px black solid"}),
            dbc.Row([
                html.P(),
                # artist_label,
                artist_collapse
            ], style={'margin-top': "11px", "margin-left": "3px", "margin-right": "3px", "border":"5px black solid"}),
            dbc.Row([
                html.P(),
                # artist_label,
                lyric_collapse
            ], style={'margin-top': "11px", "margin-left": "3px", "margin-right": "3px", "border":"5px black solid"}),
            dbc.Row([
                html.P(),
                # artist_label,
                genre_collapse
            ], style={'margin-top': "11px", "margin-left": "3px", "margin-right": "3px", "border":"5px black solid"}),
            dbc.Row([
                html.P(),
                ml_collapse
            ], style={'margin-top': "11px", "margin-left": "3px", "margin-right": "3px", "border":"5px black solid"})
        ], width=8),
        dbc.Col([
            dbc.Row([
                table_title,
                html.P(),
                slider_label,
                top_x_slider,
                display_table
            ], style={'margin-top': "11px","margin-left": "3px", "margin-right": "3px", "border":"5px black solid"})
        ], width=4)
    ], style={'margin-top': "11px","margin-left": "3px", "margin-right": "3px"})
])

@app.callback(
    Output('display_table', 'children'),
    [Input('start_year', 'value'),
    Input('end_year', 'value'),
    Input('top_x', 'value')]
)
def callback_display_table(start_year, end_year, top_x):
    """
    calls back the display table

    args:
        start_year(int): start year to narrow the dataframe
        end_year(int): end year to narrow the dataframe
        top_x(int): top number of songs displaying on the right hand side

    returns:
        display_table: an element of the dashboard that shows the top x songs
            and info about them
    """
    year_df = df_time_adjust(full_df, start_year, end_year)
    small_df = snapshot_df(year_df, top_x)
    display_table = generate_table(small_df)
    return display_table

# artist collapsable section
@app.callback(
    Output("artist_collapse", "is_open"),
    [Input("artist_collapse-button", "n_clicks")],
    [State("artist_collapse", "is_open")])
def toggle_collapse(n, is_open):
    """
    toggles collapse for artist section

    args:
        n(int): number of clicks
        is_open(boolean): state of collapsible section

    returns:
        is_open(boolean): tells the collapsible section whether it is open
    """
    if n:
        return not is_open
    return is_open

# ml collapsable section
@app.callback(
    Output("ml_collapse", "is_open"),
    [Input("ml_collapse-button", "n_clicks")],
    [State("ml_collapse", "is_open")])
def toggle_collapse(n, is_open):
    """
    toggles collapse for machine learning section

    args:
        n(int): number of clicks
        is_open(boolean): state of collapsible section

    returns:
        is_open(boolean): tells the collapsible section whether it is open
    """
    if n+1: # want it to start open
        return not is_open
    return is_open

# lyric collapsable section
@app.callback(
    Output("lyric_collapse", "is_open"),
    [Input("lyric_collapse-button", "n_clicks")],
    [State("lyric_collapse", "is_open")])
def toggle_collapse(n, is_open):
    """
    toggles collapse of the lyric section

    args:
        n(int): number of clicks
        is_open(boolean): state of collapsible section

    returns:
        is_open(boolean):: tells the collapsible section whether it is open
    """
    if n:
        return not is_open
    return is_open


# genre collapsable section
@app.callback(
    Output("genre_collapse", "is_open"),
    [Input("genre_collapse-button", "n_clicks")],
    [State("genre_collapse", "is_open")])
def toggle_collapse(n, is_open):
    """
    toggles collapse of genre section

    args:
        n(int): number of clicks
        is_open(boolean): state of collapsible section

    returns:
        is_open(boolean): tells the collapsible section whether it is open
    """
    if n:
        return not is_open
    return is_open

def keep_repeat_results(df, k):
    """
    helper functions filters out artists that do not appear a minimum
    of k times in a dataframe

    args:
        df (DataFrame): original dataframe
        k (int): minimum number of appearances in the dataframe to be included in the viz

    returns:
        final_df(DataFrame): dataframe with less common artists filtered out
    """
    # filter out the ones that aren't there more than k times
    min_bool = df.loc[:, 'artist_count'] > k

    final_df = df.loc[min_bool, :]

    return final_df

@app.callback(
    Output('artist_bar', 'figure'),
    Input('start_year', 'value'),
    Input('end_year', 'value')
)
def bar_chart_top_10(start_year, end_year):
    """
    creates bar chart for the top 10 songs

    args:
        start_year (int): start year
        end_year (int): end year

    returns:
        count_bar(Figure): bar chart
    """
    filtered_df = df_time_adjust(full_df, start_year, end_year)
    # Group Artists and count the number of appearances in the top 100
    count_df = filtered_df.groupby(filtered_df["Artist"])["Year"].count().reset_index()

    # Rename Year column to count
    count_df.rename(columns={"Year": "Count"}, inplace=True)

    # Sort column to descending order
    count_df = count_df.sort_values("Count", ascending=False)[:10]
    count_df = count_df.sort_values("Count", ascending=True)

    # Make and show the bar chart
    count_bar = px.bar(data_frame=count_df, x=count_df["Count"], y=count_df["Artist"],
                       orientation="h")

    return count_bar

@app.callback(
    Output('artist_sankey', 'figure'),
    Input('start_year', 'value'),
    Input('end_year', 'value'),
    Input('x_appearances', 'value')
)
def sankey_viz(start_year, end_year, k):
    """
    Creates and plots the sankey diagram from src to year to targ

    args:
        start_year (int): start year
        end_year (int): end year
        k(int): total limit

    returns:
        Sankey diagram
    """

    # create source and target column names and obtain filtered dataframe
    src = 'Artist'
    targ = 'Danceability Score'

    df = df_time_adjust(full_df, start_year, end_year)
    df = keep_repeat_results(df, int(k))

    df = df.groupby([src, targ]).size().reset_index(name='count')

    # return diagram
    return sk.make_sankey(df, src, targ, 'count')


@app.callback(
    Output('senti_over_time', 'figure'),
    Input('start_year', 'value'),
    Input('end_year', 'value')
)
def senti_bar(start_year, end_year):
    """
    shows the relationship between top 100 album types, artists, and songs
    only shows songs from artists who show up at least k times in the dataframe

    args:
        start_year(int): start year to narrow the dataframe
        end_year(int): end year to narrow the dataframe

    returns:
        fig(Figure): a bar graph showing average sentiment by year
    """
    # filter out years
    final_df = df_time_adjust(full_df, start_year, end_year)

    # make a smaller dataframe with counts per year by genre
    new_dict = final_df.groupby('Year')['sentiment_score']
    graph_data = pd.DataFrame(new_dict)
    graph_data = graph_data.rename(columns={0: 'Year', 1: 'score_list'})
    graph_data = graph_data.reset_index()
    graph_data['Average Sentiment'] = graph_data['score_list'].apply(lambda x: x.sum() / len(x))

    # make line graph
    fig = px.bar(graph_data, 'Year', 'Average Sentiment')

    # return fig
    return fig


@app.callback(
    Output('senti_palette', 'figure'),
    Input('start_year', 'value'),
    Input('end_year', 'value')
)
def sentiment_palette(start_year, end_year, top_x=1):
    """
    a color palette representation of the songs' sentiment scores
    (this is a modified bar graph to serve a different purpose)

    args:
        start_year(int): start year to narrow the dataframe
        end_year(int): end year to narrow the dataframe
        top_x(int): for this purpose, we will only be plotting the number 1 song
    returns:
        figure: a color palette-type visualization which shows the texts in sentiment score order
            in a color that might be associated with negative or positive emotion
    """
    # pull in the data
    new_df = df_time_adjust(full_df, start_year, end_year)
    # smaller dataframe based on how many of the top songs to include
    first_bool = new_df.loc[:, 'Rank'].astype(int) <= int(top_x)
    first_df = new_df.loc[first_bool, :]

    first_df['one height'] = 1

    first_df = first_df.sort_values(by=['sentiment_score'])

    # plot as a bar chart with some modifications to look like a color palette using a color bar
    # the inferno has dark colors for low scores and yellows for highs
    # so it is perfect to associate with sentiment
    fig = px.bar(first_df, 'one height', 'Song', color='sentiment_score', text='Song', orientation='h',
                 hover_data={'one height': False, 'Artist': True, 'Year': True, 'Rank': True,},
                 labels={'sentiment_score': 'Sentiment Score'})
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.update_traces(textfont_size=20, insidetextanchor="start")
    return fig

@app.callback(
    Output('artists-genre', 'figure'),
    Input('start_year', 'value'),
    Input('end_year', 'value'),
    Input('prime_genre', 'value')
)
def bar_top_10_by_genre(start_year, end_year, prime_genre):
    """
    creates a bar chart of the top 10 songs by genre

    args:
        start_year(int): start year to narrow the dataframe
        end_year(int): end year to narrow the dataframe
        prime_genre(string): primary genre

    returns:
        count_bar(Figure): a bar graph of the top 10

    """
    # Get filtered df for selected years and genre
    filtered_df = df_time_adjust(full_df, start_year, end_year)
    genre_bool = filtered_df.loc[:,"primary_genre"] == prime_genre
    genre_df = filtered_df.loc[genre_bool, :]

    # Group Artists and count the number of appearances in the top 100
    count_df = genre_df.groupby(filtered_df["Artist"])["Year"].count().reset_index()

    # Rename Year column to count
    count_df.rename(columns={"Year": "Count"}, inplace=True)

    # Sort column to descending order
    top_df = count_df.sort_values("Count", ascending=False)[:10]
    sort_df = top_df.sort_values("Count", ascending=True)

    # Make and show the bar chart
    count_bar = px.bar(sort_df, "Count", "Artist",
                       orientation="h",
                       title=f"Top 10 {prime_genre.title()} Artists {start_year} - {end_year}")
    return count_bar


@app.callback(
    Output('genre_line', 'figure'),
    Input('start_year', 'value'),
    Input('end_year', 'value')
)
def genre_line(start_year, end_year):
    """
    shows the relationship between top 100 album types, artists, and songs
    only shows songs from artists who show up at least k times in the dataframe

    args:
        start_year(int): start year to narrow the dataframe
        end_year(int): end year to narrow the dataframe

    returns:
        fig(multi-line graph): shows the relationship and trends of genres year by year
    """
    # filter out years
    final_df = df_time_adjust(full_df, start_year, end_year)

    # filter out unknown genres
    unk_bool = final_df.loc[:,'primary_genre'] != 'unknown'
    final_df = final_df.loc[unk_bool, :]

    # make a smaller dataframe with counts per year by genre
    new_dict = final_df.groupby('Year')['primary_genre'].value_counts()
    graph_data = pd.DataFrame(new_dict)
    graph_data = graph_data.rename(
        columns={'Year': 'Year', 'primary_genre': 'primary_genre', 'primary_genre': 'genre_counts'})
    graph_data = graph_data.reset_index()

    # make the multi-line chart
    fig = px.line(graph_data, x='Year', y='genre_counts', color='primary_genre',
                  labels={'genre_counts':'Total Songs in Top 100',
                          'primary_genre':'Primary Genre'})

    # return fig
    return fig


@app.callback(
    Output('genre_rank', 'figure'),
    Input('start_year', 'value'),
    Input('end_year', 'value')
)
def rank_by_genre(start_year, end_year):
    """
    shows the relationship between top 100 album types, artists, and songs
    only shows songs from artists who show up at least k times in the dataframe

    args:
        start_year(int): start year to narrow the dataframe
        end_year(int): end year to narrow the dataframe

    returns:
        fig(scatter plot): a scatter plot where color is sorted by genre to show
            the prominence of different genres and where in the charts they fall
    """
    # filter out years for new df
    final_df = df_time_adjust(full_df, start_year, end_year)

    # scatter for year rank and genre
    fig = px.scatter(final_df, 'Rank', 'Year', color='primary_genre',
                     labels={'primary_genre':'Primary Genre'})
    # return fig
    return fig

@app.callback(
    Output('genre_sankey', 'figure'),
    Input('start_year', 'value'),
    Input('end_year', 'value'),
    Input('min_appearances', 'value')
)
def sankey_viz(start_year, end_year, k):
    """
    Creates and plots the sankey diagram from src to year to targ

    args:
        start_year(int): start year to narrow the dataframe
        end_year(int): end year to narrow the dataframe
        k (int): min number of appearances needed to be in the threshold and included in viz

    returns:
        Sankey diagram
    """

    src = 'primary_genre'
    targ = 'Artist'
    df = df_time_adjust(full_df, start_year, end_year)
    df = keep_repeat_results(df, int(k))

    s = df.groupby([src, 'Year']).size().reset_index(name='count')
    s['Year'] = s['Year'].astype('string')

    t = df.groupby(['Year', targ]).size().reset_index(name='count')
    t['Year'] = t['Year'].astype('string')

    return sk.make_multi_sankey(s, t)


@app.callback(
    Output('match_table', 'children'),
    Input('song_picker', 'value')
)
def match_similar_songs(song_picker):
    """
    runs a k nearest neighbors algorithm to return the top 5 most similar songs to the users' favorite

    args:
        song_picker(dropdown): a dropdown menu of all songs in the full dataset
            (that have info matched from the spotify api)

    returns:
        table(dash bootstrap component): a responsive table with song name, artist, album cover, and preview

    """
    # create the df for the machine learning to take place
    ml_df, norm_feat_list = create_ml_df(full_df)

    # weigh the more influential variables more heavily
    heavier_weights = ['sentiment_score_norm','danceability_norm', 'valence']
    lighter_weights = ['key_norm', 'speechiness_norm', 'liveness_norm','duration_ms_norm', 'energy_norm']

    for feat in heavier_weights:
        ml_df[feat] = ml_df[feat]*2

    for feat in lighter_weights:
        ml_df[feat] = ml_df[feat]*.5

    # calling in the columns to use to find nearest neighbors
    X = np.array(ml_df[norm_feat_list])

    # run nearest neighbors fit
    knn = NearestNeighbors(n_neighbors=len(ml_df.index))
    knn.fit(X)

    # bring in user selection
    song_idx_list = ml_df.index[ml_df['song/artist'] == song_picker].tolist()
    idx_1 = song_idx_list[0]
    user_data = np.array(ml_df.loc[idx_1, norm_feat_list]).reshape(1, -1)
    nn_idx_array = knn.kneighbors(user_data, return_distance=False)

    # now we have an ordered array of the nearest neighbors to the users' selection
    rank_order = nn_idx_array[0]
    ordered_df = ml_df.loc[rank_order, :]
    ordered_df = ordered_df.drop_duplicates(subset=norm_feat_list, keep='first')

    # only want to show the top 5 and the first result is going to be the original song
    show_df = ordered_df.iloc[1:6, :]

    # make it presentable with song previews
    show_df['image_url'] = show_df['image_url'].apply(lambda x: img_html(x))
    show_df['preview_url'] = show_df['preview_url'].apply(lambda x: play_html(x))
    show_df['Song'] = show_df['Song'].apply(lambda x: html_header(x))
    show_df['Artist'] = show_df['Artist'].apply(lambda x: html_h3(x))
    show_df['show_elements'] = show_df['Song'] + show_df['Artist'] + show_df['preview_url']
    return_df = show_df[['image_url', 'show_elements']]
    return_df = return_df.transpose()

    return generate_table(return_df)

# deploy dashboard
if __name__ == '__main__':
    app.run_server(debug=True)


