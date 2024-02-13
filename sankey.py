import plotly.graph_objects as go
import pandas as pd

def code_mapping(df, src, targ, *cols):
    # Get distinct labels
    if cols:
        labels = sorted(list(set(list(df[src]) + list(df[targ]) + list(df[cols]))))
    else:
        labels = sorted(list(set(list(df[src]) + list(df[targ]))))

    # Get integer codes
    codes = list(range(len(labels)))

    # Create label to code mapping
    lc_map = dict(zip(labels, codes))

    # Substitute names for codes in dataframe
    df = df.replace({src: lc_map, targ: lc_map})
    return df, labels

def make_sankey(df, src, targ, vals=None, **kwargs):
    """ Create a sankey diagram linking src value to
    targ value with thickness of vals """

    if vals:
        values = df[vals]
    else:
        values = [1] * len(df)

    df, labels = code_mapping(df, src, targ)
    link = {'source':df[src], 'target':df[targ], 'value':values}
    pad = kwargs.get('pad', 50)

    node = {'label':labels, 'pad':pad}
    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)
    return fig

def make_multi_sankey(*df, **kwargs):
    """ Create a multi-layered sankey diagram linking src value to
    targ value with thickness of vals
    REQUIREMENTS: data frame has to have three columns
    """

    for d in df:
        d.columns = ['src', 'targ', 'vals']
    stacks = pd.concat(list(df))

    df, labels = code_mapping(stacks, 'src', 'targ')
    link = {'source':df['src'], 'target':df['targ'], 'value':df['vals']}
    pad = kwargs.get('pad', 50)

    node = {'label':labels, 'pad':pad}
    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)
    return fig
