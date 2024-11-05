import pandas as pd
import plotly.express as px


def plot_scatter(data: pd.DataFrame(), x: str, y: str):

    fig = px.scatter(data, x=x, y=y, labels={x: x, y: y})
    fig.show()
    return


