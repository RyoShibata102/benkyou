from matplotlib.pyplot import figimage
import plotly.graph_objects as go
import plotly.io as pio

### Simple bar plot example
graph = go.Figure(data=go.Bar(y=[42, 153, 912]),
                  layout=go.Layout(title=go.layout.Title(text="Example Plot")))
graph.show()

#graph.write_html("BLABLAPLOT.html", auto_open=True)
# graph.to_dict()
################################################################################

fig_data = {
    "layout": {
        "title": {
            "text": "Example Plot"
        }
    },
    "data": [
        {
            "type": "bar",
            "x": [0, 1, 2],
            "y": [123, 345, 126]
        }
    ]
}

pio.show(fig_data)

# graph_fig = go.Figure(fig_data)

################################################################################