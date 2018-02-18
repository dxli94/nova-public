#import plotly.plotly as py
import plotly.offline as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF

import numpy as np
import pandas as pd
import scipy


class PlottingHelper:
    images = []

    def __init__(self, images):
        self.images = images

    def PolygonSort(self, corners):
        n = len(corners)
        cx = float(sum(x for x, y in corners)) / n
        cy = float(sum(y for x, y in corners)) / n
        cornersWithAngles = []
        for x, y in corners:
            an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
            cornersWithAngles.append((x, y, an))
        cornersWithAngles.sort(key=lambda tup: tup[2])
        return cornersWithAngles  # map(lambda (x, y, an): (x, y), cornersWithAngles)

    def PolygonArea(self, corners):
        n = len(corners)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        area = abs(area) / 2.0
        return area

    def Print(self):
        trace_data = []
        for image in self.images:
            corners_sorted = self.PolygonSort(image.vertices)
            area = self.PolygonArea(corners_sorted)

            x = [corner[0] for corner in corners_sorted]
            y = [corner[1] for corner in corners_sorted]

            annotation = go.Annotation(
                x=5.5,
                y=8.0,
                text='The area of the polygon is approximately %s' % (area),
                showarrow=False
            )

            trace = go.Scatter(
                x=x,
                y=y,
                mode='markers',
                fill='tozeroy',
            )


            trace_data.append(trace)
        layout = go.Layout(
            annotations=[annotation],
            xaxis=dict(
                range=[-1, 3]
            ),
            yaxis=dict(
                range=[-1, 2]
            )
        )
        fig = go.Figure(data=trace_data, layout=layout)

        py.plot(fig)

if __name__ == '__main__':
    corners = [(0.0, 1.030454533953517), (0.0, 0.0), (2.060909067907034, 0.0), (2.060909067907034, 1.030454533953517)]
    plHelper = PlottingHelper(corners)
    corners_sorted = plHelper.PolygonSort(corners)
    area = plHelper.PolygonArea(corners_sorted)

    x = [corner[0] for corner in corners_sorted]
    y = [corner[1] for corner in corners_sorted]

    annotation = go.Annotation(
        x=5.5,
        y=8.0,
        text='The area of the polygon is approximately %s' % (area),
        showarrow=False
    )

    trace1 = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        fill='tozeroy',
    )

    layout = go.Layout(
        annotations=[annotation],
        xaxis=dict(
            range=[-1, 9]
        ),
        yaxis=dict(
            range=[-1, 12]
        )
    )

    trace_data = [trace1]
    fig = go.Figure(data=trace_data, layout=layout)

    py.plot(fig)