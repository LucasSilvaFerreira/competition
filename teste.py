__author__ = 'lucas'
from skimage.transform import resize
from scipy.spatial import distance
import sys
from skimage.data import imread
from skimage import io
import numpy as np
import pandas as pd

from bokeh.charts import Histogram
from bokeh.plotting import show, output_file
from random import shuffle, sample


def hist(df):
    hist = Histogram(df, bins=100, filename="histograms.html", legend=True)
    hist.show()

def main():
    fig = imread('/home/lucas/PycharmProjects/competition/CCLab-MSC-HE.jpg', as_grey=True)
    #fig = np.zeros((2,2), dtype=np.uint8)
    #print fig
    #fig[range(0,200),range(0,200)] = 1 #Desenhando linha na diagonal
    fig = resize(fig, (300, 300))
    hist(pd.DataFrame({'x':[y for x in fig for y in x]}))
    mask = fig >= 0.4
    fig[mask] = 1
    x_len, y_len = fig.shape
    pontos = [[x,y] for x in range(x_len) for y in range(y_len) if fig[x,y] < 0.4]
    print len(pontos)
    hist_distance = [distance.euclidean(loop_x, loop_y) for loop_x in sample(pontos, 200) for loop_y in pontos  if loop_x != loop_y]
    print 'calculo terminado...'
    hist(pd.DataFrame({'x':[distancia for distancia in hist_distance]}))



    # for x in np.nditer(fig):
    #     print x

    io.imshow(fig)

    io.show()

if __name__ == '__main__':
    sys.exit(main())
