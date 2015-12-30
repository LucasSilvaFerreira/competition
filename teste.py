__author__ = 'lucas'
from tqdm import tqdm
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
from sklearn import mixture
from skimage.draw import circle

def distancia_func(a, b):
    print a, b, distance.euclidean(a, b)
    exit()
    return distance.euclidean(a, b)

def hist(df):
    hist = Histogram(df, bins=50, filename="histograms.html", legend=True)
    hist.show()


def point_to_key(point_numeric_array):
    return '_'.join(map(str, point_numeric_array))


def point_neightbors(point, hash_black_points, color):
    global fig_for_out
    global array_original
    global clusters_hash
    array_original[map(int, point.split('_'))[0], map(int, point.split('_'))[1]] = color
    original_point = point
    ponto_marcar = map(int, point.split('_'))
    point = np.array(map(int, point.split('_')))
    # print original_point
    hash_black_points[original_point] = 'selected'
    cluster_name = 'cluster_'+ str(color)
    if cluster_name in clusters_hash:
        clusters_hash[cluster_name].append(ponto_marcar)
    else:
        clusters_hash[cluster_name] = []
        clusters_hash[cluster_name].append(ponto_marcar)


    kernel = [[-1,-1],[-1,0], [-1,1],
              [0,-1], [0, 0], [0,1],
              [1,-1], [1,0],  [1,1]]
    for k in kernel:


        if point_to_key(point + k) in hash_black_points:

            candidato = point_to_key(point + k)
            if hash_black_points[candidato] == 'unselected':
                # print point_to_key(point + k)
                #hash_black_points[candidato] = 'selected'
                # print point, candidato, map(int, candidato.split('_'))
                # io.imshow(array_original)
                #
                # io.show()
                array_original[map(int, candidato.split('_'))[0], map(int, candidato.split('_'))[1]] = color #modificando valor
                # print hash_black_points
                # print array_original

                point_neightbors(candidato, hash_black_points, color)


def get_hash_to_search(array_np):
    x_ax, y_ax = array_np.shape
    get_black_points = { str(x)+"_"+str(y): 'unselected' for x in range(x_ax) for y in range(y_ax) if array_np[x, y] == 0}
    #print get_black_points
    return get_black_points


def main():
    global fig_for_out
    imagem = '/home/lucas/PycharmProjects/competition/17218396409_591a8376c5_o.jpg'
    fig_for_out = resize(imread(imagem), (300,400))
    io.imshow(fig_for_out)
    io.show()
    fig = imread(imagem, as_grey=True)
    global array_original
    global clusters_hash
    clusters_hash = {}
    array_original = fig
    #fig = np.zeros((2,2), dtype=np.uint8)
    #print fig
    #fig[range(0,200),range(0,200)] = 1 #Desenhando linha na diagonal
    array_original = resize(array_original, (300, 400))
    #hist(pd.DataFrame({'x':[y for x in array_original for y in x]}))
    mask_white = array_original >= 0.4
    mask_black = array_original < 0.4
    array_original[mask_white] = 1
    array_original[mask_black] = 0
    x_len, y_len = fig.shape
    pontos = [(x, y) for x in range(x_len) for y in range(y_len) if fig[x, y] == 0]

    hash_pontos_pretos = get_hash_to_search(array_original)
    # print  array_original
    print 'iterating...'
    [ point_neightbors(clusters, hash_pontos_pretos, color+3) for color, clusters in enumerate(hash_pontos_pretos.keys()) if hash_pontos_pretos[clusters] == 'unselected']
    io.imshow(array_original)

    io.show()



    #hist_distance = [func_vectorized(pointos, point) for point in pontos]

 #   hist_distance = [distance.euclidean(loop_x, loop_y) for loop_x in sample(pontos, 2) for loop_y in pontos  if loop_x != loop_y]
    #print hist_distance[0]
    print 'calculo terminado...'
    #hist(pd.DataFrame({'x': hist_distance}))
    # train = [[x] for x in hist_distance]
    # g = mixture.GMM(n_components=20)
    # g.fit(train)
    # print g.means_
    # print 'bic: ',  g.bic(np.array(train))
    # # for x in np.nditer(fig):
    # #     print x

    print 'total contada no manualmente: 156'
    print 'MArcando grafico:'
    for cluster_marcar in clusters_hash.values():
        for marcar_x in cluster_marcar:
            if len(cluster_marcar) > 4:
                rr, cc = circle(marcar_x[0], marcar_x[1], 1 )
                # teria que contar apenas aqueles distantes da borda para contar
                fig_for_out[rr, cc] = [0,255,0]
                break

    print 'Quantidade de clusters:',  len([cls for cls in clusters_hash.values() if len(cls)>4])

    hist(pd.DataFrame({'x': [len(c_n) for c_n in clusters_hash.values()]}))
    io.imshow(fig_for_out)

    io.show()

if __name__ == '__main__':
    sys.exit(main())
