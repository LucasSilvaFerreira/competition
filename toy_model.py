__author__ = 'lucas'

import sys
from skimage import io
import numpy as np


def point_to_key(point_numeric_array):
    return '_'.join(map(str, point_numeric_array))


def point_neightbors(point, hash_black_points, color):
    global array_original
    array_original[map(int, point.split('_'))[0], map(int, point.split('_'))[1]] = color
    original_point = point
    point = np.array(map(int, point.split('_')))
    print original_point
    hash_black_points[original_point] = 'selected'
    kernel = [[-1,-1],[-1,0], [-1,1],
              [0,-1], [0, 0], [0,1],
              [1,-1], [1,0],  [1,1]]
    for k in kernel:


        if point_to_key(point + k) in hash_black_points:

            candidato = point_to_key(point + k)
            if hash_black_points[candidato] == 'unselected':
                print point_to_key(point + k)
                hash_black_points[candidato] = 'selected'
                print point, candidato, map(int, candidato.split('_'))
                # io.imshow(array_original)
                #
                # io.show()
                array_original[map(int, candidato.split('_'))[0], map(int, candidato.split('_'))[1]] = color #modificando valor
                print hash_black_points
                print array_original

                point_neightbors(candidato, hash_black_points, color)


def get_hash_to_search(array_np):
    x_ax, y_ax = array_np.shape
    get_black_points = { str(x)+"_"+str(y): 'unselected' for x in range(x_ax) for y in range(y_ax) if array_np[x, y] == 0}
    #print get_black_points
    return get_black_points



def main():

    global  array_original
    array_original = np.array(
        [[2,2,0,2,0,0],
         [2,0,0,2,2,2],
         [0,2,2,2,2,0],
         [2,2,2,0,2,0]])
    io.imshow(array_original)

    io.show()
    hash_pontos_pretos = get_hash_to_search(array_original)
    print  array_original
    [ point_neightbors(clusters, hash_pontos_pretos, color+3) for color, clusters in enumerate(hash_pontos_pretos.keys()) if hash_pontos_pretos[clusters] == 'unselected']


    print hash_pontos_pretos
    print '------------------------------final array-----------------'
    print array_original
    io.imshow(array_original)

    io.show()



if __name__ == '__main__':
    sys.exit(main())
