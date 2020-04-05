# -*- coding: utf-8 -*-
# @Time : 2020/4/1 上午10:44
# @Author : LuoLu
# @FileName: intersections_curves.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com





# //PLEASE NOTE: THE IMAGE MUST BE THINNED FIRST!! Read above paragraph
# def getIntersections(img):
#     width = img.length
#     height = img[0].length
#     intersections = 0
#     for i in range(width - 2):
#         for i in range(height - 2):
#             if (img[i, j] > 50):
#                 # // initialize a queue for this connected component
#                 ArrayList < Integer[] > queue = new ArrayList <> ()
#                 queue.add({i, j})
#                 img[i, j] = 0
#                 while (!queue.isEmpty()):
#                     Integer[] c = queue.remove(0)
#                     int count = 0;
#                     for (int x = -1; x < 2; x++):
#                         for (int y = -1; y < 2; y++):
#                             if (img[c[0]+ x, c[1] + y] > 50):
#                                 queue.add({c[0] + x, c[1] + y})
#                                 img[c[0] + x, c[1] + y] = 0
#                                 # // count the number of pixels connected to this one
#                                 count += 1
#                     if (count > 1):
#                         # // if more than 1 pixel was connected, it's at an intersection
#                         intersections += 1
#     # // add 1 intersection for every connected component found
#     intersections += 1
#     # // correct for the original number of connected components in the image
#     # // minus 1; helps offset against tendency for false negatives
#     intersections -= original_number_of_connected_components - 1
#     return intersections / 2
#


    
    
    