import numpy as np

def isPoiWithinBox(poi, sbox, toler=0.0001):
    """
    判断坐标是否落在矩形区域内
    sbox=[[x1,y1],[x2,y2]]
    不考虑在边界上，需要考虑就加等号
    """
    if poi[0] > sbox[0][0] and poi[0] < sbox[1][0] and poi[1] > sbox[0][1] and poi[1] < sbox[1][1]:
        return True
    if toler > 0:
        pass

    return False

# 定义一个计算距离的函数
def calculate_distance (base_coordinates ,box_find):
    """
    center_Position = [0, 3]
    box_find = [4,0]
    """
    base_coordinates= np.array(base_coordinates)
    box_find = np.array(box_find)
    dist = np.sqrt(np.sum(np.square(base_coordinates - box_find)))
    return dist