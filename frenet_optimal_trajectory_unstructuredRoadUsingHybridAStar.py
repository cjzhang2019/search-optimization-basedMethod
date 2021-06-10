# coding=utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import math
import cubic_spline
import vehicle
import collision_check
import scipy.spatial
import time
#from a_star_optimised import a_star
#from hybrid_a_star_optimised import hybrid_a_star
from frechet_distance import frenetDist
from scipy.stats import norm
import hybrid_A_star

# Parameter
MAX_CURVATURE = 1.0  # 最大曲率 [1/m]
TOP_ROAD_WIDTH = 5  # 最大道路宽度 [m]
BOTTOM_ROAD_WIDTH = -5  # 最大道路宽度 [m]
D_ROAD_W = 1.0  # 道路宽度采样间隔 [m]
frenetPathPointStep = 0.5
Pi = 3.1415
font1 = {'family' : 'Times New Roman','weight' : 'normal', 'size'   : 20}
font2 = {'family' : 'Times New Roman','weight' : 'normal', 'size'   : 30}
scriptSize = 12
#matplotlib.rcParams['text.usetex'] = True



# 损失函数权重

KD = 1.0
KC = 0.0
KO = 1.0
KK = 0.0

curvatureOfHybridA, curvatureOfExpectedPath, curvatureOfOptimalPath = [], [], []
do1, do2, do3 = [], [], []
totalS1,totalS2,totalS3 = [0], [0], [0]



class quintic_polynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        # 这里输入为初始的x,v,a以及目标的x,v,a和时间t1-t0,Takahashi的文章——Local path planning and motion control for AGV in positioning中已经证明，任何Jerk最优化问题中的解都可以使用一个5次多项式来表示
        # 计算五次多项式系数
        
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[T ** 3, T ** 4, T ** 5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T ** 2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]
        #这里输出为5次多项式的6个系数
    #以下通过这个5次函数反推轨迹点各信息
    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        dxt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return dxt

    def calc_second_derivative(self, t):
        ddxt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return ddxt

    def calc_third_derivative(self, t):
        dddxt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return dddxt



class Frenet_path:
    def __init__(self):
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []

        self.cd = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
        
class SdPoint:
    def __init__(self):
        self.s = 0.0
        self.d = 0.0
        self.d_d = 0.0
        self.d_dd = 0.0

class Node:
    def __init__(self):
        self.preSIndex = 0
        self.preDIndex = 0
        self.sdPoint = SdPoint()
        self.cost = 0.0
        self.fp = Frenet_path()

def samplePathWayPoints(curSpeed, curD, curDD,curDDD, s0, csp):
    points = []
    demandLength = 0
    sectionNumber = 0
    if(curSpeed < 2):
        demandLength = 12
    elif (curSpeed < 6):
        demandLength = 3* curSpeed + 6
    elif (curSpeed < 10):
        demandLength = 4 * curSpeed
    else:
        demandLength = 40
    totalLength = min(csp.s[-1] - s0, demandLength)
    accumulatedS = s0
    prevS = accumulatedS
    levelDistance = []
    if (totalLength < 6):
        levelDistance.append(totalLength)
        sectionNumber = sectionNumber + 1
    elif (totalLength < 13):
        sectionNumber = int(totalLength / 6)
        for i in range(sectionNumber):
            levelDistance.append(6)
        if(totalLength - sectionNumber * 6 > 3):
            levelDistance.append(totalLength - sectionNumber * 6)
            sectionNumber = sectionNumber + 1
    elif (totalLength < 25):
        sectionNumber = int(totalLength / 8)
        for i in range(sectionNumber):
            levelDistance.append(8)
        if(totalLength - sectionNumber * 8 > 4):
            levelDistance.append(totalLength - sectionNumber * 8)
            sectionNumber = sectionNumber + 1
    else:
        sectionNumber = int(totalLength / 15)
        for i in range(sectionNumber):
            levelDistance.append(15)
        if(totalLength - sectionNumber * 15 > 7.5):
            levelDistance.append(totalLength - sectionNumber * 15)
            sectionNumber = sectionNumber + 1
    
    zeroPoint = []
    initPoint = SdPoint()
    initPoint.d = curD
    initPoint.d_d = curDD
    initPoint.d_dd = curDDD
    initPoint.s = s0
    zeroPoint.append(initPoint)
    points.append(zeroPoint)
    
    for i in range(sectionNumber):
        accumulatedS = accumulatedS + levelDistance[i]
        s = accumulatedS
        kMinAllowedSampleStep = 1.0
        if(abs(s - prevS)<kMinAllowedSampleStep):
            continue
        prevS = s
        sampleD = []
        di = BOTTOM_ROAD_WIDTH
        while(di <= TOP_ROAD_WIDTH):
            sampleD.append(di)
            di = di + 1
        levelPoints = []
        for j in range(len(sampleD)):
            d = sampleD[j]
            sd = SdPoint()
            sd.s = s
            sd.d = d
            sd.d_d = 0.0
            sd.d_dd = 0.0
            levelPoints.append(sd)
        if(len(levelPoints) != 0):
            points.append(levelPoints)
    return points

def costTableInit(points):
    costTable = []
    for i in range(len(points)):
        ndLevel = []
        for j in range(len(points[i])):
            nd = Node()
            nd.sdPoint = points[i][j]
            ndLevel.append(nd)
        costTable.append(ndLevel)
    return costTable

def calculateCostTable(costTable, csp, ox, oy, kdtree):
    costTable[0][0].cost = 0
    costTable[0][0].preSIndex = -1
    costTable[0][0].preDIndex = -1
    for sIndex in range(1, len(costTable), 1):
        for dIndex in range(0, len(costTable[sIndex]),1):
            calculateCostAt(costTable, sIndex, dIndex, csp, ox, oy, kdtree)
    return costTable
            
def calculateCostAt(costTable, sIndex, dIndex, csp, ox, oy, kdtree):
    preLevel = costTable[sIndex - 1]
    minCost = 100000000
    for i in range(len(preLevel)):
        tempFp = Frenet_path()
        tempFp = calculateLineCost(preLevel[i].sdPoint, costTable[sIndex][dIndex].sdPoint, csp, ox, oy, kdtree)
        cost = tempFp.cd + preLevel[i].cost
        if cost < minCost:
            minCost = cost
            costTable[sIndex][dIndex].preSIndex = sIndex - 1
            costTable[sIndex][dIndex].preDIndex = i
            costTable[sIndex][dIndex].cost = minCost
            costTable[sIndex][dIndex].fp = tempFp
    return costTable
    
def calculateLineCost(prePoint, curPoint, csp, ox, oy, kdtree):
    curveLength = curPoint.s - prePoint.s
    fp = Frenet_path()
    lat_qp = quintic_polynomial(prePoint.d, prePoint.d_d, prePoint.d_dd, curPoint.d, curPoint.d_d, curPoint.d_dd, curveLength)
    if curveLength > frenetPathPointStep:
        si = 0
        while(si < curveLength - frenetPathPointStep):
            fp.s.append(prePoint.s + si)
            fp.d.append(lat_qp.calc_point(si))
            fp.d_d.append(lat_qp.calc_first_derivative(si))
            fp.d_dd.append(lat_qp.calc_second_derivative(si))
            fp.d_ddd.append(lat_qp.calc_third_derivative(si))
            si = si + frenetPathPointStep

    else:
        fp.s.append(prePoint.s)
        fp.d.append(lat_qp.calc_point(0))
        fp.d_d.append(lat_qp.calc_first_derivative(0))
        fp.d_dd.append(lat_qp.calc_second_derivative(0))
        fp.d_ddd.append(lat_qp.calc_third_derivative(0))
        fp.s.append(prePoint.s + curveLength)
        fp.d.append(lat_qp.calc_point(curveLength))
        fp.d_d.append(lat_qp.calc_first_derivative(curveLength))
        fp.d_dd.append(lat_qp.calc_second_derivative(curveLength))
        fp.d_ddd.append(lat_qp.calc_third_derivative(curveLength))
    if(abs(fp.d[-1]) < 5):
        fp.cd = KD * abs(fp.d[-1]/5)
    else:
        fp.cd = KD
    fp = calculateGlobalPath(fp, csp, ox, oy, kdtree)
    hardConstraintCost = calculateHardConstraintCost(fp, ox, oy, kdtree)
    fp.cd = fp.cd + hardConstraintCost
    return fp

def calculateGlobalPath(fp, csp, ox, oy, kdtree):
    for i in range(len(fp.s)):
        ix, iy = csp.calc_position(fp.s[i])
        iyaw = csp.calc_yaw(fp.s[i])
        di = fp.d[i]
        fx = ix + di * math.cos(iyaw + Pi / 2.0)
        fy = iy + di * math.sin(iyaw + Pi / 2.0)
        fp.x.append(fx)
        fp.y.append(fy)
    for i in range(len(fp.x) - 1):
        dx = fp.x[i+1] - fp.x[i]
        dy = fp.y[i+1] - fp.y[i]
        if (abs(dx) < 0.0001 and abs(dy) < 0.0001 and i == 0):
            vehicleHeading = 0.0
            fp.yaw.append(vehicleHeading)
        elif (abs(dx) < 0.0001 and abs(dy) < 0.0001 and i != 0):
            fp.yaw.append(fp.yaw[i-1])
        else:
            fp.yaw.append(math.atan2(dy,dx))
        fp.ds.append(math.sqrt(dx * dx + dy * dy))
    fp.yaw.append(fp.yaw[-1])
    fp.ds.append(fp.ds[-1])
    
    fp.c.append(0)
    for i in range(len(fp.yaw) -1 ):
        if fp.ds[i] < 0.001:
            fp.c.append(0)
        else:
            fp.c.append((fp.yaw[i+1] - fp.yaw[i]) / fp.ds[i])
    
    maxCurvature = 0
    for i in range(len(fp.c)):
        if fp.c[i] > maxCurvature:
            maxCurvature = fp.c[i]
    
    if maxCurvature < 0.2:
        fp.cd = KC * 5 * maxCurvature + fp.cd
    else:
        fp.cd = KC + fp.cd
    
    minDisSquare = 25
    for i in range(len(fp.x)):
        for j in range(len(ox)):
            dis2ObstacleSquare = (fp.x[i] - ox[j]) ** 2 + (fp.y[i] - oy[j]) ** 2
            if dis2ObstacleSquare < minDisSquare:
                minDisSquare = dis2ObstacleSquare
    
    if (minDisSquare < 4):
        fp.cd = fp.cd + KO
    elif (minDisSquare < 36):
        fp.cd = fp.cd + 9 * KO * (1 / math.sqrt(minDisSquare) - 1 / 6) ** 2
    else:
        fp.cd = fp.cd
    
    return fp

def calculateHardConstraintCost(fp, ox, oy, kdtree):
    hardConstraintCost = 0.0
    for i in range(len(fp.c)):
        if(fp.c[i] > MAX_CURVATURE):
            hardConstraintCost = 100000
            
    for i in range(len(fp.s)):
        if(fp.d[i] > 5):
            hardConstraintCost = 100000
    
    if collision_check.check_collision(ox, oy, fp.x, fp.y, fp.yaw, kdtree) == False:
        hardConstraintCost = 100000
        
    return hardConstraintCost

def calculateSimilarity(lastPath, currentNode, costTable):
    currentPath = []
    while (currentNode.preSIndex != -1):
        currentPath.append(currentNode)
        currentNode = costTable[currentNode.preSIndex][currentNode.preDIndex]
    
    currentFp = Frenet_path()
    if(len(currentPath)):
        currentPath.reverse()
        currentFp.d.append(currentPath[0].fp.d[0])
        currentFp.d_d.append(currentPath[0].fp.d_d[0])
        currentFp.d_dd.append(currentPath[0].fp.d_dd[0])
        currentFp.d_ddd.append(currentPath[0].fp.d_ddd[0])
        currentFp.s.append(currentPath[0].fp.s[0])
        currentFp.x.append(currentPath[0].fp.x[0])
        currentFp.y.append(currentPath[0].fp.y[0])
        currentFp.yaw.append(currentPath[0].fp.yaw[0])
        currentFp.ds.append(currentPath[0].fp.ds[0])
        currentFp.c.append(currentPath[0].fp.c[0])
        for i in range(0, len(currentPath), 1):
            for j in range(1, len(currentPath[i].fp.s), 1):
                if(currentPath[i].fp.s[j] - currentPath[i].fp.s[j-1] > 0.1):
                    currentFp.d.append(currentPath[i].fp.d[j])
                    currentFp.d_d.append(currentPath[i].fp.d_d[j])
                    currentFp.d_dd.append(currentPath[i].fp.d_dd[j])
                    currentFp.d_ddd.append(currentPath[i].fp.d_ddd[j])
                    currentFp.s.append(currentPath[i].fp.s[j])
                    currentFp.x.append(currentPath[i].fp.x[j])
                    currentFp.y.append(currentPath[i].fp.y[j])
                    currentFp.yaw.append(currentPath[i].fp.yaw[j])
                    currentFp.ds.append(currentPath[i].fp.ds[j])
                    currentFp.c.append(currentPath[i].fp.c[j])
                    currentFp.cd = currentFp.cd + currentPath[i].fp.cd
    P,Q = [], []
    if(len(lastPath.s)):
        for i in range(0, len(lastPath.s)):
            prePoint = []
            prePoint.append(lastPath.x[i])
            prePoint.append(lastPath.y[i])
            P.append(prePoint)
        for i in range(0, len(currentFp.s)):
            curPoint = []
            curPoint.append(currentFp.x[i])
            curPoint.append(currentFp.y[i])
            Q.append(curPoint)
        if(len(P) > len(Q)):
            while(len(P) != len(Q)):
                P.pop()
        if(len(P) < len(Q)):
            while(len(P) != len(Q)):
                Q.pop()
        
        dis = frenetDist(P,Q)
        d_min = 100000
        d_max = 0
        for i in range(len(P)):
            for j in range(len(Q)):
                tempDistance = math.sqrt((P[i][0] - Q[j][0])**2 + (P[i][1] - Q[j][1])**2)
                if(tempDistance > d_max):
                    d_max = tempDistance
                if(tempDistance < d_min):
                    d_min = tempDistance
        similarity = 1 - 2 * (1 - norm.cdf(3 * (dis - d_min)/d_max))
            
    else:
        similarity = 0
    return similarity
        
    

def frenet_optimal_planning(curSpeed, csp, s0, c_d, c_d_d, c_d_dd, ox, oy, lastPath):
    global tox,toy
    ox, oy = ox[:], oy[:]
    tox, toy = ox[:], oy[:]
    kdtree = KDTree(np.vstack((tox, toy)).T)
    samplePoints = samplePathWayPoints(curSpeed, c_d, c_d_d, c_d_dd, s0, csp)
    costTable = costTableInit(samplePoints)
    costTable = calculateCostTable(costTable, csp, ox, oy, kdtree)
    minCost = 100000
    minCostLastNode = costTable[0][0]
    minCostLastNode.cost = 100000
    for level in range(len(costTable)-1, 0, -1):
        for item in range(0, len(costTable[level]), 1):
            currentNode = costTable[level][item]
            if (currentNode.cost > 100000):
                continue
            else:
               # difToLastPathCost = calculateSimilarity(lastPath, currentNode, costTable)
                difToLastPathCost = 0
                totalCost = currentNode.cost / level + KK * difToLastPathCost
                if (totalCost < minCost):
                    minCost = totalCost
                    minCostLastNode = currentNode
        if (minCostLastNode.cost < 100000):
            break
        
    minCostPath = []
    while (minCostLastNode.preSIndex != -1):
        minCostPath.append(minCostLastNode)
        minCostLastNode = costTable[minCostLastNode.preSIndex][minCostLastNode.preDIndex]
    
    finalFp = Frenet_path()
    if(len(minCostPath)):
        minCostPath.reverse()
        finalFp.d.append(minCostPath[0].fp.d[0])
        finalFp.d_d.append(minCostPath[0].fp.d_d[0])
        finalFp.d_dd.append(minCostPath[0].fp.d_dd[0])
        finalFp.d_ddd.append(minCostPath[0].fp.d_ddd[0])
        finalFp.s.append(minCostPath[0].fp.s[0])
        finalFp.x.append(minCostPath[0].fp.x[0])
        finalFp.y.append(minCostPath[0].fp.y[0])
        finalFp.yaw.append(minCostPath[0].fp.yaw[0])
        finalFp.ds.append(minCostPath[0].fp.ds[0])
        finalFp.c.append(minCostPath[0].fp.c[0])
        for i in range(0, len(minCostPath), 1):
            for j in range(1, len(minCostPath[i].fp.s), 1):
                if(minCostPath[i].fp.s[j] - minCostPath[i].fp.s[j-1] > 0.1):
                    finalFp.d.append(minCostPath[i].fp.d[j])
                    finalFp.d_d.append(minCostPath[i].fp.d_d[j])
                    finalFp.d_dd.append(minCostPath[i].fp.d_dd[j])
                    finalFp.d_ddd.append(minCostPath[i].fp.d_ddd[j])
                    finalFp.s.append(minCostPath[i].fp.s[j])
                    finalFp.x.append(minCostPath[i].fp.x[j])
                    finalFp.y.append(minCostPath[i].fp.y[j])
                    finalFp.yaw.append(minCostPath[i].fp.yaw[j])
                    finalFp.ds.append(minCostPath[i].fp.ds[j])
                    finalFp.c.append(minCostPath[i].fp.c[j])
                    finalFp.cd = finalFp.cd + minCostPath[i].fp.cd
    return finalFp


def generate_target_course(x, y):
    csp = cubic_spline.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp

def re_generate_target_course(lastPathX, lastPathY, lastPathYaw):
    tempWx, tempWy = [], []
    for i in range(100):
        tempWx.append(lastPathX + i * math.cos(lastPathYaw))
        tempWy.append(lastPathY + i * math.sin(lastPathYaw))
    tx, ty, tyaw, tk, csp = generate_target_course(tempWx, tempWy)
    return tx, ty, tyaw, tk, csp

class KDTree:
    """
    Nearest neighbor search class with KDTree
    Dimension is two
    """

    def __init__(self, data):
        # store kd-tree
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, inp, k=1):
        """
        k=1 means to query the nearest neighbours and return squeezed result
        inp: input data
        """

        if len(inp.shape) >= 2:  # multi input 
            index = []
            dist = []
            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)

            return index, dist
        else:
            dist, index = self.tree.query(inp, k=k)
            return index, dist

    def search_in_distance(self, inp, r):
        """
        find points within a distance r
        """
        index = self.tree.query_ball_point(inp, r)
        return index


def main():
    # 路线
    wx = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    wy = [0.0, 4.0, 5.0, 6.0, 10.0, 10.0]
    sDecision = 2

    # 障碍物列表
    ox = []
    oy = []
    obstacleInFrenet = []
    boundryX = []
    boundryY = []
    
    #case1
    for i in range (6):
        ox.append(10)
        oy.append(i-1)
    for i in range (6):
        ox.append(13)
        oy.append(i-1)
    for i in range (2):
        ox.append(11+i)
        oy.append(4)
    for i in range (2):
        ox.append(11+i)
        oy.append(-1)

    for i in range (6):
        ox.append(25)
        oy.append(5+i)
    for i in range (6):
        ox.append(28)
        oy.append(5+i)
    for i in range (2):
        ox.append(26+i)
        oy.append(5)
    for i in range (2):
        ox.append(26+i)
        oy.append(10)

    for i in range (6):
        ox.append(40)
        oy.append(6+i)
    for i in range (6):
        ox.append(43)
        oy.append(6+i)
    for i in range (2):
        ox.append(41+i)
        oy.append(6)
    for i in range (2):
        ox.append(41+i)
        oy.append(11)

    txo, tyo, tyawo, tko, cspo = generate_target_course(wx, wy)
    #original global path
    
    for i in range(len(txo)):
        boundryX.append(txo[i])
        boundryY.append(tyo[i] + 6)
        ox.append(txo[i])
        oy.append(tyo[i] + 6)
    for i in range(len(txo)):
        boundryX.append(txo[i])
        boundryY.append(tyo[i] - 6)
        ox.append(txo[i])
        oy.append(tyo[i] - 6)
    #generate boundries
    
    for i in range(len(ox)):
        minDisSquare = 100000
        minIndex = 0
        for j in range (len(txo)):
            dis2csp = (txo[j] - ox[i]) ** 2 + (tyo[j] - oy[i]) ** 2
            if(dis2csp < minDisSquare):
                minDisSquare = dis2csp
                minIndex = j
        s = np.arange(0, cspo.s[-1], 0.1)
        obInFrenetS = s[minIndex]
        absObInFrenetD = math.sqrt(minDisSquare)
        if(((txo[minIndex] - txo[minIndex-1]) * (oy[i] - tyo[minIndex]) - (tyo[minIndex] - tyo[minIndex-1]) * (ox[i] - txo[minIndex])) < 0):
            ob = (obInFrenetS, -absObInFrenetD)
        else:
            ob = (obInFrenetS, absObInFrenetD)
        obstacleInFrenet.append(ob)
    obS = []
    obD = []
    for i in range(len(obstacleInFrenet)):
        obS.append(obstacleInFrenet[i][0])
        obD.append(obstacleInFrenet[i][1])
    #obstacles in Frenet Coordinates

    s0 = 0
    c_d = 0
    c_d_d = 0
    c_d_dd = 0
    curSpeed = 1 
    if (sDecision == 1):
        tx, ty, tyaw, tk, csp = txo, tyo, tyawo, tko, cspo
    
    if (sDecision == 2):
        Path1, Path = hybrid_A_star.calc_hybrid_astar_path(0 , 0 , np.deg2rad(0) , 50 ,0 , np.deg2rad(0),  obS , obD , 1 , 1)
        rs, rd = [], []
        rx, ry = [], []
        for i in range(len(Path.x)):
            rs.append(Path.x[i])
            rd.append(Path.y[i])
        rs.pop()
        rd.pop()

        for j in range(0, len(rs), int(len(rs) / 10)):
            ix, iy = cspo.calc_position(rs[j])
            iyaw = cspo.calc_yaw(rs[j])
            di = rd[j]
            fx = ix + di * math.cos(iyaw + Pi / 2.0)
            fy = iy + di * math.sin(iyaw + Pi / 2.0)
            rx.append(fx)
            ry.append(fy)
        
        plt.figure(1)

        plt.plot([0,50], [0,0], '--', markersize = 2, color = 'black')

        plt.plot(rs, rd, "-", color = 'magenta')
        plt.legend(['Reference lane','Dispersed expected path'], prop=font1, loc=2, bbox_to_anchor=(0.21,1.3),borderaxespad = 0.)
#        plt.plot([0, 50], [5,5], '-', markersize = 2, color = 'black')
#        plt.plot([0, 50], [-5,-5], '-', markersize = 2, color = 'black')
        plt.plot(obS, obD, 'o', markersize = 2, color = 'black')
        
        plt.axis("equal")
        plt.xlabel('s (m)',font1)
        plt.ylabel('l (m)',font1)
        plt.xlim(0, 50)
        plt.ylim(-10, 10)
        plt.tick_params(labelsize=scriptSize)
#        plt.savefig('improvedAStar.eps', format='eps', bbox_inches='tight')
        plt.show()
        
        plt.figure(2)
        tx, ty, tyaw, tk, csp = generate_target_course(rx, ry)
        plt.plot(txo, tyo, "--", color = 'black')
        plt.plot(tx, ty, color = 'red')
        plt.plot(boundryX,boundryY, "o", markersize = 2, color = 'black')
        plt.xlabel('x (m)',font1)
        plt.ylabel('y (m)',font1)
        plt.tick_params(labelsize=scriptSize)
        plt.legend(['Reference lane','Continuous expected path'], prop=font1, loc=2, bbox_to_anchor=(0.18,1.3),borderaxespad = 0.)
        plt.fill([10,13,13,10], [-1,-1,4,4], color = 'black')
        plt.fill([40,43,43,40], [10,10,5,5], color = 'black')
        plt.fill([25,28,28,25], [10,10,5,5], color = 'black')
        plt.axis("equal")
#        plt.savefig('curveFitter.eps', format='eps', bbox_inches='tight')
        plt.show()
    lastPath = Frenet_path()
    totalXList = []
    totalYList = []
    totalYawList = []
    for i in range(50000):
        
        start = time.time()
        path = frenet_optimal_planning(curSpeed, csp, s0, c_d, c_d_d, c_d_dd, ox, oy, lastPath)
        lastPath = path
        end = time.time()

        s0 = path.s[1]
        c_d = path.d[1]
#        print(c_d)
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 2.0:
            print("到达目标")
            break

        plt.cla()
        plt.plot(txo, tyo, "--", color = 'black')
        plt.plot(tx, ty, color = 'red')
        
        plt.plot(path.x[1:], path.y[1:], "-ob", markersize = 3)
        plt.plot(boundryX,boundryY, "o", markersize = 2, color = 'black')
        plt.legend(['Reference lane','Expected path', 'Optimal path'], prop=font1, loc=2, bbox_to_anchor=(0.44,1.43),borderaxespad = 0.)
        plt.fill([10,13,13,10], [-1,-1,4,4], color = 'black')
        plt.fill([40,43,43,40], [10,10,5,5], color = 'black')
        plt.fill([25,28,28,25], [10,10,5,5], color = 'black')
        plt.axis("equal")
        
        plt.xlim(0, 50)
        plt.ylim(-7, 18)
        plt.tick_params(labelsize=scriptSize)
        totalXList.append(path.x[0])
        totalYList.append(path.y[0])
        totalYawList.append(path.yaw[0])
        vehicle.plot_trailer(path.x[0], path.y[0], path.yaw[0], 0)
        plt.xlabel('x (m)',font1)
        plt.ylabel('y (m)',font1)
        
        
#        plt.savefig('result(3)%d.eps'%(i), format='eps', bbox_inches='tight')
        plt.grid(False)
        plt.pause(0.001)

#    最后一针break的会有问题
    plt.show()
    
#与单纯使用混合A*算法比较
    for i in range(len(txo)):
        ox.append(txo[i])
        oy.append(tyo[i] + 6)
    for i in range(len(txo)):
        ox.append(txo[i])
        oy.append(tyo[i] - 6)
    ox.append(0)
    oy.append(-5)
    ox.append(0)
    oy.append(15)
    ox.append(51)
    oy.append(-5)
    ox.append(51)
    oy.append(15)

    Path1, Path = hybrid_A_star.calc_hybrid_astar_path(0 , 0 , np.deg2rad(30) , 45 ,10 , np.deg2rad(-10),  ox , oy , 1 , 1)
    Path.x.pop()
    Path.y.pop()
    
    hybridYaw = []
    DS1 = []
    for i in range(len(Path.x) - 1):
        dx = Path.x[i+1] - Path.x[i]
        dy = Path.y[i+1] - Path.y[i]
        if (abs(dx) < 0.0001 and abs(dy) < 0.0001 and i == 0):
            vehicleHeading = 0.0
            hybridYaw.append(vehicleHeading)
        elif (abs(dx) < 0.0001 and abs(dy) < 0.0001 and i != 0):
            hybridYaw.append(hybridYaw[i-1])
        else:
            hybridYaw.append(abs(math.atan2(dy,dx)))
        DS1.append(math.sqrt(dx * dx + dy * dy))
        totalS1.append(math.sqrt(dx * dx + dy * dy) + totalS1[i])
        if(totalS1[-1] > 50):
            break
    hybridYaw.append(hybridYaw[-1])
    
    curvatureOfHybridA.append(0)
    for i in range(len(hybridYaw) -1):
        if DS1[i] < 0.001:
            curvatureOfHybridA.append(0)
        else:
            curvatureOfHybridA.append(abs((hybridYaw[i+1] - hybridYaw[i]) / DS1[i]))

    expectedYaw = []
    DS2 = []
    for i in range(len(tx) - 1):
        dx = tx[i+1] - tx[i]
        dy = ty[i+1] - ty[i]
        if (abs(dx) < 0.0001 and abs(dy) < 0.0001 and i == 0):
            vehicleHeading = 0.0
            expectedYaw.append(vehicleHeading)
        elif (abs(dx) < 0.0001 and abs(dy) < 0.0001 and i != 0):
            expectedYaw.append(expectedYaw[i-1])
        else:
            expectedYaw.append(abs(math.atan2(dy,dx)))
        DS2.append(math.sqrt(dx * dx + dy * dy))
        totalS2.append(math.sqrt(dx * dx + dy * dy) + totalS2[i])
        if(totalS2[-1] > 50):
            break
    expectedYaw.append(expectedYaw[-1])
    
    curvatureOfExpectedPath.append(0)
    for i in range(len(expectedYaw) -1):
        if DS2[i] < 0.001:
            curvatureOfExpectedPath.append(0)
        else:
            curvatureOfExpectedPath.append(abs((expectedYaw[i+1] - expectedYaw[i]) / DS2[i]))
    
    optimalYaw = []
    DS3 = []
    for i in range(len(totalXList) - 1):
        dx = totalXList[i+1] - totalXList[i]
        dy = totalYList[i+1] - totalYList[i]
        if (abs(dx) < 0.0001 and abs(dy) < 0.0001 and i == 0):
            vehicleHeading = 0.0
            optimalYaw.append(vehicleHeading)
        elif (abs(dx) < 0.0001 and abs(dy) < 0.0001 and i != 0):
            optimalYaw.append(optimalYaw[i-1])
        else:
            optimalYaw.append(abs(math.atan2(dy,dx)))
        DS3.append(math.sqrt(dx * dx + dy * dy))
        totalS3.append(math.sqrt(dx * dx + dy * dy) + totalS3[i])
        if(totalS3[-1] > 50):
            break
    optimalYaw.append(optimalYaw[-1])
    
    curvatureOfOptimalPath.append(0)
    for i in range(len(optimalYaw) -1):
        if DS3[i] < 0.001:
            curvatureOfOptimalPath.append(0)
        else:
            curvatureOfOptimalPath.append(abs((optimalYaw[i+1] - optimalYaw[i]) / (2 * DS3[i])))
    
    for i in range(len(totalS1)):
        minDis = 1000000
        for j in range(len(ox)):
            dis = math.sqrt((Path.x[i] - ox[j]) ** 2 + (Path.y[i] - oy[j]) ** 2)
            if(dis < minDis):
                minDis = dis
        do1.append(minDis)
    
    for i in range(len(totalS2)):
        minDis = 1000000
        for j in range(len(ox)):
            dis = math.sqrt((tx[i] - ox[j]) ** 2 + (ty[i] - oy[j]) ** 2)
            if(dis < minDis):
                minDis = dis
        do2.append(minDis)
    
    for i in range(len(totalS3)):
        minDis = 1000000
        for j in range(len(ox)):
            dis = math.sqrt((totalXList[i] - ox[j]) ** 2 + (totalYList[i] - oy[j]) ** 2)
            if(dis < minDis):
                minDis = dis
        do3.append(minDis)
        

    plt.figure(3)
    for i in range(len(totalXList)):
        vehicle.plot_trailer(totalXList[i], totalYList[i], totalYawList[i], 0)
    plt.plot(txo, tyo, "--", color = 'black')
    plt.plot(tx, ty, color = 'red')
    plt.plot(boundryX,boundryY, "o", markersize = 2, color = 'black')
    plt.fill([10,13,13,10], [-1,-1,4,4], color = 'black')
    plt.fill([40,43,43,40], [10,10,5,5], color = 'black')
    plt.fill([25,28,28,25], [10,10,5,5], color = 'black')
    plt.axis("equal")
    plt.xlim(0, 50)
    plt.ylim(-7, 18)
    plt.xlabel('x (m)',font1)
    plt.ylabel('y (m)',font1)
    plt.tick_params(labelsize=scriptSize)
#    plt.savefig('resultOfTheHybridPathPlanningAlgorithm.eps', format='eps', bbox_inches='tight')
    plt.grid(False)
    plt.show()
    
    plt.figure(4)
    plt.plot(txo, tyo, "--", color = 'black')
    plt.plot(Path.x, Path.y, "--", color = 'cyan')
    plt.plot(tx, ty, "-", color = 'red')
    plt.plot(totalXList, totalYList, "-o", color = 'blue', markersize = 3)
    plt.legend(['Reference lane','Result of hybrid A*','Expected path', 'Optimal path'], prop=font1, loc=2, bbox_to_anchor=(0.33,1.55),borderaxespad = 0.)
    plt.plot(boundryX,boundryY, "o", markersize = 2, color = 'black')
    plt.fill([10,13,13,10], [-1,-1,4,4], color = 'black')
    plt.fill([40,43,43,40], [10,10,5,5], color = 'black')
    plt.fill([25,28,28,25], [10,10,5,5], color = 'black')
    plt.axis("equal")
    plt.xlim(0, 50)
    plt.ylim(-7, 18)
    plt.tick_params(labelsize=scriptSize)
    plt.xlabel('x (m)',font1)
    plt.ylabel('y (m)',font1)
#    plt.savefig('comparisionWithHybridAStar.eps', format='eps', bbox_inches='tight')
    plt.grid(False)
    plt.show()
    
    plt.figure(5)
    plt.plot(totalS1, curvatureOfHybridA, "--", color = 'cyan')
    plt.plot(totalS2, curvatureOfExpectedPath, "-", color = 'red')
    plt.plot(totalS3, curvatureOfOptimalPath, "-o", color = 'blue',markersize = 3)
    plt.legend(['Hybrid A*','Expected','Optimal'], prop=font1, loc=2, bbox_to_anchor=(0.55,1.43),borderaxespad = 0.)
    plt.xlabel('s (m)',font1)
    plt.ylabel('Curvature (m${}^{-1}$)',font1)
    plt.xlim(0, 50)
    plt.tick_params(labelsize=scriptSize)
#    plt.savefig('Curvature.eps', format='eps', bbox_inches='tight')
    plt.grid(False)
    plt.show()
    
    plt.figure(6)
    plt.plot(totalS1, do1, "--", color = 'cyan')
    plt.plot(totalS2, do2, "-", color = 'red')
    plt.plot(totalS3, do3, "-o", color = 'blue', markersize = 3)
    plt.legend(['Hybrid A*','Expected','Optimal'], prop=font1, loc=2, bbox_to_anchor=(0.55,1.43),borderaxespad = 0.)
    plt.xlabel('s (m)',font1)
    plt.ylabel('Distance (m)',font1)
    plt.xlim(0, 50)
    plt.ylim(1, 6)
    plt.tick_params(labelsize=scriptSize)
#    plt.savefig('ClosestObstacleDistance.eps', format='eps', bbox_inches='tight')
    plt.grid(False)
    plt.show()
    
    curvatureOfHybridATotal = 0
    minCurvatureOfHybridA = 100
    maxCurvatureOfHybridA = 0
    for i in range(len(curvatureOfHybridA)):
        curvatureOfHybridATotal = curvatureOfHybridATotal + curvatureOfHybridA[i]
        if(curvatureOfHybridA[i] > maxCurvatureOfHybridA):
            maxCurvatureOfHybridA = curvatureOfHybridA[i]
        if(curvatureOfHybridA[i] < minCurvatureOfHybridA):
            minCurvatureOfHybridA = curvatureOfHybridA[i]
    averageCurvatureOfHybridA = curvatureOfHybridATotal / len(curvatureOfHybridA)
    print("averageCurvatureOfHybridA:",averageCurvatureOfHybridA,"minCurvatureOfHybridA:",minCurvatureOfHybridA,"maxCurvatureOfHybridA:",maxCurvatureOfHybridA)
    
    curvatureOfExpectedPathTotal = 0
    minCurvatureOfExpectedPath = 100
    maxCurvatureOfExpectedPath = 0
    for i in range(len(curvatureOfExpectedPath)):
        curvatureOfExpectedPathTotal = curvatureOfExpectedPathTotal + curvatureOfExpectedPath[i]
        if(curvatureOfExpectedPath[i] > maxCurvatureOfExpectedPath):
            maxCurvatureOfExpectedPath = curvatureOfExpectedPath[i]
        if(curvatureOfExpectedPath[i] < minCurvatureOfExpectedPath):
            minCurvatureOfExpectedPath = curvatureOfExpectedPath[i]
    averageCurvatureOfExpectedPath = curvatureOfExpectedPathTotal / len(curvatureOfExpectedPath)
    print("averageCurvatureOfExpectedPath:",averageCurvatureOfExpectedPath,"minCurvatureOfExpectedPath:",minCurvatureOfExpectedPath,"maxCurvatureOfExpectedPath:",maxCurvatureOfExpectedPath)
    
    curvatureOfOptimalPathTotal = 0
    minCurvatureOfOptimalPath = 100
    maxCurvatureOfOptimalPath = 0
    for i in range(len(curvatureOfOptimalPath)):
        curvatureOfOptimalPathTotal = curvatureOfOptimalPathTotal + curvatureOfOptimalPath[i]
        if(curvatureOfOptimalPath[i] > maxCurvatureOfOptimalPath):
            maxCurvatureOfOptimalPath = curvatureOfOptimalPath[i]
        if(curvatureOfOptimalPath[i] < minCurvatureOfOptimalPath):
            minCurvatureOfOptimalPath = curvatureOfOptimalPath[i]
    averageCurvatureOfOptimalPath = curvatureOfOptimalPathTotal / len(curvatureOfOptimalPath)
    print("averageCurvatureOfOptimalPath:",averageCurvatureOfOptimalPath,"minCurvatureOfOptimalPath:",minCurvatureOfOptimalPath,"maxCurvatureOfOptimalPath:",maxCurvatureOfOptimalPath)
    
    minDo1 = 100
    for i in range(len(do1)):
        if(do1[i] < minDo1):
            minDo1 = do1[i]
            
    minDo2 = 100
    for i in range(len(do2)):
        if(do2[i] < minDo2):
            minDo2 = do2[i]

    minDo3 = 100
    for i in range(len(do3)):
        if(do3[i] < minDo3):
            minDo3 = do3[i]
    
    print("do1:",minDo1,"do2:",minDo2,"do3:",minDo3)

if __name__ == '__main__':
    main()