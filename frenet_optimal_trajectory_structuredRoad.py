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
from hybrid_a_star_optimised import hybrid_a_star
from frechet_distance import frenetDist
from scipy.stats import norm

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
KC = 0.3
KO = 1.0
KK = 1.0

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
        fp.cd = fp.cd + 9 * KO * (1 / minDisSquare - 1 / 6) ** 2
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
                difToLastPathCost = calculateSimilarity(lastPath, currentNode, costTable)
#                difToLastPathCost = 0
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
    wy = [0, 0, 0, 0, 0, 0]
    sDecision = 1

    # 障碍物列表
    ox = []
    oy = []
    obstacleInFrenet = []
    boundryX = []
    boundryY = []
    
    #case1
    for i in range (3):
        ox.append(20)
        oy.append(1-i)
    for i in range (3):
        ox.append(25)
        oy.append(1-i)
    for i in range (4):
        ox.append(21+i)
        oy.append(1)
    for i in range (4):
        ox.append(21+i)
        oy.append(-1)

    txo, tyo, tyawo, tko, cspo = generate_target_course(wx, wy)
    #original global path
    
    for i in range(len(txo)):
        boundryX.append(txo[i])
        boundryY.append(tyo[i] + 6)
    for i in range(len(txo)):
        boundryX.append(txo[i])
        boundryY.append(tyo[i] - 6)
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
#        simple_a_star = a_star(-5, 55, -6, 6, obstacle = obstacleInFrenet)
        simple_a_star = hybrid_a_star(-5, 55, -6, 6, obstacle = obstacleInFrenet, vehicle_length = 2)
        candidatePoints = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
        for j in candidatePoints:
            path = simple_a_star.find_path((0, 0, 0), (50, j, 0))
            if len(path) != 0:
                break
        rs, rd = [], []
        rx, ry = [], []
        for node in path:
            rs.append(node[0])
            rd.append(node[1])
        rs.pop()
        rd.pop()
        rs.reverse()
        rd.reverse()
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
        plt.legend(['Reference lane','Dispersed expected path'], prop=font1, loc=2, bbox_to_anchor=(0.19,1.3),borderaxespad = 0.)
        plt.plot([0, 50], [5,5], '-', markersize = 2, color = 'black')
        plt.plot([0, 50], [-5,-5], '-', markersize = 2, color = 'black')
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
        plt.legend(['Reference lane','Continuous expected path'], prop=font1, loc=2, bbox_to_anchor=(0.16,1.3),borderaxespad = 0.)
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

#        if (path != None):
#            lastPathX = path.x[1]
#            lastPathY = path.y[1]
#            lastPathYaw = path.yaw[1]
#
##            print(lastPathX, lastPathY)
#        else:
#            tx, ty, tyaw, tk, csp = re_generate_target_course(lastPathX, lastPathY, lastPathYaw)
#            c_d = 0  # 当前的d方向位置 [m]
#            c_d_d = 0.0  # 当前横向速度 [m/s]
#            c_d_dd = 0.0  # 当前横向加速度 [m/s2]
#            s0 = 0.0  # 当前所在的位置
#            print(111)
#            path = frenet_optimal_planning(csp, s0, c_d, c_d_d, c_d_dd, ox, oy)
        s0 = path.s[1]
        c_d = path.d[1]
#        print(c_d)
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 2.0:
            print("到达目标")
            break

        plt.cla()
        plt.plot(tx, ty, "--", color = 'black')
        
        plt.plot(path.x[1:], path.y[1:], "-ob", markersize = 3)
        plt.plot(boundryX,boundryY, "o", markersize = 2, color = 'black')
        plt.legend(['Reference lane', 'Ours'], prop=font1, loc=2, bbox_to_anchor=(0.44,1.30),borderaxespad = 0.)
        plt.fill([20,25,25,20], [-1,-1,1,1], color = 'black')
        plt.axis("equal")
        
#        file = open('C:/Users/86159/Desktop/7.txt','a')
#        l=len(path.x)
#        for i in range(l-1):
#            file.write("%f"%path.x[i+1])
#            file.write(" ")
#            file.write("%f"%path.y[i+1])
#            file.write(" ")
#            file.write("%f\n"%path.s_d[i+1])
#        file.write("\n")
        
#        file.close
        plt.xlim(0, 50)
        plt.ylim(-4, 7)
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
    plt.plot(tx, ty, "--", color = 'black')
    plt.plot(boundryX,boundryY, "o", markersize = 2, color = 'black')
    plt.fill([20,25,25,20], [-1,-1,1,1], color = 'black')
    plt.axis("equal")
    plt.xlim(0, 50)
    plt.ylim(-6, 6)
    plt.xlabel('x (m)',font1)
    plt.ylabel('y (m)',font1)
    plt.tick_params(labelsize=scriptSize)
#    plt.savefig('resultOfTheHybridPathPlanningAlgorithm.eps', format='eps', bbox_inches='tight')
    plt.grid(False)
    plt.show()
    
    plt.figure(4)
    plt.plot(totalS3, curvatureOfOptimalPath, "-o", color = 'blue',markersize = 3)
    plt.legend(['Ours'], prop=font1, loc=2, bbox_to_anchor=(0.52,1.43),borderaxespad = 0.)
    plt.xlabel('s (m)',font1)
    plt.ylabel('Curvature (m${}^{-1}$)',font1)
    plt.xlim(0, 50)
    plt.tick_params(labelsize=scriptSize)
#    plt.savefig('Curvature.eps', format='eps', bbox_inches='tight')
    plt.grid(False)
    plt.show()
    plt.figure(6)
    plt.plot(totalS3, do3, "-o", color = 'blue', markersize = 3)
    plt.legend(['Ours'], prop=font1, loc=2, bbox_to_anchor=(0.52,1.43),borderaxespad = 0.)
    plt.xlabel('s (m)',font1)
    plt.ylabel('Distance (m)',font1)
    plt.xlim(0, 50)
    plt.ylim(1, 6)
    plt.tick_params(labelsize=scriptSize)
#    plt.savefig('ClosestObstacleDistance.eps', format='eps', bbox_inches='tight')
    plt.grid(False)
    plt.show()
    
    
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
    

    minDo3 = 100
    for i in range(len(do3)):
        if(do3[i] < minDo3):
            minDo3 = do3[i]
    
    print("do3:",minDo3)
#averageCurvatureOfOptimalPath: 0.025280761827384052 minCurvatureOfOptimalPath: 0 maxCurvatureOfOptimalPath: 0.1544894629870346
#do3: 2.040354528927823
    
#averageCurvatureOfOptimalPath: 0.02594288054284643 minCurvatureOfOptimalPath: 0 maxCurvatureOfOptimalPath: 0.2279601544036403
#do3: 1.077229089955242

#s1 = [0,
# 0.5,
# 1.0,
# 1.5,
# 2.0,
# 2.5,
# 3.0,
# 3.5,
# 4.0,
# 4.5,
# 5.0,
# 5.5,
# 6.0,
# 6.5,
# 7.0,
# 7.5,
# 8.0,
# 8.5,
# 9.0,
# 9.5,
# 10.000104002200507,
# 10.503534238732186,
# 11.019557500795525,
# 11.561227242772901,
# 12.133719242053447,
# 12.730029006837064,
# 13.336479491604601,
# 13.939081108764245,
# 14.527480463452752,
# 15.0963896897206,
# 15.64520918742436,
# 16.176624970670986,
# 16.694893703182693,
# 17.204384573477178,
# 17.708689277915674,
# 18.210313626309468,
# 18.71076752197452,
# 19.210830361261255,
# 19.710830953270477,
# 20.210831457329093,
# 20.711090997805922,
# 21.211634911779836,
# 21.71204926449356,
# 22.21221066906304,
# 22.712225841171726,
# 23.21224201031082,
# 23.71234298716957,
# 24.212536153574405,
# 24.71278303620734,
# 25.21303457289414,
# 25.713429760447898,
# 26.216174226302154,
# 26.732957215406408,
# 27.279390333005438,
# 27.859455218419725,
# 28.463917447606736,
# 29.077380705073274,
# 29.68502744690558,
# 30.276414155635305,
# 30.846620536748127,
# 31.395607073653476,
# 31.92660753510572,
# 32.444290312089684,
# 32.95326199630664,
# 33.45720428518911,
# 33.9586211723771,
# 34.4589812651408,
# 34.959017421911895,
# 35.45902250247512,
# 35.95906986355441,
# 36.45914837811794,
# 36.9592302679093,
# 37.459296602590165,
# 37.95934121192291,
# 38.45936653552847,
# 38.95937846182094,
# 39.459382817607285,
# 39.95938381544906,
# 40.45938385517733,
# 40.95938396488433,
# 41.45938437577712,
# 41.959384979511555,
# 42.459385601430085,
# 42.95938611921518,
# 43.45938648602886,
# 43.95938671071863,
# 44.45938682880211,
# 44.95938688003992,
# 45.45938689647018,
# 45.95938689892117,
# 46.45938689838465,
# 46.95938689923186,
# 47.459386902241754,
# 47.959386906631174,
# 48.459386911258015]
#
#curvature1 = [0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.020348252165677386,
# 0.09638845945447874,
# 0.13214420717437916,
# 0.1404499487100558,
# 0.10514058999344031,
# 0.05902415519728619,
# 0.02116566056383282,
# 0.0077248169879010386,
# 0.030618720269437863,
# 0.04925473793078294,
# 0.06355700568502258,
# 0.07234846497985999,
# 0.07458022552851254,
# 0.07041040674464291,
# 0.06140189811495891,
# 0.04982586398654699,
# 0.037781230180426714,
# 0.02671583530600412,
# 0.014219504695110916,
# 0.00011884234303277196,
# 0.03070020565393922,
# 0.014401760083664797,
# 0.00591909600015389,
# 0.015278980913781347,
# 0.017609642923276977,
# 0.0003445243458950362,
# 0.012053222472240634,
# 0.007695063698513857,
# 0.0036247404439402418,
# 0.000294488497628573,
# 0.00802834754913184,
# 0.06473976949203958,
# 0.15019030807262782,
# 0.1544894629870346,
# 0.10652924185943266,
# 0.056092399431077154,
# 0.017591649622743843,
# 0.011081129507255904,
# 0.03377243008151059,
# 0.052315961861077724,
# 0.06648425396123718,
# 0.07492302706251003,
# 0.07652398064721888,
# 0.07155302020573483,
# 0.06176470500859383,
# 0.049579523433481765,
# 0.037148275114093035,
# 0.025896388413758728,
# 0.007609690428454395,
# 0.009255078084002236,
# 0.003956803140331835,
# 0.00037677089248900795,
# 0.0018087372022305573,
# 0.0029303703506087762,
# 0.0032929264228946913,
# 0.003157267698608478,
# 0.0027325684006771272,
# 0.002175951108982646,
# 0.001597050916713156,
# 0.0003553904822568102,
# 0.0006187957631624824,
# 0.0002718416405877181,
# 2.3219019951182093e-05,
# 0.0001380249097419567,
# 0.00022770221412200876,
# 0.00026302779082244506,
# 0.00026033491296198816,
# 0.00023374547793325853,
# 0.00019456619873145332,
# 0.00015119673473089964,
# 0.00010966801476715635,
# 1.7904619342978318e-05,
# 4.4706457882829344e-05,
# 2.1266328915077492e-05,
# 3.3430345959741085e-06,
# 0.0]
#
#do1 = [5.0,
# 5.024937810560445,
# 5.0990195135927845,
# 5.220153254455275,
# 5.385164807134504,
# 5.5901699437494745,
# 5.830951894845301,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 5.989824459887481,
# 5.931179032170065,
# 5.803609062952554,
# 5.59529089629972,
# 5.3164824087411295,
# 4.99156659970612,
# 4.648395662601892,
# 4.312072710957213,
# 4.00191237913988,
# 3.7305360070033404,
# 3.5042787220766556,
# 3.3242942883975077,
# 3.1879265448391076,
# 3.0900673943343495,
# 3.0243392427669047,
# 2.984026473993109,
# 2.8547969569342535,
# 2.5361927661803003,
# 2.2757794628031904,
# 2.1038808494477075,
# 2.0597270460751833,
# 2.1421672007521515,
# 2.1033689213368163,
# 2.174288115047889,
# 2.1199233882335258,
# 2.1741204562252094,
# 2.105806617150595,
# 2.1507752118006205,
# 2.076146041250501,
# 2.1200314723966955,
# 2.040354528927823,
# 2.0498226745822516,
# 2.109388418861456,
# 2.2202170730646125,
# 2.4089849636549068,
# 2.6937777479907794,
# 3.0691595697732668,
# 3.5130817271506953,
# 4.000068961674601,
# 4.50922512125249,
# 5.0235397488150095,
# 5.508559793677338,
# 5.827454322733946,
# 5.922620409414818,
# 5.985554867919071,
# 5.976753742142011,
# 5.957751038016183,
# 5.951714671240193,
# 5.9539456430592015,
# 5.960804620230265,
# 5.969642679346534,
# 5.978669213936645,
# 5.986790964139382,
# 5.993447011712374,
# 5.998456221843423,
# 5.998113396820672,
# 5.9960493750672565,
# 5.995073349137079,
# 5.994895851576464,
# 5.9952510444300104,
# 5.995915635535423,
# 5.996716147846971,
# 5.997528269710971,
# 5.998271378877171,
# 5.998900636617969,
# 5.999398380212807,
# 5.999765956204905,
# 5.99998334060808,
# 5.999829920542951,
# 5.999752098850239,
# 5.99972911116528,
# 5.999743146540633,
# 5.999779535145129,
# 5.999826556914304,
# 5.999875250200821]
#
#s2 = [0,
# 0.5,
# 1.0,
# 1.5,
# 2.0,
# 2.5,
# 3.0,
# 3.5,
# 4.0,
# 4.5,
# 5.0,
# 5.5,
# 6.0,
# 6.5,
# 7.0,
# 7.5,
# 8.0,
# 8.5,
# 9.0,
# 9.5,
# 10.000232207617191,
# 10.503518276655313,
# 11.004982293064955,
# 11.507133695027177,
# 12.007263147878847,
# 12.507498751800865,
# 13.007785761464733,
# 13.507831490507417,
# 14.008838830324748,
# 14.509053044973053,
# 15.010197685248313,
# 15.510734877988083,
# 16.03018829058545,
# 16.58835072904607,
# 17.182759457423977,
# 17.797687006006544,
# 18.414808412221735,
# 19.01953342916277,
# 19.603444526104553,
# 20.16419602350076,
# 20.70397904087881,
# 21.227436796502023,
# 21.739753589275782,
# 22.244650332340633,
# 22.744936151198115,
# 23.24629148190163,
# 23.752733794677983,
# 24.265780427289815,
# 24.788951100118034,
# 25.323160793252455,
# 25.86707713876453,
# 26.423443334243995,
# 26.990681321183466,
# 27.562602938831173,
# 28.131729683591004,
# 28.691931318397042,
# 29.239748101748994,
# 29.774502808745257,
# 30.29762413993101,
# 30.81166044735056,
# 31.31937746707251,
# 31.82316209323318,
# 32.32476906684527,
# 32.825323526836755,
# 33.32545548067303,
# 33.82546611413694,
# 34.32546962368811,
# 34.82549128762271,
# 35.32552438041422,
# 35.82555728815846,
# 36.32558288542105,
# 36.82559937685333,
# 37.32560825145047,
# 37.82561211544391,
# 38.32561333826219,
# 38.8256135269732,
# 39.3256135285349,
# 39.825613654609754,
# 40.32561392985743,
# 40.82561427433254,
# 41.32561460411366,
# 41.82561486906657,
# 42.325615054696975,
# 42.82561516984687,
# 43.325615233138784,
# 43.82561526354858,
# 44.32561527584758,
# 44.82561527960732,
# 45.32561528007745,
# 45.825615279639,
# 46.32561527910772,
# 46.82561527862361,
# 47.325615278136816,
# 47.8256152776129,
# 48.32561527707937]
#
#do2 = [5.0,
# 5.024937810560445,
# 5.0990195135927845,
# 5.220153254455275,
# 5.385164807134504,
# 5.5901699437494745,
# 5.830951894845301,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 6.0,
# 5.9847366898312355,
# 5.927295168592382,
# 5.888981532659548,
# 5.842525288120273,
# 5.831123629832949,
# 5.815749246053057,
# 5.83266988902565,
# 5.8394092451490405,
# 5.871140689056829,
# 5.50119169783511,
# 5.000650357719122,
# 4.501197719468222,
# 4.007475139793584,
# 3.5345145227378243,
# 3.0057957429607076,
# 2.5059655026128436,
# 2.070069054961901,
# 1.7360869573090956,
# 1.543393849055941,
# 1.514367893799179,
# 1.632813394626977,
# 1.8563497922402097,
# 1.8994673683181154,
# 2.032077214145259,
# 1.9865701608675221,
# 2.0128095527978886,
# 1.8692545739666742,
# 1.8241401154443269,
# 1.600363777121431,
# 1.4981504468559044,
# 1.1981999569885808,
# 1.077229089955242,
# 1.2128164409690014,
# 1.5546234024063328,
# 2.004626364458352,
# 2.5026363111941525,
# 3.019123462586209,
# 3.531506582181113,
# 4.012531524077937,
# 4.504335038759627,
# 5.001196377501047,
# 5.500207270400904,
# 5.992091110554745,
# 5.9843785403080805,
# 5.972913806406913,
# 5.969675970953582,
# 5.9715726604429875,
# 5.976250385090325,
# 5.982026323741182,
# 5.987786155476943,
# 5.9928688103806635,
# 5.996953037771517,
# 5.99995533064231,
# 5.998055661382292,
# 5.996926443594429,
# 5.996468254231901,
# 5.996490897457394,
# 5.99682355904378,
# 5.997325547332144,
# 5.997889760955566,
# 5.998441330340683,
# 5.998933423703116,
# 5.9993417310246535,
# 5.999658694739591,
# 5.999888174284007,
# 5.999959073388091,
# 5.999868942714837,
# 5.99982656005623,
# 5.999817995310095,
# 5.999831254631005,
# 5.999856712746239,
# 5.99988711736114,
# 5.999917334148861,
# 5.999944049885688,
# 5.999965475586351]
#
#curvature2 = [0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.0,
# 0.03051718643966383,
# 0.08382608120554422,
# 0.037656623194204916,
# 0.016121604498955767,
# 0.06954825722749579,
# 0.007937681476941724,
# 0.003087742174197106,
# 0.020338760595993252,
# 0.04989525655332513,
# 0.034088397424881935,
# 0.03831725716194767,
# 0.02112549357353682,
# 0.2279601544036403,
# 0.1790699027055186,
# 0.0992302986457876,
# 0.042043796347451784,
# 0.004023665830095613,
# 0.023429741270783826,
# 0.045159754005103955,
# 0.062464699838583895,
# 0.07442354086669066,
# 0.079498844613053,
# 0.07716600194371177,
# 0.0784021525589354,
# 0.10455763608105115,
# 0.039626988733831796,
# 0.08589386900716305,
# 0.06548398270406876,
# 0.07087936438657906,
# 0.05837952333069613,
# 0.041919448681294845,
# 0.04543305828607853,
# 0.03398532954566114,
# 0.013287159187841638,
# 0.0077938347178442245,
# 0.026481190551769816,
# 0.04189602206376541,
# 0.05330655302687475,
# 0.059925504271692284,
# 0.06135861785838269,
# 0.058024138978487697,
# 0.05113274766994538,
# 0.04226553751204229,
# 0.03288909070967558,
# 0.02407377792200747,
# 0.016445507711812926,
# 0.00268215995709507,
# 0.005561779930907214,
# 0.002196094102043734,
# 3.2207446711085974e-05,
# 0.0013541064978633297,
# 0.001996586058934436,
# 0.002163689590625724,
# 0.0020264835791713996,
# 0.0017195504396175767,
# 0.0013420501826747117,
# 0.0008710917278218789,
# 0.0006200366013044152,
# 0.0003386530532186848,
# 0.0001244504476754351,
# 2.5288425085990383e-05,
# 0.0001189518257020765,
# 0.0001675718427709375,
# 0.00018268703649207876,
# 0.00017496823919862058,
# 0.0001534543854479557,
# 0.00012524329130756117,
# 9.549602402400281e-05,
# 6.763582391213345e-05,
# 9.389149581606891e-06,
# 2.439758871971141e-05,
# 9.892999362745428e-06,
# 3.756543596400644e-07,
# 7.002101810315085e-06,
# 1.0580072360175706e-05,
# 0.0]
#
#plt.figure(5)
#plt.plot(s2, curvature2, "--d", color = 'm',markersize = 3)
#plt.plot(s1, curvature1, "-o", color = 'blue',markersize = 3)
#plt.legend(['Sampling-based','Ours'], prop=font1, loc=2, bbox_to_anchor=(0.42,1.30),borderaxespad = 0.)
#plt.xlabel('s (m)',font1)
#plt.ylabel('Curvature (m${}^{-1}$)',font1)
#plt.xlim(0, 50)
#plt.tick_params(labelsize=scriptSize)
##plt.savefig('Curvature.eps', format='eps', bbox_inches='tight')
#plt.grid(False)
#plt.show()
#
#plt.figure(6)
#plt.plot(s2, do2, "--d", color = 'm', markersize = 3)
#plt.plot(s1, do1, "-o", color = 'blue', markersize = 3)
#plt.legend(['Sampling-based', 'Ours'], prop=font1, loc=2, bbox_to_anchor=(0.42,1.30),borderaxespad = 0.)
#plt.xlabel('s (m)',font1)
#plt.ylabel('Distance (m)',font1)
#plt.xlim(0, 50)
#plt.ylim(1, 6)
#plt.tick_params(labelsize=scriptSize)
##plt.savefig('ClosestObstacleDistance.eps', format='eps', bbox_inches='tight')
#plt.grid(False)
plt.show()


    

if __name__ == '__main__':
    main()