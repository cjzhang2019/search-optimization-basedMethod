"""
Created on Wed May  6 16:58:57 2020
using hybrid A* and mobile model to solve planning problem
@author: cjzhang
"""
import heapq as hq
import math
import matplotlib.pyplot as plt
import numpy as np
import time

# total cost f(n) = actual cost g(n) + heuristic cost h(n)

def collision_check(neighbour_x_cts, neighbour_y_cts, neighbour_theta_cts, obstacles):
    minDistance = 1000000
    for obstacle in obstacles:
        tempDistance = math.sqrt((neighbour_x_cts-obstacle[0]) ** 2 + (neighbour_y_cts - obstacle[1]) ** 2)
        if tempDistance < minDistance:
            minDistance = tempDistance
    return minDistance

class hybrid_a_star:
    def __init__(self, min_x, max_x, min_y, max_y, obstacle = [], vehicle_length = 2):

        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.obstacle = obstacle
        self.vehicle_length = vehicle_length


    def euc_dist(self, position, target):
        output = np.sqrt(((position[0] - target[0]) ** 2) + ((position[1] - target[1]) ** 2) + (math.radians(position[2]) - math.radians(target[2])) ** 2)
        return float(output)
    
    def find_path(self, start, end):
        steering_inputs = [-40, 0, 40]
        cost_steering_inputs= [0.1, 0, 0.1]
        
        speed_inputs = [-1,1]
        cost_speed_inputs = [1,0]
        deltaT = 1
        
        start = (float(start[0]), float(start[1]), float(start[2]))
        end = (float(end[0]), float(end[1]), float(end[2]))
        # Not only XY but also heading
        
        open_heap = [] # element of this list is like (cost,node_d)
        open_diction = {} #  element of this is like node_d:(cost,node_c,(parent_d,parent_c))
        
        visited_diction = {} #  element of this is like node_d:(cost,node_c,(parent_d,parent_c))
        
        obstacles = set(self.obstacle)
        cost_to_neighbour_from_start = 0

        hq.heappush(open_heap,(cost_to_neighbour_from_start + self.euc_dist(start, end),start))
        
        open_diction[start] = (cost_to_neighbour_from_start + self.euc_dist(start, end), start,(start,start))
        count = 0
        while len(open_heap) > 0:
            chosen_d_node =  open_heap[0][1]
            chosen_node_total_cost = open_heap[0][0]
            chosen_c_node = open_diction[chosen_d_node][1]

            visited_diction[chosen_d_node] = open_diction[chosen_d_node]
            
            if self.euc_dist(chosen_d_node, end) < 1:
                rev_final_path = [end] # reverse of final path
                node = chosen_d_node
                while (1):
                    open_node_contents = visited_diction[node] # (cost,node_c,(parent_d,parent_c))                   
                    parent_of_node = open_node_contents[2][1]
                    
                    rev_final_path.append(parent_of_node)
                    node = open_node_contents[2][0]
                    if node == start:
                        rev_final_path.append(start)
                        break
                return rev_final_path
                

            hq.heappop(open_heap)
            
            for i in range(0,3):
                for j in range(0,2):
                    
                    delta = steering_inputs[i]
                    velocity = speed_inputs[j]
                    
                    
                    cost_to_neighbour_from_start =  chosen_node_total_cost - self.euc_dist(chosen_d_node, end)
                    
                    neighbour_x_cts = chosen_c_node[0] + (velocity * deltaT * math.cos(math.radians(chosen_c_node[2]))) 
                    neighbour_y_cts = chosen_c_node[1]  + (velocity * deltaT * math.sin(math.radians(chosen_c_node[2])))
                    neighbour_theta_cts = math.radians(chosen_c_node[2]) + (velocity * deltaT * math.tan(math.radians(delta))/(float(self.vehicle_length)))
                    
                    neighbour_theta_cts = math.degrees(neighbour_theta_cts)
                    #return angel
                    
                    neighbour_x_d = round(neighbour_x_cts)
                    neighbour_y_d = round(neighbour_y_cts)
                    neighbour_theta_d = round(neighbour_theta_cts)
                    #Rounding value
                    
                    
                    neighbour = ((neighbour_x_d,neighbour_y_d,neighbour_theta_d),(neighbour_x_cts,neighbour_y_cts,neighbour_theta_cts))
                    minDistanceFromObstacles = collision_check(neighbour_x_cts, neighbour_y_cts, neighbour_theta_cts, obstacles)
                    if ((minDistanceFromObstacles > 1.4) and (neighbour_x_d >= self.min_x) and (neighbour_x_d <= self.max_x) and (neighbour_y_d >= self.min_y) and (neighbour_y_d <= self.max_y)):
                        heurestic = self.euc_dist((neighbour_x_d,neighbour_y_d,neighbour_theta_d),end)
                        cost_to_neighbour_from_start = abs(velocity) + cost_to_neighbour_from_start + cost_steering_inputs[i] + cost_speed_inputs[j]
                        
                        #print(heurestic,cost_to_neighbour_from_start)
                        total_cost = heurestic+cost_to_neighbour_from_start
                        
                        # If the cost of going to this successor happens to be more
                        # than an already existing path in the open list to this successor,
                        # skip this successor
                        
                        
                                            
                        skip=0
                        #print(open_set_sorted)
                        # If the cost of going to this successor happens to be more
                        # than an already existing path in the open list to this successor,
                        # skip this successor
                        found_lower_cost_path_in_open=0 
                        
                        if neighbour[0] in open_diction:
                            
                            if total_cost>open_diction[neighbour[0]][0]: 
                                skip=1
                                
                            elif neighbour[0] in visited_diction:
                                
                                if total_cost>visited_diction[neighbour[0]][0]:
                                    found_lower_cost_path_in_open=1
                                    
                         
                        if skip==0 and found_lower_cost_path_in_open==0:
                            
                            hq.heappush(open_heap,(total_cost,neighbour[0]))
                            open_diction[neighbour[0]]=(total_cost,neighbour[1],(chosen_d_node,chosen_c_node))

        print("Did not find the goal - it's unattainable.")
        return []

def main():
    print(__file__ + " start!!")
    start = time.clock()

    # start and goal position
    #(x, y, theta) in meters, meters, degrees
    sx, sy, stheta= 1, -10, 90
    
    gx, gy, gtheta = -3, 19, 90

    #create obstacles
    obstacle = []

    obstacle.append((-2,-6))
    obstacle.append((4,-6))
    obstacle.append((-3,-4))
    obstacle.append((3,-4))
    obstacle.append((-4,-2))
    obstacle.append((2,-2))
    obstacle.append((-5,0))
    obstacle.append((1,0))
    obstacle.append((-5,2))
    obstacle.append((1,2))
    obstacle.append((-4,4))
    obstacle.append((2,4))
    obstacle.append((-3,6))
    obstacle.append((3,6))
    obstacle.append((-2,8))
    obstacle.append((4,8))
    obstacle.append((-1,10))
    obstacle.append((5,10))
    obstacle.append((0,12))
    obstacle.append((6,12))
    
    obstacle.append((-10,-10))
    obstacle.append((-10,20))
    obstacle.append((10,20))
    obstacle.append((10,-10))

    
    ox, oy = [], []
    for (x,y) in obstacle:
        ox.append(x)
        oy.append(y)

    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "xr")
    plt.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")

    hy_a_star = hybrid_a_star(-10, 10, -20, 20, obstacle = obstacle, vehicle_length = 2)
    path = hy_a_star.find_path((sx,sy,stheta), (gx,gy,gtheta))
    print(path)

    rx, ry = [], []
    for node in path:
        rx.append(node[0])
        ry.append(node[1])
    font1 = {'family' : 'Times New Roman','weight' : 'normal', 'size'   : 20}
    plt.plot(rx, ry, "-r")
    plt.xlabel('x/(m)',font1)
    plt.ylabel('y/(m)',font1)
    plt.legend(['Obstacles', 'VehiclePose', 'Destination', 'Path'],prop=font1, bbox_to_anchor=(1.05, 0), loc=3, borderaxespad = 0)
    plt.title('Path Planning Using Hybrid A Star\n(Add Heading Information)', font1)
    plt.savefig("1.png")
    plt.show()
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)


if __name__ == '__main__':
    main()
