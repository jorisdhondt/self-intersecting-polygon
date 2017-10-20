from __future__ import division 
from os.path import expanduser
from ast import literal_eval
from collections import defaultdict

import shutil, argparse as arg
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import numpy as np


#Function calculates the area described by <coordinates> using the shoelace formula
def calculate_area(coordinates):
    x, y = zip(*coordinates)
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))  #implementation of shoelace formula


#Function calculates the coefficients of the general equation form of a line (segment).  The general equation is in the form Ax + By + C = 0
def line_coefficients(inputSegment):
    
    startVertex = inputSegment[0]
    endVertex = inputSegment[1]  
    
    #obtain the coefficients of the general linear equation:
    #slope is defined as -A/B  or  y2-y1/(x2-x1)
    #this makes A = y1 - y2  and B = x2 - x1
    # (x1,y1) represent the startVertex
    # (x2,y2) represent the endVertex
    A = startVertex[1] - endVertex[1] 
    B = endVertex[0] - startVertex[0]
    
    # Replacing A & B in the general equation with these the A & B notations, C can be obtained
    C = startVertex[0]*endVertex[1] - endVertex[0]*startVertex[1]
    return A, B, C

#Function calculates the coefficients of the general equation form of a line (segment).  The general equation is in the form Ax + By + C = 0
def intersect_segments(S1,S2):
    
    startVertexS1 = S1[0]
    endVertexS1 = S1[1]
    startVertexS2 = S2[0]
    endVertexS2 = S2[1]
    
    A1 = endVertexS1[0] - startVertexS1[0]
    B1 = startVertexS2[0] - endVertexS2[0]
    C1 = startVertexS1[0] - startVertexS2[0]
    
    A2 = endVertexS1[1] - startVertexS1[1]
    B2 = startVertexS2[1] - endVertexS2[1]
    C2 = startVertexS1[1] - startVertexS2[1]    

    denominator  = A1*B2 - A2*B1 
    numeratorX = C2*B1 - C1*B2 
    numeratorY = A2*C1 - A1*C2
    
    if denominator != 0:
        s = round(numeratorX / denominator,3)
        t = round(numeratorY / denominator,3)
        if (s > 0 and s < 1 and t > 0 and t < 1):
            x = startVertexS1[0] + s*(endVertexS1[0] - startVertexS1[0])
            y = startVertexS1[1] + s*(endVertexS1[1] - startVertexS1[1])
            return (round(x,3),round(y,3))
        else:
            return None
    else: # parallel
        return None 

#Function returns the (x,y) intersection point of two lines L1 and L2.  Both lines are described by their (A,B,C) coefficients. 
#The implementation is based on Cramer's rule:
# A1*x + B1*y + C = 0   (L1)
# A2*x + B2*y + C = 0   (L2)
#
# Intersection of both lines is defined as:
# x = (C2 * B1 - C1 * B2) / (A1 * B2 - A2 * B1)
# y = (A2 * C1 - A1 * C2) / (A1 * B1 - A2 * B1)
#
# if L1 & L2 are parellel, the denominator equals to zero 


def intersect_lines(L1, L2):
    denominator  = L1[0] * L2[1] - L2[0] * L1[1]  #L1 is a tuple with form (A1,B1,C1), similar for L2
    numeratorX = L1[1] * L2[2] - L1[2] * L2[1]
    numeratorY = L1[2] * L2[0] - L1[0] * L2[2] 
    if denominator != 0:
        x = round(numeratorX / denominator,2)
        y = round(numeratorY / denominator,2)
        return (x,y)
    else: # parallel
        return None 

def valid_vertex(vertex,S1,S2):
    #L1 has format [(x1,y1),(x2,y2)]
    #L2 has format [(x1,y1),(x2,y2)]
    #p has format (x,y)
    if p == None or p == S1[0] or p == S1[1]:
        return False
    if p[0] >= min(S1, key = lambda t:t[0])[0] and p[0] <= max(S1, key = lambda t:t[0])[0] and p[1] >= min(S1, key = lambda t:t[1])[1] and p[1] <= max(S1, key = lambda t:t[1])[1] and \
    p[0] >= min(S2, key = lambda t:t[0])[0] and p[0] <= max(S2, key = lambda t:t[0])[0] and p[1] >= min(S2, key = lambda t:t[1])[1] and p[1] <= max(S2, key = lambda t:t[1])[1]:    
        return True
    else:
        return False

def find_loops(adj,vertex,visited=None,path=None):
    if visited is None: visited = []
    if path is None: path = [vertex]
    visited.append(vertex)
    loops = []
    for neighbour in adj[vertex]:
        if neighbour not in visited or neighbour == path[0]:
            new_path = path + [neighbour]
            if neighbour == new_path[0]:
                loops.append(tuple(new_path))
            else:    
                loops.extend(find_loops(adj, neighbour, visited[:], new_path))
    return loops

        
def main(path):
    segments = [literal_eval(p.strip()) for p in open(path).readlines()]
    #segments = []
    #with open(path) as f:
        #for line in f:
            #pair = literal_eval(line)
            #segments.append(pair)    
    
    
    #######################
    #
    #FOR DEBUGGING PURPOSES  <START>
    #
    #This block plots the line segments provided in <segments>
    #######################

    #plotSegments = [[p,q] for p,q in [segments[z] for z in range(len(segments))]]
    
    #points = []
    #for i in range(len(plotSegments)):
        #startPoint = plotSegments[i][0]
        #endPoint = plotSegments[i][1]
        #if (len(points) == 0 or points[-1] != startPoint):
            #points.append(startPoint)
        #points.append(endPoint)    
    
    #lc = mc.LineCollection(plotSegments, linewidths=2)
    
    #fig, ax = plt.subplots()
    #ax.add_collection(lc)    
    #ax.autoscale()
    #for pt in points:
        #plt.annotate(str(pt),xy=pt)
    #plt.show()

    #######################
    #
    #FOR DEBUGGING PURPOSES  <END>
    #
    #######################    
    
    #double iteration to check if there is an intersection between every pair of segments
    i = 0
    while i < len(segments):
        startPointL1 = segments[i][0]
        endPointL1 = segments[i][1]
        j = i+1
        while j<len(segments):
            startPointL2 = segments[j][0]
            endPointL2 = segments[j][1]            

            intersectPoint = intersect_segments(segments[i],segments[j])

            #if intersection is found, the two intersecting segments are replaced with four new segments: the intersecting point is now the end or start of each segment preserving the original direction and position in the <segments> array
            if intersectPoint is not None:
                del segments[j]
                segments.insert(j,(intersectPoint,endPointL2))
                segments.insert(j,(startPointL2,intersectPoint))
                del segments[i]
                segments.insert(i,(intersectPoint,endPointL1))
                segments.insert(i,(startPointL1,intersectPoint))                
                j = i+1
            else:
                j = j+1
        i = i+1
        
    adj = defaultdict(set)
    vertices = []
    for i in range(len(segments)):
        startPoint = segments[i][0]
        endPoint = segments[i][1]
        adj[startPoint].add(endPoint)
        
        if (len(vertices) == 0 or vertices[-1] != startPoint):
            vertices.append(startPoint)
        vertices.append(endPoint)
        
    loops = []
    for vertex in vertices:
        vertexLoops = find_loops(adj, vertex)
        if vertexLoops:
            maxLength   = max(len(p) for p in vertexLoops)
            maxLoops = [p for p in vertexLoops if len(p) == maxLength]
            maxLoop = max(maxLoops,key=len)
            loops.append(maxLoop)
            for i in range(len(maxLoop)):
                for j in range(len(maxLoop)):
                    if maxLoop[j] in adj[maxLoop[i]]:
                        adj[maxLoop[i]].remove(maxLoop[j])           
    
    surfaceArea = 0
    for i in loops:
        surfaceArea += calculate_area(i)
        
    
    
    print("Total covered surface is "+str(surfaceArea))
    return surfaceArea
    
    
if __name__ == '__main__':
    parser = arg.ArgumentParser(description='Area calculater for irregular polygon')
    parser.add_argument('path', help='path to a ASCII file listing the ordered cartesian coordinates, one per line in format (x,y)')

    args = parser.parse_args()
    path = args.path

    
    main(path)    
    
