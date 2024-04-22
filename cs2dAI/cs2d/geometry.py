import numpy as np
import math

def pointInRect(x,y,Rect):
    return y >= Rect[1] and y <= Rect[1]+Rect[3] and x >= Rect[0] and x <= Rect[0]+Rect[2]
         

    
def circleRectIntersect(circle,Rect):
    if pointInRect(circle[0],circle[1],Rect):
        return True
    return False

def angleToVec(angle):
    return np.array([math.sin(angle),math.cos(angle)])

def lineRectIntersect(line,rect):
    pass
    
def lineLineIntersect(lineA,lineB):
    #build direction Vectors
    dirA = -np.array([lineA[0]-lineA[2],lineA[1]-lineA[3]])
    dirB = -np.array([lineB[0]-lineB[2],lineB[1]-lineB[3]])
    # #solve ax=b
    b=np.array([lineA[0]-lineB[0],lineA[1]-lineB[1]])
    
    a = np.array([dirB,dirA]).T
  
    
    if np.linalg.det(a) == 0:
        return []
    
    sol = np.linalg.solve(a,b)
  
    t_B = sol[0]
    t_A = -sol[1]
    #this way too slow (learned about branch predriction)
    if t_A > 1 or t_B > 1 or t_A < 0 or t_B < 0:
        return []
    
    uA = np.array([lineA[0],lineA[1]])

    return uA+t_A*dirA
    
def rayRayIntersect(lineA,lineB):
    #build direction Vectors
    dirA = -np.array([lineA[0]-lineA[2],lineA[1]-lineA[3]])
    dirB = -np.array([lineB[0]-lineB[2],lineB[1]-lineB[3]])
    # #solve ax=b
    b=np.array([lineA[0]-lineB[0],lineA[1]-lineB[1]])
    
    a = np.array([dirB,dirA]).T
  
    
    if np.linalg.det(a) == 0:
        return []
    
    sol = np.linalg.solve(a,b)
  
    t_B = sol[0]
    t_A = -sol[1]
 
    uA = np.array([lineA[0],lineA[1]])

    return uA+t_A*dirA

    
def lineRectIntersect(line,rect):
    #use linelineIntersect on 4 sides
    s1 = [rect[0],rect[1],rect[0]+rect[2],rect[1]]
    s2 = [rect[0],rect[1],rect[0],rect[1]+rect[3]]
    s3 = [rect[0]+rect[2],rect[1],rect[0]+rect[2],rect[1]+rect[3]]
    s4 = [rect[0],rect[1]+rect[3],rect[0]+rect[2],rect[1]+rect[3]]
    
    sides = [s1,s2,s3,s4]
    
    
    
    schnittpunkte = []
    for side in sides:
        sp = lineLineIntersect(line,side)
        if len(sp) > 0:
            schnittpunkte.append(sp)
    
    return schnittpunkte
    
def fastLineRectIntersect(ro,rd):
    pass
    # source https://tavianator.com/2011/ray_box.html
    #wtf
    
    
 

    