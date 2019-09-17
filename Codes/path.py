#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:36:11 2019

@author: jeevan
"""
import numpy as np
import pygame
import matplotlib.pyplot as plt

class Game():
    def __init__(self,viewer_xy):
        pygame.init()
        self.viewer_xy = viewer_xy
        self.screen = pygame.display.set_mode(self.viewer_xy)
        pygame.display.set_caption('Pathplanning')
        self.white = 255,255,255
        self.green = 0,255,0
        self.blue = 0,0,255
        self.red=255,0,0
        self.purple=127,0,255
        self.pink=255,0,255
        self.yellow=215,215,0
        self.orange=255,128,0
        self.grey=230,230,230
#        self.color=[self.green,self.orange,self.yellow,self.purple,self.blue,self.blue,self.red]
        self.color=[self.grey,self.blue,self.blue,self.blue,\
                    self.blue,self.blue,self.blue,self.blue,\
                    self.blue,self.red]
        self.screen.fill(self.white)
        self.clock = pygame.time.Clock()

class Env(object):
    action_dim=5
    state_dim=[100,130,2]
    process_noise_variance=1
    #Pyglet functions
    viewer=None
    viewer_xy = (1900,400)
    action_bound = [-1, 1]
    actions=np.array([[-1,0],[1,0],[0,0],[0,2],[0,-3]])
    episodes=10
    #New parameters
    #x(0),y(1),theta(2),w(3),l(4)
    #speed_y(5),level(6),identity(7),speed_x(8)),dist_from_agent(9) Accoding to paper
    leader_speed=4
    agent_init_speed=0
    w=48
    l=48
    agent_car=9
    car0=np.array([520, 200, 0,w,l,leader_speed,0,0,0,0], dtype=np.float64)   # car coordination
    car1=np.array([100,150,0,w,l,5,0,1,0,0], dtype=np.float64)
    car2=np.array([200,250,0,w,l,5,0,2,0,0], dtype=np.float64)
    car3=np.array([350,350,0,w,l,5,0,3,0,0], dtype=np.float64)
    car4=np.array([350,350,0,w,l,5,0,4,0,0], dtype=np.float64)
    car5=np.array([350,350,0,w,l,5,0,5,0,0], dtype=np.float64)
    car6=np.array([350,350,0,w,l,5,0,5,0,0], dtype=np.float64)
    car7=np.array([350,350,0,w,l,5,0,5,0,0], dtype=np.float64)
    car8=np.array([350,350,0,w,l,5,0,5,0,0], dtype=np.float64)
    car9=np.array([70,210,0,w,l,agent_init_speed,0,agent_car,0,0], dtype=np.float64)
    agent_speed=6
    agent_steer=0
    agent_max_speed=12
    car=np.array([car0,car1,car2,car3,car4,car5,car6,car7,car8,car9])
    scale_acc=0.5
    no_cars=10
    a=np.array([[-1,0],[-1,1],[-1,-1],[1,0],[1,1],[1,-1],[0,0],[0,1],[0,-1]])
    #Y-Direction
#    by=-0.01
#    ky=0.01
#    bx=0.01100
#    kx=0.01

    by=-0.005
    ky=0.005
#    bx=0.05
#    kx=0.02
    bx=0.05
    kx=0.001

    
    k_bar=50
    gy=-100
    dt=1
    acc=0
    N=1
    level_check_gap=30
    step1=0
    
    x_cone=[40,140]
    x_speed_max=3
    y_speed_max=15
    ################################# Initialization var
    x_min=100
    y_min=50
    min_gap=90
    ################################ RL variables
    end_episode = False
    roadside_crash=False
    collide=False
    collide_dist=60
    near_collision=False
    ############################# Stats
    state_cars=[]
    theta_state=2.15

#Pygame var
    car_vert=np.array([])
    car_holder=np.array([])

    def __init__(self):
        self.game=Game(self.viewer_xy)
        
    def step(self,action):
#        print(action)
    # Agent car
        ac=self.agent_car
        self.step1+=1
        steer,acc_y=self.actions[action]
        
# Steering     
        self.agent_steer += steer * np.pi/60#180/60=3 degrees
        
        if (self.agent_steer>np.pi/6): self.agent_steer=np.pi/6
        elif (self.agent_steer<(-np.pi/6)): self.agent_steer=-np.pi/6
        
        self.car[ac,2]+=self.dt*self.agent_speed*np.tan(self.agent_steer)/self.l
#        self.car[ac,2] += steer * np.pi/60# 180/60=3 degrees
        if self.car[ac,2] <-np.pi/4:   self.car[ac,2]=-np.pi/4## Restrict rotation to +-45 degrees..
        elif self.car[ac,2] >+np.pi/4: self.car[ac,2]=+np.pi/4
        
#Speed
        self.agent_speed+=acc_y*self.dt
        if self.agent_speed<0: self.agent_speed=0
        if self.agent_speed>self.agent_max_speed: self.agent_speed=12
# Y_acc
        self.car[ac,5]=self.agent_speed*np.cos(self.car[ac,2])
        self.car[ac,8]=self.agent_speed*np.sin(self.car[ac,2])
        self.car[ac,:2] = self.car[ac,:2] + \
                            self.agent_speed * self.dt * np.array([np.cos(self.car[ac,2]), np.sin(self.car[ac,2])])
                 
    #Leader car
        if self.step1 % 10 ==0:
            self.car[0,5]+=np.random.uniform(-1,1)#3,-3
        self.car[0,5]=np.clip(self.car[0,5],3,8)#10
        self.car[0,0]+=self.car[0,5]*self.dt
    #Traffic network
        self.y_acc_cal()
        self.x_acc_cal()
    #Graphics
        self.car_holder=np.array([self.car[0][:5],self.car[1][:5],self.car[2][:5],self.car[3][:5]\
                                  ,self.car[4][:5],self.car[5][:5],self.car[6][:5] \
                                  ,self.car[7][:5],self.car[8][:5],self.car[9][:5] \
                                  ])
        self.car_vert=np.array([])
        ob = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
        for i in range(self.no_cars):
            ob[i] = self.vertices(self.car_holder[i],i)
        self.car_vert=([ob[0],ob[1],ob[2],ob[3],ob[4],ob[5],ob[6],ob[7],ob[8],ob[9]])
    #Graphics end
        
        self.collision()
        r=self.reward(action)
        self.check_end_episode()
        if self.collide==True or self.roadside_crash==True or self.end_episode==True:
            done=True
        else:done= False
##        
#        s=self.get_state()
        s=self.CNN_image()
        s=s.reshape(100,130)#,1)
        s2=self.agent_states()
#        print(s.shape)
        return s,s2,r,done
    
    def y_level_assignment(self):
        
        car=self.car
#        n=self.no_cars
        dist_sorted=car[np.argsort(car[:, 0])]
        leader_car_index=np.where(dist_sorted[:,7]==0)
        agent_car_index=np.where(dist_sorted[:,7]==self.agent_car)#Removed bug
        dist_sorted[leader_car_index,6]=0 #  Assign level 0 to the leader car
#        print("DS.",dist_sorted)
#        Leader
    # The car after leader become level 1 by default
        #print(dist_sorted)
#        print(leader_car_index[0],"Leadere car inde")
        dist_sorted[leader_car_index[0]-1,6]=1 # Assign level 1   
#        level_leader_dist=dist_sorted[np.where(dist_sorted[:,7]==0),1]
        level_leader_dist=dist_sorted[leader_car_index[0]-1,0] #Level 1 leader distance
        current_level=1
        for i in range(leader_car_index[0][0]-2, -1, -1):
        
            if dist_sorted[i][0]>level_leader_dist-self.level_check_gap:
                dist_sorted[i][6]=current_level
            else :
                current_level+=1
                dist_sorted[i,6]=current_level
                level_leader_dist=dist_sorted[i,0]
#        print("dist_sorted=\n", dist_sorted)
               
        if dist_sorted[agent_car_index[0],0] > dist_sorted[leader_car_index[0],0] :
            dist_sorted[agent_car_index[0],6]=-1
                
        return dist_sorted
    
    def y_acc_cal(self):
        by=self.by
        gy=self.gy
        k_bar=self.k_bar
        ky=self.ky
        
#        Call function for assigning level
        ds=self.y_level_assignment()
        w=1
        acc=0
        
        
#        print("Levels",ds[:,6])
        for j1 in range(int(max(ds[:,6]))+1): #generate levels
#            print(j1,"j1")
            if j1==0:continue
            level_indices=np.where(ds[:,6]==j1)
            for j in level_indices[0]: #Different cars in each level.
                # If agent then ignore calculations
                if ds[j,7]==self.agent_car:continue
                vx=ds[j][5] #Velocity of the vehicle whose acc is to be calculated
                x=ds[j][0]
                
                if j1==1:
                    i=np.where(ds[:,7]==0)[0][0]
                    x1=by*w*(ds[i][5]-vx)
                    x2=w*(ds[i][0]-x)
                    x3=(1)*(gy-k_bar*vx)
                    acc=-(x1-ky*(x2+x3)) 
                    ds[j][5]=ds[j][5]+self.dt*acc
                    if ds[j][5]>self.y_speed_max:ds[j][5]=self.y_speed_max
                    ds[j][0]+=ds[j][5]*self.dt
                    
                else:
                    current_car=j
                    next_level_los=[]
                    #####################################################Correct till ehre

                    for j2 in range(1,int(max(ds[:,6]))):
                        next_level_indices=np.where(ds[:,6]==j1-j2)[0]
#                        print(next_level_indices,"next_level_indices")
                        next_level_los=self.y_los(current_car,next_level_indices,ds)
                        next_level_los=next_level_los.astype(int)
                        if len(next_level_los)!=0:break
                    if len(next_level_los)==0: #if level 1 los is empty
                        next_level_los=[3] #leader car
                    w=1/len(next_level_los)
#                    if len(next_level_indices)==3:
#                        print(w,"w\n")
#                        print(next_level_indices,"nxt\n")
#                        print(ds,"ds\n")
                    acc=0
                    for i in next_level_los:
                        x1=by*w*(ds[i][5]-vx)
                        x2=w*(ds[i][0]-x)
                        x3=(1/len(next_level_los))*(gy-k_bar*vx)
                        acc+=-(x1-ky*(x2+x3)) 
#                    if j1==2:print("acc_y",acc)
                    ds[j][5]=ds[j][5]+self.dt*acc
                    if ds[j][5]>self.y_speed_max:ds[j][5]=self.y_speed_max
                    ds[j][0]+=ds[j][5]*self.dt 
        self.car=ds[np.argsort(ds[:, 7])]
        
    def y_los(self,current_car,next_level_indices,ds):
        c1=ds[current_car][:2]
#        print(current_car,"current_car")
#        print(c1,"c1")
        next_level_los=np.array([])
        for i in next_level_indices:
#            print(i,"i")
            c2=ds[i][:2]
#            print(c2,"c2")
            y=c2-c1
            theta=np.arctan2(y[1], y[0]) * 180 / np.pi
            if theta>-30 and theta<30:
                next_level_los=np.append(next_level_los,i)
#                24.9
        return next_level_los

## X-Dynamics modelling
              
    def x_acc_cal(self):
        
        bx=self.bx
        kx=self.kx
        road_width=self.viewer_xy[1]
        car=self.car
        #J is the current car in calculation
        for j in range(self.no_cars):
            
#        for j in range(self.no_cars-2):
#            print("_______________________________________________________")
            if j==0 or j==self.agent_car:continue
            vx=car[j][8]
            x=car[j][1]
#            print("\nvx and x,j",vx,x,j)
            #LOS check
            current_car=j
            xl_los_indices=self.xl_los(current_car)
            xl_los_indices=xl_los_indices.astype(int)
            xr_los_indices=self.xr_los(current_car)
            xr_los_indices=xr_los_indices.astype(int)
#            print("xl_los_indices",xl_los_indices)
#            print("xr_los_indices",xr_los_indices)
            #Continued
            wl=len(xl_los_indices)
            if wl==0:wl=1
            wr=len(xr_los_indices)
            if wr==0:wr=1
            acc=0 # reset acc values
            w=1/(wl+wr)
    #Left        
            if len(xl_los_indices)==0:
                x1=bx*w*(0-vx)
                x2=kx*w*(road_width-x)
                acc+=(x1+x2) 
#                print("xl",w)
            else:
                for i in xl_los_indices:
                    x1=bx*w*(car[i][8]-vx)
#                    print("x1",x1)
                    x2=kx*w*(car[i][1]-x)
#                    print("x2",x2)
                    acc+=(x1+x2)
#                    print("acc",acc)
#                print("xl_car",acc)
     #Right
            if len(xr_los_indices)==0:
                x1=bx*w*(0-vx)
                x2=kx*w*(0-x)
                acc+=(x1+x2) 
            else:
                for i in xr_los_indices:
                    x1=bx*w*(car[i][8]-vx)
                    x2=kx*w*(car[i][1]-x)
                    acc+=(x1+x2) 
#            print("acc_x",acc)
            car[j][8]=car[j][8]+self.dt*acc
            if car[j][8]>self.y_speed_max:car[j][8]=self.x_speed_max
            car[j][1]+=car[j][8]*self.dt            
        self.car=car
#        print("Car_1",car[1][8])
#        print("\nCar_2",car[2][8])
#        print("\nCar_3",car[3][8])
#        print("\n")
        
    def xl_los(self,current_car):
        c1=self.car[current_car][:2]
        x_los_indices=np.array([])
        other_cars=np.arange(self.no_cars)
        other_cars=np.delete(other_cars,np.where(other_cars==current_car))
        other_cars=np.delete(other_cars,np.where(other_cars==0))
        for i in other_cars:
            c2=self.car[i][:2]
            y=c2-c1
            #&0 degrees of aov
            theta=np.arctan2(y[1], y[0]) * 180 / np.pi
            if (theta>self.x_cone[0] and theta<self.x_cone[1]):
#                print("\n xl_Theta=",theta)
                x_los_indices=np.append(x_los_indices,i)
        x_los_indices=x_los_indices.astype(int)
#        print("\nx_los_indices",x_los_indices)
#        print("car values!",self.car)
        if len(x_los_indices)>0:
            cc=self.car[x_los_indices,1]
            cnd=np.min(cc)
            nd_max=cnd+20
#            print("nd_max",nd_max)
#            print("am in")
            for k in x_los_indices:
#                print("heretoo")
                if (self.car[k,1])>nd_max:
#                    print("double",self.car[k,1])
#                    print("k",k)
                    x_los_indices=np.delete(x_los_indices,np.where(x_los_indices==k))
#                    print("new indices",x_los_indices)
        
        return x_los_indices

    def xr_los(self,current_car):
        c1=self.car[current_car][:2]
        x_los_indices=np.array([])
        other_cars=np.arange(self.no_cars)
        other_cars=np.delete(other_cars,np.where(other_cars==current_car))
        other_cars=np.delete(other_cars,np.where(other_cars==0))
        for i in other_cars:
            c2=self.car[i][:2]
            y=c2-c1
            #&0 degrees of aov
            theta=np.arctan2(y[1], y[0]) * 180 / np.pi
            if (theta>-self.x_cone[1] and theta<-self.x_cone[0]):
#                print("\n xr_Theta=",theta)
                x_los_indices=np.append(x_los_indices,i)
        x_los_indices=x_los_indices.astype(int)
#        print("\nx_los_indices",x_los_indices)
#        print("car values!",self.car)
        if len(x_los_indices)>0:
            cc=self.car[x_los_indices,1]
            cnd=np.max(cc)
            nd_max=cnd
#            print("nd_max",nd_max)
#            print("am in")
            for k in x_los_indices:
#                print("heretoo")
                if (self.car[k,1])<nd_max-20:
#                    print("double",self.car[k,1])
#                    print("k",k)
                    x_los_indices=np.delete(x_los_indices,np.where(x_los_indices==k))
#                    print("new indices",x_los_indices)
                
        return x_los_indices


###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    
    def CNN_image(self):

        m=np.empty([100,130])
        m.fill(0)
        agent=(self.car[9,:2]/4).astype(int)
        agent_raw=self.car[9,:2]

        
    #Agent car
        agent_pos=np.array([30,50])
        ap=agent_pos
        cx,cy=ap[1],ap[0]
        r=-self.car[9,2]
        l=6
        vert=[]
        for i in range(-l,l+1,1):
            for j in range(-l,l+1,1):
                vert.append([cx+i,cy+j])
        r_xys = []
        for x, y in vert:
            tempX = x - cx
            tempY = y - cy
            rotatedX = tempX * np.cos(r) - tempY * np.sin(r)
            rotatedY = tempX * np.sin(r) + tempY * np.cos(r)
            x = rotatedX + cx
            y = rotatedY + cy
            r_xys += [[x, y]]
        for i in range(len(r_xys)):
            x,y=np.array(r_xys[i]).astype(int)
            m[x,y]=1
        
    #Surrounding cars
        for i in (self.car[1:9,:2]/4).astype(int):    
            d=i-agent
            if (d[0]<100 and d[0]>-30) and abs(d[1])<50:
                d=i-agent
                d=d+ap
                m[d[1]-l:d[1]+l,d[0]-l:d[0]+l]=0.5
    
    # Road
        y_agent=agent_raw[1]
        if y_agent<200:
            top_layer=200-y_agent
            t=int(top_layer/4)
            m[:t,:]=0.25
            
#        print(y_agent)
        if y_agent>200:
            bottom_layer=400-y_agent
            b=int(bottom_layer/4)
            m[50+b:,:]=0.5

#     Plot the image! 
#        plt.imshow(m)
#        plt.show()
        
        return m
    
    def agent_states(self):
        s1=self.agent_speed/self.agent_max_speed
        s2=self.car[self.agent_car,2]/(np.pi/4)
        s2=(s2+1)/2
#        print(np.array([s1,s2]))
        return np.array([s1,s2])
###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def collision(self):
        vx,vy=self.viewer_xy
        cx, cy, r ,w ,l = self.car[self.agent_car,:5]
        r_xys = []
        
        vert = [[cx + l / 2, cy + w / 2],
            [cx - l / 2, cy + w / 2],
            [cx - l / 2, cy - w / 2],
            [cx + l / 2, cy - w / 2]]
        
        for x, y in vert:
            tempX = x - cx
            tempY = y - cy
            # apply rotation
            rotatedX = tempX * np.cos(r) - tempY * np.sin(r)
            rotatedY = tempX * np.sin(r) + tempY * np.cos(r)
            # rotated x y
            x = rotatedX + cx
            y = rotatedY + cy
            r_xys += [x, y]
        x1,y1,x2,y2,x3,y3,x4,y4=r_xys
        
#    Collision with viewer
        
        if x1<0 or x4<0 or x2>vx or x3>vx or y3 <0 or y4<0 or y1>vy or y2>vy:   
            self.roadside_crash=True
     
# Distance of agent from other cars

        for i in range(self.no_cars-1):
            self.car[i,9]=(np.linalg.norm([cx,cy]-self.car[i,:2]))
    
# Theta of agent from other cars        
            #dist_arranged=np.argsort(self.car[:9, 9] )
        for i in range(self.no_cars-1):
            y=self.car[i,:2]-[cx,cy]
            theta=np.arctan2(y[1], y[0]) 
            self.car[i,2]=theta
#        print(self.car[:,:2])
#        print(self.car[:,2])
        
        distances=self.car[1:9,9]
        distances=(distances<self.collide_dist)
        if True in distances:
            self.collide=True
###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    

    def check_end_episode(self):
        if self.car[0,0]>2*self.viewer_xy[0]:
            self.end_episode=True

    def reward(self,action):
        
        if self.collide==True:
            return -1
        if self.roadside_crash==True:
            return -1
        
        return self.car[self.agent_car,5]*0.01-self.car[self.agent_car,2]*0.02#-self.car[self.agent_car,8]*0.01
        
    def reset_position(self):
        x_leader=np.random.uniform(1500,self.viewer_xy[0]-200,1)
        y_leader=self.viewer_xy[1]/2#np.random.uniform(50,self.viewer_xy[1],1)    

        near=True
        while(near==True):
#            print("Searching_____________\n ")
            k=0 # Zero near vehicles
            x=np.random.uniform(0,x_leader-100,self.no_cars)
            y=np.random.uniform(40,self.viewer_xy[1]-40,self.no_cars)
            x[0]=x_leader
            y[0]=y_leader
            xy=np.array([x,y]).T
            for i in range(self.no_cars):
                if i==0: continue
                for j in range(self.no_cars):
                    if i==j:continue
                    if np.linalg.norm(xy[i]-xy[j])<self.min_gap: k+=1 
            # K near vehicles
            if k ==0: near=False
#            print("Stuck")
            
        self.car[:,0]=x
        self.car[:,1]=y
        self.car[self.agent_car,2]=0
        
        self.car[:,5]=0#np.random.uniform(0,3,self.no_cars)
        self.car[:,8]=0#np.random.uniform(0,0,self.no_cars)
        
        #Agent car
        self.agent_speed=np.random.uniform(0,10) # Speed=0
        self.agent_steer=0
        self.car[self.no_cars-1,2]=0 # Steering =0
#        self.car[self.no_cars-1,5]=5
#        self.car[self.no_cars-1,8]=0

    def reset(self):
        self.end_episode=False
        self.collide=False
        self.roadside_crash=False
        self.reset_position()
        
        s=self.CNN_image()
        s=s.reshape(100,130)#,1)
        s2=self.agent_states()
        return s,s2
        
    def random_action(self):
#        a=(np.random.rand(2)-0.5)*2
        a=np.random.randint(0,5)
        return a 
           
    def render(self):
        self.game.screen.fill(self.game.white)
#        virtual_pos = np.copy(self.car_vert)
#        if float(self.car_vert[-1][0][0])>(self.viewer_xy[0]/2):
#            print("ok")
#            diff=self.car_vert[-1][0][0]-(self.viewer_xy[0]/2)
###            
#            for i in range(self.no_cars):
#                for j in range(4):
#                    virtual_pos[i][j][0]=self.car_vert[i][j][0]-diff
##        print(self.car_vert)            

        for i in range(len(self.car_vert)):
            pygame.draw.polygon(self.game.screen,self.game.color[i],self.car_vert[i])
            
            
        for i in np.array(self.state_cars):
            temp = (int(self.car[i,0]),int(self.car[i,1]))
            pygame.draw.circle(self.game.screen,self.game.yellow,temp, 15)
#        car_test=[[0,70],[-70,0],[-70,0],[70,0]]
#        pygame.draw.polygon(self.game.screen,self.game.color[0],car_test)
        self.game.clock.tick(40)
        pygame.display.update()
        
    def vertices(self,car,car_no):
#        print(car)
        cx,cy,r,w,l=car
        if car_no!=9: r=0
        vert = [[cx + l / 2, cy + w / 2],
            [cx - l / 2, cy + w / 2],
            [cx - l / 2, cy - w / 2],
            [cx + l / 2, cy - w / 2]]
        r_xys = []
        for x, y in vert:
            tempX = x - cx
            tempY = y - cy
            # apply rotation
            rotatedX = tempX * np.cos(r) - tempY * np.sin(r)
            rotatedY = tempX * np.sin(r) + tempY * np.cos(r)
            # rotated x y
            x = rotatedX + cx
            y = rotatedY + cy
            r_xys += [[x, y]]
#        print(r_xys)
        return r_xys

#demo simulation

if  __name__=='__main__':
#    np.random.seed(1)
    env=Env()
    
    for episode in range(15):#(env.episodes):
        s=env.reset()
        steps=0
        ret=0
        while True:
            steps+=1
            s1,s2,r,done=env.step(env.random_action())
            ret+=r
            env.render()
            if done or steps==100:
                print("ret",ret)
                break
        
        



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    