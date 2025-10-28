import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
from matplotlib.patches import Rectangle
from scipy.signal import convolve2d

from datetime import timedelta,datetime
#import time

class WFSim:
    def __init__(self, 
                 fire=0.01, tree=1e-4, grass=1e-3, 
                 stone=0.005, water=0.04, 
                 h=16, w=16):
        self.fire = fire #chance a random tree catches fire
        self.tree = tree #chance a random grassland grow into a tree
        self.grass = grass #chance a random Ash area grown grass
        self.stone = stone #chance a random spot is Stone
        self.water = water #chance a random spot is Water
        self.h = h #grid height
        self.w = w #grid width

        self.states = ["Ash",         #0
                       "Burning",     #1
                       "Grass",       #2
                       "Tree",        #3
                       "Stone",       #4
                       "Water"        #5
        ]

        #self.stepspeed = list()

        self.landscape = np.random.randint(2,4,(self.h,self.w)) #generating grid
        # the landscape is binary 2/3 (grass/tree) both are equally likely
        for i in range(self.landscape.shape[0]):
            for j in range(self.landscape.shape[1]):
                coef = 7 if self.nonburn_neighbors_check(i, j, "S") else 1
                if self.stone*coef>np.random.random():
                    self.landscape[i, j] = 4     

                coef = 10 if self.nonburn_neighbors_check(i, j, "W") else 0.1
                if self.water*coef>np.random.random():
                    self.landscape[i, j] = 5 

    
    def nonburn_neighbors_check(self, idx, jdx, mat):
        check = False
        offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        #offsets = coordinates of all the neighbors of a cell (0,0)
        if mat == "S":
            m=4
        elif mat == "W":
            m=5
        else:
            raise Exception(KeyError, mat+" is not defined")
        
        for di,dj in offsets: #checking if any of the neighbors have fire
            ni, nj = idx+di, jdx+dj
            if nj>=0 and ni>=0 and ni<self.h and nj<self.w:
                if self.landscape[ni, nj] == m:
                    check +=True
        return check
    

    # probagates the state into the new state
    def step(self, step):
        #start = time.time()
        new_landscape = self.landscape.copy()

    #all spots that are burning convert to ash
        new_landscape[self.landscape == 1] = 0

    #spots that are grass convert to tress with a certain chance
        rand = np.random.rand(len(self.landscape),len(self.landscape[0]))
        #checks if the spot is within the percentage chance and a grass spot, those are set to 1
        change = (self.tree>rand) * (self.landscape==2) 
        new_landscape += change #2(grass) +1 = 3(tree)

    #spots that are trees convert to fire with a certain chance and if the neighbours are on fire
        rand = np.random.rand(len(self.landscape),len(self.landscape[0]))
        conv_fire = [[1,1,1],[1,0,1],[1,1,1]]
        prop = convolve2d((self.landscape==1), conv_fire, mode='same', boundary='fill', fillvalue=0) #calculates how many neighbours are on fire
        #checks if the spot is within the percentage chance or at least one neighbour is on fire, and is a tree spot, those are set to 1
        change = np.logical_or(self.fire>rand, prop!=0)*(self.landscape==3) 
        new_landscape -= 2*change #3(tree) -2 = 1(fire)

    #spots that are ash convert to grass with a certain chance that is higher if more direct neighbours are grass
        rand = np.random.rand(len(self.landscape),len(self.landscape[0]))
        conv_grass = [[0,1,0],[1,0,1],[0,1,0]]
        coef_grass = 1+convolve2d((self.landscape==2), conv_grass, mode='same', boundary='fill', fillvalue=0)*19/4 #calculates a coef depending on how many direct neighbours are grass
        #checks if the spot is within the adjusted percentage chance and a ash spot, those are set to one
        change = ((self.grass*coef_grass)>rand)*(self.landscape==0)
        new_landscape += 2*change #0(ash) +2 = 2(grass)

        self.landscape = new_landscape.copy()
        
        #end = time.time()
        #self.stepspeed.append(end-start)



def update(frame):
    im.set_data(Sim.landscape)
    ax.axis('off')
    Sim.step(frame)
    return [im]

print("\n--------------------------------------------- ! ! ! ANFANG ! ! ! ---------------------------------------------\n")

Sim = WFSim(h=64, w=64) #initializing the Simulation

#               ash   ,  fire   ,  grass     ,  tree        , stone ,  water
colors_list = ['black', 'orange', 'olivedrab', 'forestgreen', 'grey', 'steelblue'] #setting cell colors
cmap = colors.ListedColormap(colors_list)
bounds = range(len(Sim.states)+1) 
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(16,16))
im = ax.imshow(Sim.landscape, cmap=cmap, norm=norm)
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)


ani = animation.FuncAnimation(fig, update, frames=80, interval=20) #frames - the number of steps in the simulation

ani.save('Animations/simple.gif', fps=1.5, savefig_kwargs={'pad_inches':0})

#print("\n\taverage Stepspeed: "+str(np.sum(Sim.stepspeed)/len(Sim.stepspeed))+"s")
print("\n--------------------------------------------- ! ! ! FERTIG ! ! ! ---------------------------------------------\n")
#plt.show()

# time with array           t_a: 0.0003820643012906298 s
# time with loops           t_l: 0.016065229604273666  s
# time difference t_l-t_a = d_t: 0.015683165302983037  s
# time factor     t_l/t_a = f_t: 42.04849694149551     
