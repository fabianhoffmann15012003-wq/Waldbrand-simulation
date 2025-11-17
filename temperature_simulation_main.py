import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib import animation

# Constants ? from the Table 2 
FMC = 0.25
H = 2
T_A = 300
T_MAX_I = 1200
RHO_G = 1
RHO_S = 700
C_PG = 1043
C_PS = 1800
C_S1 = 30
C_S2 = 40
B_1 = 4500
B_2 = 7000
A_1 = 22 * 10**5
A_2 = 2 * 10**7
D_RB = 0.1
R_M_0 = 0.002
R_M_C = 0.004
GAMMA_D = 0.03
SIGMA = 20
A_NC = 0.2
A_D = 0.125
ETA = 3
ALPHA = 0.002
EPSILON = 0.2
GAMMA = C_PG/C_PS
LAMBDA = RHO_G/RHO_S
C_2 = ALPHA*A_1/C_PS
C_3 = ALPHA*A_2/C_PS
C_4 = 1/(H*RHO_S*C_PS)
KAPPA = 0.41 # Karman's Konstant

def gauss2d(x, y, mx, my, s):
    return 1./(2.*np.pi*s*s)*np.exp(-((x-mx)**2./(2.*s**2.)+(y-my)**2./(2.*s**2.)))

class Sim:

    def __init__(self, NX=100, NY=100, U_10_X=10, U_10_Y=0):
        # --- Initial conditions
        # Sparse Canopy
        Z_0 = 0.5
        DELTA = 0.08
        # Dense Canopy
        #Z_0 = 0.25
        #DELTA = 0.04

        self.NX, self.NY = 100, 100
        self.S_1_matrix = np.ones((self.NX, self.NY))*0.8 # shape of the forest
        self.S_2_matrix = np.ones((self.NX, self.NY))*0.8 # shape of the forest
        self.S_2_0_matrix = self.S_2_matrix
        self.S_matrix = self.calc_S()
        self.T_matrix = np.ones((self.NX,self.NY))*T_A # shape of the forest 
        for i in range(self.NX):
            for j in range(self.NY):
                self.T_matrix[i,j] += gauss2d(i, j, self.NX//2, self.NY//2, self.NX//14)*700

        self.U_10_X = U_10_X # wind with speed 10 m/s in only the x-direction
        self.U_10_Y = U_10_Y # wind with speed 10 m/s in only the x-direction
        self.U_H_X = self.U_10_X*0.9
        self.U_B_STAR_X = self.U_10_X*KAPPA/np.log(10/Z_0)
        self.AVG_U_V_X = self.U_H_X/ETA * (1-np.exp(-ETA))
        self.AVG_U_B_X = self.U_B_STAR_X/KAPPA * (H/(H-Z_0)*np.log(H/Z_0)-1)

        # --- simulation conditions
        self.N = 1000 # number of steps
        self.dt = 0.1 # length of time steps from paper in seconds
        self.dx = 0.5 # length of a "pixel" from paper in m

        self.x_c = self.calc_x_c()
        self.avg_u_x = self.calc_avg_u_x()
        self.avg_u_y = 0
        self.c_0 = self.calc_c_0()
        self.c_1 = self.calc_c_1()
        self.L_x, self.L_y = self.calc_L()
        self.omega_x, self.omega_y = self.calc_omega()
        self.D_eff_x, self.D_eff_y = self.calc_D_eff()

        
    # calculations 
    def calc_c_0(self):
        return ALPHA*self.S_matrix + (1-ALPHA)*LAMBDA*GAMMA + ALPHA*GAMMA*(1-self.S_matrix)
    
    def calc_c_1(self):
        return self.c_0-ALPHA*self.S_matrix
    
    def calc_S_1(self):
        r_1 = C_S1*np.exp(-B_1/self.T_matrix)
        return self.S_1_matrix-self.S_1_matrix*r_1*self.dt
    
    def calc_S_2(self):
        r_2 = C_S2*np.exp(-B_2/self.T_matrix)
        r_m = R_M_0+R_M_C*(self.avg_u_x-1)
        r_2t = (r_2*r_m)/(r_2+r_m)
        return self.S_2_matrix-self.S_2_matrix*r_2t*self.dt
    
    def calc_S(self):
        return self.S_1_matrix+self.S_2_matrix

    def calc_x_c(self):
        return self.S_2_matrix/self.S_2_0_matrix
    
    def calc_avg_u_x(self):
        return self.AVG_U_V_X+(self.AVG_U_B_X-self.AVG_U_V_X)*(1-self.x_c)

    def calc_L(self):
        T_max = np.max(self.T_matrix)
        T_dropped = 0.1*T_max +T_A
        T_max_x, T_max_y = np.unravel_index(np.argmax(self.T_matrix), self.T_matrix.shape)
        L_x = (np.sum(self.T_matrix[T_max_x,:]>T_dropped)-1)*self.dx
        L_y = (np.sum(self.T_matrix[:,T_max_y]>T_dropped)-1)*self.dx
        return L_x, L_y

    def calc_omega(self):
        T_max_x = [np.argmax(self.T_matrix[i][0:]) for i in range(self.NX)]    #finding maxima in x direction
        yA = yB = xA = xB = -1      #initialize values
        for i in range(self.NX):     #finding endpoints xA, xB, yA, yB by condition
            if yA < 0 and self.T_matrix[i][T_max_x[i]] > 550: #only once at beginning for y
                yA = i
                xA = T_max_x[i]
            if self.T_matrix[i][T_max_x[i]] > 550:   #reset values until done
                yB = i
                if T_max_x[i] >= xB:    #check if new xB is actually larger than old one
                    xB = T_max_x[i]
                if T_max_x[i] < xA:
                    xA = T_max_x[i]

        omega_x = abs(xB-xA)*self.dx     #computing the difference between endpoints
        omega_y = abs(yB-yA)*self.dx
        return omega_x, omega_y

    def calc_D_eff(self):
        D_eff_x = D_RB+A_D*self.avg_u_x*self.L_x*(1-np.exp(-GAMMA_D*self.omega_x))
        D_eff_y = D_RB+A_D*self.avg_u_y*self.L_y*(1-np.exp(-GAMMA_D*self.omega_y))
        return D_eff_x, D_eff_y




    # calculation of the Temperature => calculating a step

    def calc_T(self):
        T_matrix_new = np.copy(self.T_matrix)
        dy2_kern = [[1,-2,1]]
        dy_kern = [[-1/2,0,1/2]]
        dx2_kern = [[1],[-2],[1]]
        dx_kern = [[-1/2],[0],[1/2]]
        T_matrix_dx2 = convolve2d(self.T_matrix, dx2_kern, mode='same', boundary='symm')
        T_matrix_dy2 = convolve2d(self.T_matrix, dy2_kern, mode='same', boundary='symm')
        T_matrix_dx = convolve2d(self.T_matrix, dx_kern, mode='same', boundary='symm')
        T_matrix_dy = convolve2d(self.T_matrix, dy_kern, mode='same', boundary='symm')
        dT_dt_matrix = self.c_1/self.c_0*(self.D_eff_x*T_matrix_dx2 + self.D_eff_y*T_matrix_dy2-self.avg_u_x*T_matrix_dx-self.avg_u_y*T_matrix_dy)
        T_matrix_new = T_matrix_new + dT_dt_matrix*self.dt
        return T_matrix_new
            




    def step(self, step):
        x_c = self.calc_x_c()
        avg_u_x = self.calc_avg_u_x()
        S_1_matrix_new = self.calc_S_1()
        S_2_matrix_new = self.calc_S_2()
        S_matrix_new = self.calc_S()
        c_0 = self.calc_c_0()
        c_1 = self.calc_c_1()
        L_x, L_y = self.calc_L()
        omega_x, omega_y = self.calc_omega()
        omega_x, omega_y = self.calc_D_eff()
        self.T_matrix = self.calc_T()
        self.x_c = x_c
        self.avg_u_x = avg_u_x
        self.S_1_matrix = S_1_matrix_new
        self.S_2_matrix = S_2_matrix_new
        self.S_matrix = S_matrix_new
        self.c_0 = c_0
        self.c_1 = c_1
        self.L_x, self.L_y = L_x, L_y
        self.omega_x, self.omega_y = omega_x, omega_y
        self.D_eff_x, self.D_eff_y = omega_x, omega_y



def update(frame):
    im.set_data(simualtion.T_matrix)
    ax.axis('off')
    simualtion.step(frame)
    return [im]


simualtion = Sim()

fig, ax = plt.subplots(figsize=(16,16))
im = ax.imshow(simualtion.T_matrix, vmin=T_A)
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)

ani = animation.FuncAnimation(fig, update, frames=500, interval=1) #frames - the number of steps in the simulation
ani.save('Animations/simple.gif', fps=10, savefig_kwargs={'pad_inches':0})

print("\n------------------------------ ! ! ! FERTIG ! ! ! ------------------------------\n")

