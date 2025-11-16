import numpy as np
from scipy.signal import convolve2d
#Code, der die DGL löst oder so

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

# calculations 

def calc_c_0(S):
    return ALPHA*S + (1-ALPHA)*LAMBDA*GAMMA + ALPHA*GAMMA*(1-S)

def calc_c_1(S, c_0):
    return c_0-ALPHA*S

def calc_S_1(S1_alt_matrix, T_matrix, dt):
    r_1 = C_S1*np.exp(-B_1/T_matrix)
    return S1_alt_matrix-S1_alt_matrix*r_1*dt

def calc_S_2(S2_alt_matrix, T_matrix, avg_u_x, dt):
    r_2 = C_S2*np.exp(-B_2/T_matrix)
    r_m = R_M_0+R_M_C*(avg_u_x-1)
    r_2t = (r_2*r_m)/(r_2+r_m)
    return S2_alt_matrix-S2_alt_matrix*r_2t*dt

def calc_S(S_1, S_2):
    return S_1+S_2

def calc_x_c(S_2_matrix, S_2_0_matrix):
    return S_2_matrix/S_2_0_matrix



def calc_avg_u_x(x_c):
    avg_u_x = AVG_U_V_X+(AVG_U_B_X-AVG_U_V_X)*(1-x_c)
    return avg_u_x

def calc_L(T_matrix):
    T_max = np.max(T_matrix)
    T_dropped = 0.1*T_max +T_A
    T_max_x, T_max_y = np.unravel_index(np.argmax(T_matrix), T_matrix.shape)
    L_x = (np.sum(T_matrix[T_max_x,:]>T_dropped)-1)*dx
    L_y = (np.sum(T_matrix[:,T_max_y]>T_dropped)-1)*dx
    return L_x, L_y

def calc_omega():
    print("\n--------------------- TO DO calc_omega() ---------------------\n")
    omega_x = np.nan
    omega_y = np.nan
    return omega_x, omega_y

def calc_D_eff():
    avg_u_x, avg_u_y = calc_avg_u_x(), 0
    L_x, L_y = calc_L()
    omega_x, omega_y = calc_omega()
    D_eff_x = D_RB+A_D*avg_u_x*L_x*(1-np.exp(-GAMMA_D*omega_x))
    D_eff_y = D_RB+A_D*avg_u_y*L_y*(1-np.exp(-GAMMA_D*omega_y))
    return D_eff_x, D_eff_y




# calculation of the Temperature => calculating a step

def calc_T(T_matrix, c_0, c_1, D_eff_x, D_eff_y, avg_u_x, dt, avg_u_y=0):
    T_matrix_new = np.copy(T_matrix)
    dy2_kern = [[1,-2,1]]
    dy_kern = [[-1/2,0,1/2]]
    dx2_kern = [[1],[-2],[1]]
    dx_kern = [[-1/2],[0],[1/2]]
    T_matrix_dx2 = convolve2d(T_matrix, dx2_kern, mode='same', boundary='symm')
    T_matrix_dy2 = convolve2d(T_matrix, dy2_kern, mode='same', boundary='symm')
    T_matrix_dx = convolve2d(T_matrix, dx_kern, mode='same', boundary='symm')
    T_matrix_dy = convolve2d(T_matrix, dy_kern, mode='same', boundary='symm')
    dT_dt_matrix = c_1/c_0*(D_eff_x*T_matrix_dx2 + D_eff_y*T_matrix_dy2-avg_u_x*T_matrix_dx-avg_u_y*T_matrix_dy)
    T_matrix_new = T_matrix_new + dT_dt_matrix*dt
    return T_matrix
            



# --- Initial conditions
# Sparse Canopy
Z_0 = 0.5
DELTA = 0.08
# Dense Canopy
#Z_0 = 0.25
#DELTA = 0.04

NX, NY = 100, 100
S_1_matrix = np.zeros((NX, NY)) # shape of the forest
S_2_matrix = np.zeros((NX, NY)) # shape of the forest
S_1_0_matrix = S_1_matrix
T_matrix = np.zeros((NX,NY)) # shape of the forest 

U_10_X = 5 # wind with speed 10 m/s in only the x-direction
U_H_X = U_10_X*0.9
U_B_STAR_X = U_10_X*KAPPA/np.ln(10/Z_0)
AVG_U_V_X = U_H_X/ETA * (1-np.exp(-ETA))
AVG_U_B_X = U_B_STAR_X/KAPPA * (H/(H-Z_0)*np.ln(H/Z_0)-1)

# --- simulation conditions
N = 100 # number of steps
dt = 0.1 # length of time steps from paper in seconds
dx = 0.5 # length of a "pixel" from paper in m

for t in range(N):
    D_eff_x, D_eff_y = calc_D_eff()
    avg_u_x = calc_avg_u_x()
    S_1_matrix_new = calc_S_1(S_1_matrix, T_matrix, dt)
    S_2_matrix_new = calc_S_2(S_2_matrix, T_matrix, avg_u_x, dt)
    S_matrix_new = calc_S(S_1_matrix_new, S_2_matrix_new)
    c_0 = calc_c_0(S_matrix_new)
    c_1 = calc_c_1(S_matrix_new, c_0)
    T_new = calc_T(T_matrix, c_0, c_1, D_eff_x, D_eff_y, avg_u_x, dt)