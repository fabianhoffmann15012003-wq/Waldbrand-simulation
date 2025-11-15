import numpy as np
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

# calculations 

def calc_c_0(S):
    return ALPHA*S + (1-ALPHA)*LAMBDA*GAMMA + ALPHA*GAMMA*(1-S)

def calc_c_1(S, c_0):
    return c_0-ALPHA*S

def calc_S_2(s2_alt, T, dt):
    _,_, avg_u = calc_avg_u()
    r_2 = C_S2*np.exp(-B_2/T)
    r_m = R_M_0+R_M_C*(avg_u-1)
    r_2t = (r_2*r_m)/(r_2+r_m)
    return s2_alt-s2_alt*r_2t*dt

def calc_x_c(S_2_matrix, S_2_0_matrix):
    return S_2_matrix/S_2_0_matrix

def calc_avg_u_v(): # WAS ZUR HÖLLE IST u_H?????
    print("\n--------------------- TO DO calc_avg_u_v() ---------------------\n")
    return np.nan

def calc_avg_u_b(): # WAS ZUR HÖLLE IST u_b*?????
    print("\n--------------------- TO DO calc_avg_u_b() ---------------------\n")
    return np.nan

def calc_avg_u(avg_u_v, avg_u_b, x_c):
    print("\n--------------------- TO DO calc_avg_u() ---------------------\n")
    avg_u_x = np.nan
    avg_u_y = np.nan
    avg_u = avg_u_v+(avg_u_b-avg_u_v)*(a-x_c)
    return avg_u_x, avg_u_y, avg_u

def calc_L():
    print("\n--------------------- TO DO calc_L() ---------------------\n")
    L_x = np.nan
    L_y = np.nan
    return L_x, L_y

def calc_omega():
    print("\n--------------------- TO DO calc_omega() ---------------------\n")
    omega_x = np.nan
    omega_y = np.nan
    return omega_x, omega_y

def calc_D_eff():
    avg_u_x, avg_u_y, _ = calc_avg_u()
    L_x, L_y = calc_L()
    omega_x, omega_y = calc_omega()
    D_eff_x = D_RB+A_D*avg_u_x*L_x*(1-np.exp(-GAMMA_D*omega_x))
    D_eff_y = D_RB+A_D*avg_u_y*L_y*(1-np.exp(-GAMMA_D*omega_y))
    return D_eff_x, D_eff_y


# calculations for reaction and convection
def calc_S_1(S1_alt_matrix, T_matrix, dt):
    r_1 = C_S1*np.exp(-B_1/T_matrix)
    return S1_alt_matrix-S1_alt_matrix*r_1*dt


def calc_S(S_1, S_2):
    return S_1+S_2

# calculation of the Temperature => calculating a step

def calc_T(T_matrix, S_1, D_eff, avg_u_all):
    T_matrix_new = np.copy(T_matrix)
    for i in range(np.shape(T_matrix[0])):
        for j in range(np.shape(T_matrix[1])):
            T_xy = T_matrix[i, j]


def main():
    N = 100 # number of steps
    dt = 0.1 # length of time steps from paper in seconds
    S_1_matrix = np.zeros((250,250)) # shape of the forest
    S_1_0_matrix = S_1_matrix
    T_matrix = np.zeros((250,250)) # shape of the forest 
    for t in range(N):
        S_1_matrix_new = calc_S_1(S_1_matrix, T_matrix, dt)
        D_eff, D_eff = calc_D_eff()
        
        avg_u_all = calc_avg_u()
        T_new = calc_T(T_matrix, S_1_matrix, D_eff, avg_u_all)