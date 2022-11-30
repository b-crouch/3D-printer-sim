import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rc
plt.rcParams['animation.ffmpeg_path'] = r'ffmpeg'

#----Part 1: specify constants for robotic arm and filament material properties----#
#Assigning robotic arm and print bed constants
r0 = np.array([0, 0.5, 0]) #Fixed arm end position
theta1_init, theta2_init, theta3_init = np.pi/2, 0, 0 #Initial rod angles
w1, w2, w3 = 0.2, -0.2, 10 #Angular velocities of linkages
L1, L2, L3 = 0.3, 0.2, 0.08 #Rod lengths
L_bed = 0.8 #Side length of print bed
qp = -8*10**(-5) #Grid pixel charge

#Assigning material properties and calculating mixture characteristics
epsilon = 8.854*10**(-12) #Electric permittivity
R = 0.001 #Droplet radius
V = (4/3)*np.pi*R**3 #Droplet volume
v2 = 0.25 #Volume fraction of phase 2
rho1, rho2 = 2000, 7000 #Densities of phases
q1, q2 = 0, 10**(-3) #Charge densities of phases
rho_star = (1-v2)*rho1 + v2*rho2 #Effective denisty of mixture
q_star = (1-v2)*q1 + v2*q2 #Effective charge density of mixture
m = V*rho_star #Mass of droplets
q = V*q_star #Charge of droplets
v_d = np.array([0, -1.2, 0]) #Relative extrusion velocity

#Assigning characteristics to surrounding medium
v_f = np.array([0.5, 0, 0.5]) #Surrounding medium velocity
rho_a = 1.225 #Surrounding medium density
mu_f = 1.8*10**(-5) #Surrounding medium viscosity
A = np.pi*R**2 #Drag reference area

#Assigning general simulation constants
dt = 0.001 #Time step size
T = 3 #Total simulation time
g = 9.81 #Gravitational acceleration
F_grav = np.array([0, -m*g, 0]) #Force of gravity
w1_bounds = [15, 16]
w2_bounds = [15, 16]
w3_bounds = [6, 7]
v_d_bounds = [-3.5, -3]

#----Part 2: define physics engine for simulation----#
def printer_model(T, w1, w2, w3, dv, electro=True):
    """
    Simulates trajectories of robotic arm and filament droplets in electrophoretic 3D printer
    :param float T: Time for simulation
    :param float w1: Angular velocity of first linkage in robotic arm
    :param float w2: Angular velocity of second linkage in robotic arm
    :param float w3: Angular velocity of third linkage in robotic arm
    :param float dv: Relative velocity of filament head
    :param bool electro: if True, model the 3D printer as having an electrophoretic print bed
    :return: Dictionary containing the coordinate positions of all robotic arm linkages and extruded filament droplets at each timestep in the simulation
    """
    n = int(T/dt)
    counter = 0 
    t = 0 
    pos, arm1, arm2 = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    thetas = np.zeros((n, 3))
    thetas[0, :] = np.array([theta1_init, theta2_init, theta3_init])
    vel, vel1, vel2 = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    w = np.array([w1, w2, w3])
    #Initialize charged particle bed
    charges = np.zeros((100, 3))
    charge_coords = np.meshgrid(np.linspace(-L_bed/2, L_bed/2, 10), np.linspace(-L_bed/2, L_bed/2, 10))
    charges[:, 0] = charge_coords[0].flatten()
    charges[:, 2] = charge_coords[1].flatten()
    #Initialize droplet positions and velocities arrays
    drops_hist = []
    drops = np.empty((n, 3))
    drops_vel = np.zeros((n, 3))
    flight_time = np.zeros(n)
    #Initialize force vectors
    gravity = np.tile(F_grav, (n, 1))
    elec = np.zeros((n, 3))
    drag = np.zeros((n, 3))
    active = np.zeros(n, dtype=bool)
    active[0] = True
    drops_hist.append(drops.copy())
    #Begin simulation
    while any(active):   
        if t != n:
            #Update robotic arm position
            thetas[t, :] = thetas[0, :] + w*t*dt
            theta1, theta2, theta3 = thetas[t, :]
            vel1[t, :] = np.array([-L1*w1*np.sin(theta1), L1*w1*np.cos(theta1), 0])
            vel2[t, :] = np.array([-L2*w2*np.sin(theta2), L2*w2*np.cos(theta2), 0])
            vel[t, :] = np.array([L3*w3*np.cos(theta3), 0, -L3*w3*np.sin(theta3)])
            arm1[t, :] = r0 + np.array([L1*np.cos(theta1), L1*np.sin(theta1), 0])
            arm2[t, :] = arm1[t, :] + np.array([L2*np.cos(theta2), L2*np.sin(theta2), 0])
            pos[t, :] = arm2[t, :] + np.array([L3*np.sin(theta3), 0, L3*np.cos(theta3)])
            #Release new droplet
            drops[t, :] = pos[t, :] 
            drops_vel[t, :] = vel1[t, :] + vel2[t, :] + vel[t, :] + np.array([0, dv, 0])
            active[t] = True
            flight_time[t] = dt*counter
            t += 1          
        else:
            #Once all droplets have been released, do not update robotic arm movement
            arm1 = np.vstack([arm1, arm1[-1, :]])
            arm2 = np.vstack([arm2, arm2[-1, :]])
            pos = np.vstack([pos, pos[-1, :]])
        #Update existing droplets
        for drop in range(t):
            if active[drop]:
                if electro:
                    diff = drops[drop, :] - charges
                    coef = qp*q/(4*np.pi*epsilon*(np.linalg.norm(diff, axis=1)[:, np.newaxis])**3)
                    elec[drop, :] = np.sum(coef*diff, axis=0)
                reynold = (2*R*rho_a/mu_f)*np.linalg.norm(v_f - drops_vel[drop, :])
                if 0 < reynold <= 1:
                    c_d = 24/reynold
                elif 1 < reynold <= 400:
                    c_d = 24/(reynold**0.646)
                elif 400 < reynold <= 3*10**5:
                    c_d = 0.5
                elif 3*10**5 < reynold <=2*10**6:
                    c_d = 0.000366*reynold**0.4275
                else:
                    c_d = 0.18
                drag[drop, :] = 0.5*rho_a*c_d*A*np.linalg.norm(v_f - drops_vel[drop, :])*(v_f - drops_vel[drop, :])
        force = gravity + elec + drag
        active[drops[:, 1] < 0] = False
        flight_time[drops[:, 1] < 0] = counter*dt - flight_time[drops[:, 1] < 0]
        drops[drops[:, 1] < 0, 1] = 0
        drops[active, :] = drops[active, :] + dt*drops_vel[active, :]
        drops_vel[active, :] = drops_vel[active, :] + (dt/m)*force[active, :]
        if not counter%20:
            drops_hist.append(drops.copy())
        counter += 1
    return {"arm1":arm1, "arm2":arm2, "dispens":pos, "drops":drops, "steps":counter, \
            "tof":flight_time, "charges":charges, "drops_hist":drops_hist}

