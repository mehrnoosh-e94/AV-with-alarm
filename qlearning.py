import numpy as np
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import statistics
import math
import pickle
from tqdm import tqdm
import collections
import variables as var
import matplotlib.patches as mpatches
import copy
from matplotlib.lines import Line2D
import cmath

import particles as ps
import state as st
import plot as myplt
import state as st
import in_out as io
import variables as var

T_Y = var.T_Y
T_G = var.T_G
T_R = var.T_R
total_T = T_Y + T_G + T_R

def reward_function(distance, velocity, phi, t_phi, v_max, T_Y):
    reward = -2
    
    if distance<=0:
        if (phi=='G') or (phi=='Y'):
            if velocity>0:
                reward = reward + 100
        elif phi=='R':

            if velocity>0:
                reward = reward - 200
    
    return reward



def reward_function_smooth(distance, velocity, phi, t_phi, prev_a, a, v_max, T_Y):
    reward = -2
    
    delta_a = abs(a-prev_a)
    if delta_a>1:
        reward = reward - 2*(delta_a)
        
    if distance<=0:
        if (phi=='G') or (phi=='Y'):
            if velocity>0:
                reward = reward + 100
        elif phi=='R':

            if velocity>0:
                reward = reward - 200
    
    return reward



def TD_function(reward, discount_factor, Q_row, q):
    return reward + discount_factor*np.max(Q_row.flatten()) - q



def action_selection(Q_row, velocity, actions, epsilon, v_max, training):

    if (velocity>v_max) and training:
        feasible_index = [index for index,value in enumerate(actions) if value <0]
    
    if not(velocity>v_max) and training:
        feasible_index = [index for index,value in enumerate(actions) if (velocity+value >=0)]
        
    if not(training):
        feasible_index = [i for i in range(len(actions))]

    if (velocity<0) and (len(feasible_index)==0):
        feasible_index = [np.argmax(np.asarray(actions))]
            
    feasible_actions = actions[feasible_index]

    rand = np.random.uniform(0,1,1)
    if rand>=epsilon:
        candidates = [i for i, x in enumerate(Q_row.flatten()[feasible_index]) if x == np.max(Q_row.flatten()[feasible_index])]
        
        if training:
            candidates = [feasible_index[c] for c in candidates]
            action_idx = random.choice(candidates)
            action = actions[action_idx]
            
        elif not(training):
            action_idx = random.choice(candidates) 
            
            action = feasible_actions[action_idx]
        
    elif rand<epsilon:
        action_idx =random.choice(feasible_index)
        action = actions[action_idx]
        
    return action, action_idx


def PF_action_selection(Q_table, N, d_particle, v_particle, phi_particle, t_phi_particle, weights, actions, epsilon,discrete_d, discrete_v, discrete_phi_t, T_Y, T_R,v_max, phi_prev_phi_dict):
    particles_Q_row = np.zeros((N, len(actions)))
    for i in range(N):
        particle_state = (d_particle[i,0], v_particle[i,0], phi_particle[i,0], t_phi_particle[i,0])
        idx = st.map_state_to_idx(particle_state, discrete_d, discrete_v, discrete_phi_t, T_Y, T_R, phi_prev_phi_dict)
        particles_Q_row[i] = Q_table[idx]
        
    weighted_Q_row = np.average(particles_Q_row, axis=0, weights=weights)

    velocity = np.average(v_particle, axis=0, weights=weights)
        
    feasible_index = [i for i in range(len(actions))]
    feasible_actions = actions[feasible_index]
    rand = np.random.uniform(0,1,1)
    
    if rand>=epsilon:
        candidates = [i for i, x in enumerate(weighted_Q_row.flatten()[feasible_index]) if x == np.max(weighted_Q_row.flatten()[feasible_index])]


        candidates = [feasible_actions[c] for c in candidates]
        action_idx = np.random.randint(0,len(candidates))
        action = candidates[action_idx]

        
    else:
        action_idx = np.random.randint(0,len(feasible_index))
        
        action = actions[action_idx]

    return action


