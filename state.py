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
import qlearning as ql
import plot as myplt
import in_out as io
import variables as var

T_Y = var.T_Y
T_G = var.T_G
T_R = var.T_R
total_T = T_Y + T_G + T_R


def map_idx_to_state(idx, discrete_d, discrete_v, discrete_phi_t):
    len_1 = len(discrete_d)
    len_2 = len(discrete_v)
    len_3 = len(discrete_phi_t)
    
    idx_1 = int(np.floor(idx/(len_2*len_3)))
    distance = discrete_d[idx_1]
    
    idx_2 = int(np.floor((idx - idx_1*len_2*len_3)/len_3))
    velocity = discrete_v[idx_2]
    
    idx_3 = idx - idx_1*len_2*len_3 - idx_2*len_3

    phi = discrete_phi_t[idx_3].flatten()[0]
    t_phi = discrete_phi_t[idx_3].flatten()[1]
    
    return (distance, velocity, phi, int(t_phi))


def map_state_to_idx(state, discrete_d, discrete_v, discrete_phi_t, T_Y, T_R, phi_prev_phi_dict):
    distance = state[0]
    velocity = state[1]
    if state[3]<=0.5:
        phi=phi_prev_phi_dict[state[2]]
        t_phi=10
    elif state[3]>0.5:
        phi = state[2]
        t_phi = state[3]
    distance_diff = np.asarray([(distance - d)**2 for d in discrete_d])
    distance_idx = np.argmin(distance_diff)

    velocity_diff = np.asarray([(velocity - v)**2 for v in discrete_v])
    velocity_idx = np.argmin(velocity_diff)
        
    t_phi_candidates = np.asarray([int(x[1]) for x in discrete_phi_t if x[0]==phi])
    t_phi_diff = np.asarray([(t_phi - t)**2 for t in t_phi_candidates])
    t_phi_idx = np.argmin(t_phi_diff)
    if phi=='Y':
        phi_t_idx = t_phi_idx
    elif phi=='R':
        phi_t_idx = t_phi_idx + (T_Y)
    elif phi=='G':
        phi_t_idx = t_phi_idx + (T_Y+T_R)

    idx = distance_idx*len(discrete_v)*len(discrete_phi_t) + velocity_idx*len(discrete_phi_t) + phi_t_idx
    
    return idx
    
    
    
def update_state(state, action, delta_t, phi_to_T_dict, phi_to_next_phi_dict, std_d, std_v, std_t_phi, training):
    distance = state[0]
    velocity = state[1]
    phi = state[2]
    t_phi = state[3]
   
    distance = distance - velocity*delta_t - 0.5*action*(delta_t**2) + np.random.normal(0, std_d, 1)
    velocity = velocity + action*delta_t + np.random.normal(0, std_v, 1)

    rnd_tphi = np.random.normal(0, std_t_phi, 1)
    if (t_phi + delta_t + rnd_tphi) <= phi_to_T_dict[phi]:
        t_phi = (t_phi) + delta_t + rnd_tphi
        phi = phi

    else:
        t_phi = max(t_phi + delta_t + rnd_tphi - phi_to_T_dict[phi], 0)
        phi = phi_to_next_phi_dict[phi]
        
    
    if (velocity[0]<0) and not(training):
        
        distance[0]-=0.5*np.abs(velocity[0])
        velocity[0]=0

    return (distance[0], velocity[0], phi, t_phi[0])


def random_init_state(d_lower, d_upper, v_lower, v_upper, T_Y, T_G, T_R):
    actual_d_0 = np.random.uniform(d_lower, d_upper, 1)
    actual_v_0 = np.random.uniform(v_lower, v_upper, 1)
    actual_phi_probs = np.random.uniform(0, 1, 1)

    actual_phi_0 = np.full((actual_phi_probs.shape), '-')
    actual_phi_0[actual_phi_probs<(T_Y/total_T)] = 'Y'
    actual_phi_0[((actual_phi_probs>=(T_Y/total_T)) & (actual_phi_probs<((T_Y+T_G)/total_T)))] = 'G'
    actual_phi_0[actual_phi_probs>=((T_Y+T_G)/total_T)] = 'R'

    Y_occurance = np.count_nonzero(actual_phi_0 == 'Y')
    G_occurance = np.count_nonzero(actual_phi_0 == 'G')
    R_occurance = np.count_nonzero(actual_phi_0 == 'R')
    actual_t_phi_0 = np.full((actual_phi_probs.shape), 0, dtype=float)
    actual_t_phi_0[actual_phi_0=='Y'] = T_Y * np.random.random_sample(size= Y_occurance) 
    actual_t_phi_0[actual_phi_0=='G'] = T_G * np.random.random_sample(size= G_occurance) 
    actual_t_phi_0[actual_phi_0=='R'] = T_R * np.random.random_sample(size= R_occurance) 
    return actual_d_0[0], actual_v_0[0], actual_phi_0[0], actual_t_phi_0[0]



def observation_generation(actual_trajectories, std_d_m, std_v_m, phi_obs_mode, phi_obs_prob, max_phi_prob, min_phi_prob, d_upper, phi_list):
    obs_trajectories=[]
    
    for traj in actual_trajectories:
        obs_traj=[]
        for state in traj:
            d = state[0]
            v = state[1]
            phi = state[2]
            
            if phi_obs_mode=='linear':
                phi_obs_prob = min(((-(max_phi_prob-min_phi_prob)/d_upper)*d)+max_phi_prob,max_phi_prob)
                
            d_obs = d + np.random.normal(0,std_d_m,size=1)
            v_obs = v + np.random.normal(0,std_v_m,size=1)
            dummy_prob = np.random.uniform(0,1,1)
            if dummy_prob<=phi_obs_prob:
                phi_obs = phi
            else:
                list1 = [ele for ele in phi_list if ele != phi]
                phi_obs = random.choice(list1)

            obs_state = (d_obs[0], v_obs[0], phi_obs)
            obs_traj.append(obs_state)
        obs_trajectories.append(obs_traj)
    
    return obs_trajectories



def generate_policy_traj(N, Q_table, observ_mode, traversed_traj, traversed_traj_estimated, traversed_action, alarm_list, d_particle, v_particle, phi_particle, t_phi_particle, uncertainty_aware, actions, epsilon, \
                         discrete_d, discrete_v, discrete_phi_t, T_Y, T_R, v_max, delta_t, phi_T_dict, phi_prev_phi_dict, phi_next_phi_dict,\
                        std_d, std_v, std_t_phi, std_d_m, std_v_m, phi_obs_prob, max_phi_prob, min_phi_prob, d_upper, phi_list):
  
    idxs=[i for i, x in enumerate(alarm_list) if x]
    if len(idxs)==0:
        return [], [], []
    else:
        idx=idxs[0]
    policy_traj=traversed_traj[:idx+1].copy()
    policy_traj_estimated=traversed_traj_estimated[:idx+1].copy()
    policy_action=traversed_action[:idx+1].copy()
    state=policy_traj[-1]
    distance=state[0]
    estimated_state=policy_traj_estimated[-1]
    
    while distance>0:
        if uncertainty_aware:
            action = ql.PF_action_selection(Q_table, N, d_particle, v_particle, phi_particle, t_phi_particle, N*[1/N], actions, epsilon,\
                                              discrete_d, discrete_v, discrete_phi_t, T_Y, T_R,v_max, phi_prev_phi_dict)
        elif not(uncertainty_aware):
            estimated_idx = map_state_to_idx(estimated_state, discrete_d, discrete_v, discrete_phi_t, T_Y, T_R, phi_prev_phi_dict)
            action,_ = ql.action_selection(Q_table[estimated_idx], estimated_state[1], actions, epsilon, v_max, 0)
            
        state = update_state(state, action, delta_t, phi_T_dict, phi_next_phi_dict, std_d, std_v, std_t_phi, 0)
        distance = state[0]
        velocity = state[1]
        policy_traj.append(state)
        
        policy_action.append(action)
        
        observ = observation_generation([[state]], std_d_m, std_v_m, 'constant', phi_obs_prob, max_phi_prob, min_phi_prob, d_upper, phi_list)
        observ = observ[0][0]
        if observ_mode=='dphi':
            observ=(observ[0], observ[2])
        if observ_mode=='vphi':
            observ=(observ[1], observ[2])
        if observ_mode=='dv':
            observ=(observ[0], observ[1])

        
        d_particle, v_particle, phi_particle, t_phi_particle = ps.state_process(N, 1, d_particle, v_particle, phi_particle, t_phi_particle, action, \
                                                                                  std_d, std_v, std_t_phi, phi_T_dict,phi_next_phi_dict,0)
        weights = ps.particle_filter_weights(N, observ_mode, 'constant', observ, d_particle, v_particle, phi_particle, t_phi_particle, std_d_m, std_v_m, \
                                               phi_obs_prob, max_phi_prob, min_phi_prob,d_upper,std_t_phi)
        #normalizing the weights
        normalized_weights = ps.normalize(weights)
        #resampling
        d_particle, v_particle, phi_particle, t_phi_particle = ps.particles_resampling(N, normalized_weights, d_particle, v_particle, phi_particle, t_phi_particle)
        estimated_state = ps.state_estimation(np.asarray(N*[1/N]), d_particle, v_particle, phi_particle, t_phi_particle)
        policy_traj_estimated.append(estimated_state)

    return policy_traj, policy_traj_estimated, policy_action


