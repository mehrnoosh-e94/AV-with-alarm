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

import state as st
import qlearning as ql
import plot as myplt
import in_out as io
import variables as var

T_Y = var.T_Y
T_G = var.T_G
T_R = var.T_R
total_T = T_Y + T_G + T_R



def state_process(N, steps, d_0, v_0, phi_0, t_phi_0, a, std_d, std_v, std_t_phi, phi_T_dict,phi_next_phi_dict,training):
    steps = steps + 1
    d = np.zeros((N, steps))
    v = np.zeros((N, steps))
    phi = np.full((N, steps), '-')
    t_phi = np.zeros((N, steps))

    d[:, 0] = d_0.flatten()
    v[:, 0] = v_0.flatten()
    phi[:, 0] = phi_0.flatten()
    t_phi[:, 0] = t_phi_0.flatten()

    delta_t = 1
    for i in range(1,steps):
        
        d[:,i] = d[:,i-1] - v[:,i-1]*delta_t - 0.5*a*(delta_t**2) + np.random.normal(0, std_d, N)
        v[:,i] = v[:,i-1] + a*delta_t + np.random.normal(0, std_v, N)
        prev_T_phi = np.vectorize(phi_T_dict.get)(phi[:,i-1])
        curr_phi = np.vectorize(phi_next_phi_dict.get)(phi[:,i-1])
        
        for j in range(0,N):
            if (v[j,i]<0) and not(training):
                d[j,i]-=0.5*np.abs(v[j,i])
                v[j,i]=0
                
            rnd_tphi = np.random.normal(0, std_t_phi, 1)              
            if (t_phi[j,i-1] + delta_t + rnd_tphi) <= prev_T_phi[j]:
                phi[j,i] = phi[j,i-1]
                t_phi[j,i] = t_phi[j,i-1] + delta_t + rnd_tphi
            else:
                t_phi[j,i] = max(t_phi[j,i-1] + delta_t + rnd_tphi - prev_T_phi[j],0)
                phi[j,i] = curr_phi[j]
                


                
    return d[:,1:], v[:,1:], phi[:,1:], t_phi[:,1:]



def phi_likelihood(phi_obs_prob, phi_obs_prob_c, phi_obs, phi_particle):
    if phi_obs == phi_particle:
        return phi_obs_prob
    else:
        return phi_obs_prob_c

def normalize(w):
    n = len(w)
    w = np.asarray(w)
    w_sum = np.sum(w)
    normalized_w = w/w_sum
    return normalized_w




def state_estimation(normalized_weights, curr_particle_d, curr_particle_v, curr_particle_phi, curr_particle_t_phi):
    estimated_d = np.dot(normalized_weights, curr_particle_d)
    estimated_v = np.dot(normalized_weights, curr_particle_v)
    red_weights = np.sum(normalized_weights[curr_particle_phi.flatten()=='R'])
    yellow_weights = np.sum(normalized_weights[curr_particle_phi.flatten()=='Y'])
    green_weights = np.sum(normalized_weights[curr_particle_phi.flatten()=='G'])
    max_w = max(red_weights, yellow_weights, green_weights)
    

    if max_w == red_weights:
        estimated_phi = 'R'
        estimated_t_phi = np.dot(normalized_weights[curr_particle_phi.flatten()=='R'], curr_particle_t_phi[curr_particle_phi.flatten()=='R'])
        
    elif max_w == yellow_weights:
        estimated_phi = 'Y'
        estimated_t_phi = np.dot(normalized_weights[curr_particle_phi.flatten()=='Y'], curr_particle_t_phi[curr_particle_phi.flatten()=='Y'])
        
    elif max_w == green_weights:
        estimated_phi = 'G'
        estimated_t_phi = np.dot(normalized_weights[curr_particle_phi.flatten()=='G'], curr_particle_t_phi[curr_particle_phi.flatten()=='G'])

    return estimated_d[0], estimated_v[0], estimated_phi, estimated_t_phi[0]



def particle_initialization(N, d_lower, d_upper, v_lower, v_upper, T_Y, T_G, T_R):
    total_T = T_Y + T_G + T_R
    d_particle_0 = np.random.uniform(d_lower, d_upper, (N,1))
    v_particle_0 = np.random.uniform(v_lower, v_upper, (N,1))
    phi_probs = np.random.uniform(0, 1, (N,1))
    
    phi_particle_0 = np.full((phi_probs.shape), '-')
    phi_particle_0[phi_probs<(T_Y/total_T)] = 'Y'
    phi_particle_0[((phi_probs>=(T_Y/total_T)) & (phi_probs<((T_Y+T_G)/total_T)))] = 'G'
    phi_particle_0[phi_probs>=((T_Y+T_G)/total_T)] = 'R'
    
    Y_occurance = np.count_nonzero(phi_particle_0 == 'Y')
    G_occurance = np.count_nonzero(phi_particle_0 == 'G')
    R_occurance = np.count_nonzero(phi_particle_0 == 'R')
    t_phi_particle_0 = np.full((phi_probs.shape), 0, dtype=float)

    t_phi_particle_0[phi_particle_0 =='Y'] = T_Y * np.random.random_sample(size= Y_occurance) 
    t_phi_particle_0[phi_particle_0 =='G'] = T_G * np.random.random_sample(size= G_occurance) 
    t_phi_particle_0[phi_particle_0 =='R'] = T_R * np.random.random_sample(size= R_occurance) 
        
    return d_particle_0, v_particle_0, phi_particle_0, t_phi_particle_0



def particle_filter_weights(N, obs_scenario, phi_prob_scenario, obs_state, d_particle, v_particle, phi_particle, t_phi_particle, std_d_m, std_v_m, phi_obs_prob, max_phi_prob, min_phi_prob,d_upper,std_t_phi):
    weights = []
    phi_obs_prob_c = (1-phi_obs_prob)/2
    
    if obs_scenario=='dvphitphi':
        weights = N*[1]

    if obs_scenario=='dvphi':
        d_obs = obs_state[0]
        v_obs = obs_state[1]
        phi_obs = obs_state[2]

        for j in range(0,N):
            
            if phi_prob_scenario=='linear':
                phi_obs_prob = min(((-(max_phi_prob-min_phi_prob)/d_upper)*d_particle[j,0])+max_phi_prob,max_phi_prob)                
                phi_obs_prob_c = (1-phi_obs_prob)/2
                
            obs_likelihood = norm.pdf(d_obs, d_particle[j,0], std_d_m) * norm.pdf(v_obs, v_particle[j,0], std_v_m) * phi_likelihood(phi_obs_prob, phi_obs_prob_c, phi_obs, phi_particle[j,0])
            weights.append(obs_likelihood)
        
    if obs_scenario=='dv':
        d_obs = obs_state[0]
        v_obs = obs_state[1]
    
        for j in range(0,N):
            if phi_prob_scenario=='linear':
                phi_obs_prob = min(((-(max_phi_prob-min_phi_prob)/d_upper)*d_particle[j,0])+max_phi_prob,max_phi_prob)
                phi_obs_prob_c = (1-phi_obs_prob)/2
            
            obs_likelihood = norm.pdf(d_obs, d_particle[j,0], std_d_m) * norm.pdf(v_obs, v_particle[j,0], std_v_m)
            weights.append(obs_likelihood)
            
    if obs_scenario=='vphi':
        v_obs = obs_state[0]
        phi_obs = obs_state[1]

        for j in range(0,N):
            if phi_prob_scenario=='linear':
                phi_obs_prob = min(((-(max_phi_prob-min_phi_prob)/d_upper)*d_particle[j,0])+max_phi_prob,max_phi_prob)               
                phi_obs_prob_c = (1-phi_obs_prob)/2
            
            obs_likelihood = norm.pdf(v_obs, v_particle[j,0], std_v_m) * phi_likelihood(phi_obs_prob, phi_obs_prob_c, phi_obs, phi_particle[j,0])
            weights.append(obs_likelihood)
            
    if obs_scenario=='dphi':
        d_obs = obs_state[0]
        phi_obs = obs_state[1]
        for j in range(0,N):
            if phi_prob_scenario=='linear':
                phi_obs_prob = min(((-(max_phi_prob-min_phi_prob)/d_upper)*d_particle[j,0])+max_phi_prob,max_phi_prob)                
                phi_obs_prob_c = (1-phi_obs_prob)/2
            
            obs_likelihood = norm.pdf(d_obs, d_particle[j,0], std_d_m) * phi_likelihood(phi_obs_prob, phi_obs_prob_c, phi_obs, phi_particle[j,0])
            weights.append(obs_likelihood)
            
    return weights



def particles_resampling(N, normalized_weights, d_particle, v_particle, phi_particle, t_phi_particle):
    
    cum_weights = np.cumsum(normalized_weights)
    random_w = np.random.uniform(0,1,N)
    sampled_idx = []
    for idx, w in enumerate(random_w):
        min_val = cum_weights[cum_weights>w].min()
        sampled_idx.append(np.where(cum_weights==min_val)[0][0])

    #resampling
    d_particle = d_particle[sampled_idx]
    v_particle = v_particle[sampled_idx]
    phi_particle = phi_particle[sampled_idx]
    t_phi_particle = t_phi_particle[sampled_idx]
    
    return d_particle, v_particle, phi_particle, t_phi_particle



