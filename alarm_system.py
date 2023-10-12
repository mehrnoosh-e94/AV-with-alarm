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
import state as st
import in_out as io
import variables as var


T_Y = var.T_Y
T_G = var.T_G
T_R = var.T_R
total_T = T_Y + T_G + T_R


def dist_based_alarm_detection(state, actual_action, Q_table, deceleration, critical_distance, t_phi_margin, alpha_N, discrete_d, discrete_v, discrete_phi_t, T_Y, T_R, phi_prev_phi_dict, phi_to_T_dict, phi_to_next_phi_dict, std_d, std_v, std_t_phi, actions, epsilon, v_max):
    N=100
    rrl=0
    
    d_particle = np.full((N,1), state[0])
    v_particle = np.full((N,1), state[1])
    phi_particle = np.full((N,1), state[2])
    t_phi_particle = np.full((N,1), state[3])
    
    prev_d_particle=d_particle.copy()
    prev_v_particle=v_particle.copy()
    prev_phi_particle=phi_particle.copy()
    prev_t_phi_particle=t_phi_particle.copy()
    
    action=actual_action
    
    init_state=state
    distance=state[0]

    while distance>0:
        action = ql.PF_action_selection(Q_table, N, d_particle, v_particle, phi_particle, t_phi_particle, N*[1/N], actions, epsilon,discrete_d, discrete_v, discrete_phi_t, T_Y, T_R,v_max, phi_prev_phi_dict)
        
        prev_d_particle=d_particle.copy()
        prev_v_particle=v_particle.copy()
        prev_phi_particle=phi_particle.copy()
        prev_t_phi_particle=t_phi_particle.copy()
        
        d_particle, v_particle, phi_particle, t_phi_particle = ps.state_process(N, 1, d_particle, v_particle, phi_particle, t_phi_particle, action, std_d, std_v, std_t_phi, phi_to_T_dict,phi_to_next_phi_dict,0)
        
        state = st.update_state(state, action, 1, phi_to_T_dict, phi_to_next_phi_dict, std_d, std_v, std_t_phi, 0)
        distance = state[0]
    
    for i in range(N):
        prev_state=(prev_d_particle[i][0], prev_v_particle[i][0], prev_phi_particle[i][0], prev_t_phi_particle[i][0])
        state=(d_particle[i][0], v_particle[i][0], phi_particle[i][0], t_phi_particle[i][0])

        if RRL_test(prev_state, state, action, phi_to_T_dict):
            rrl+=1

    alarm=0
    if rrl>(alpha_N*N):

        for i in range(N):
            state = init_state
            velocity = state[1]
            distance = state[0]
            action = deceleration
            while (velocity>0) and (distance>0):
                idx = st.map_state_to_idx(state, discrete_d, discrete_v, discrete_phi_t, T_Y, T_R, phi_prev_phi_dict)
                state = st.update_state(state, action, 1, phi_to_T_dict, phi_to_next_phi_dict, std_d, std_v, std_t_phi, 0)
                velocity = state[1]
                distance = state[0]
            if velocity<=0:
                if (distance<critical_distance) and ((state[2]=='R') or ((state[2]=='G') and (state[3]<t_phi_margin)) or ((state[2]=='Y') and (state[3]>T_Y-t_phi_margin))):
                    alarm+=1
            if distance<=0:
                if (velocity>0) and ((state[2]=='R') or ((state[2]=='G') and (state[3]<t_phi_margin)) or ((state[2]=='Y') and (state[3]>T_Y-t_phi_margin))):
                    alarm+=1
 
    if alarm>(alpha_N*N):

        return True
        
    return False


def point_based_alarm_detection(state, actual_action, Q_table, deceleration, critical_distance, t_phi_margin, alpha_N, discrete_d, discrete_v, discrete_phi_t, T_Y, T_R, phi_prev_phi_dict, phi_to_T_dict, phi_to_next_phi_dict, std_d, std_v, std_t_phi, actions, epsilon, v_max):

    init_state = state
    prev_state=state
    if init_state[0]<=0:
        action=actual_action
    rrl=False
    distance = state[0]
    velocity = state[1]
    while distance>0:

        idx = st.map_state_to_idx(state, discrete_d, discrete_v, discrete_phi_t, T_Y, T_R, phi_prev_phi_dict)
        action, _ = ql.action_selection(Q_table[idx], velocity, actions, epsilon, v_max, 0)
        prev_state=state
        state = st.update_state(state, action, 1, phi_to_T_dict, phi_to_next_phi_dict, std_d, std_v, std_t_phi, 0)
        distance = state[0]
        velocity = state[1]
    
    
    if RRL_test(prev_state, state, action, phi_to_T_dict):
        state = init_state
        distance = state[0]
        velocity = state[1]
        action = deceleration
        while (velocity>0) and (distance>0):
            idx = st.map_state_to_idx(state, discrete_d, discrete_v, discrete_phi_t, T_Y, T_R, phi_prev_phi_dict)
            state = st.update_state(state, action, 1, phi_to_T_dict, phi_to_next_phi_dict, std_d, std_v, std_t_phi, 0)
            distance = state[0]
            velocity = state[1]

        if velocity<=0:
            if (distance<critical_distance) and ((state[2]=='R') or ((state[2]=='G') and (state[3]<t_phi_margin)) or ((state[2]=='Y') and (state[3]>T_Y-t_phi_margin))):
          
                return True
        if distance<=0:
            if (velocity>0) and ((state[2]=='R') or ((state[2]=='G') and (state[3]<t_phi_margin)) or ((state[2]=='Y') and (state[3]>T_Y-t_phi_margin))):
                return True
        
    
    return False

def quadratic_equation(a,b,c):
    dis = (b**2) - (4 * a*c)
    # find two results
    ans1 = (-b-cmath.sqrt(dis))/(2 * a)
    ans2 = (-b + cmath.sqrt(dis))/(2 * a)
    return ans1, ans2


def RRL_test(prev_state, state, action, phi_T_dict):
    if state[0]==0:
        if (state[1]>0) and (state[3]=='R'):
            return True
        
#     if state[0]>0:
#         while state[0]>0:
#             idx = map_state_idx
#             action = 
#             prev_state=state
#             state = udpate_state
#         if RRL_test(prev_state, state):
#             return True
#         else:
#             return False
        
    if state[0]<0:
        t1, t2 = quadratic_equation(action/2, prev_state[1], -1*prev_state[0])
#         print(t1.real, t1)
        t1 = t1.real
        t2 = t2.real
        
        if (t1>0) and (t1<=1):
            pass_t = prev_state[3]+t1
            pass_v = prev_state[1]+action*t1
            if pass_t>=phi_T_dict[prev_state[2]]:
                pass_t = pass_t-phi_T_dict[prev_state[2]]
                pass_phi = state[2]
            elif pass_t<phi_T_dict[prev_state[2]]:
                pass_phi=prev_state[2]
                
        if (t2>0) and (t2<=1):
            pass_t = prev_state[3]+t2
            pass_v = prev_state[1]+action*t2
            if pass_t>=phi_T_dict[prev_state[2]]:
                pass_t = pass_t-phi_T_dict[prev_state[2]]
                pass_phi = state[2]
            elif pass_t<phi_T_dict[prev_state[2]]:
                pass_phi=prev_state[2]
        else:
            pass_v = state[1]
            pass_phi = state[2]
            pass_t = state[3]
            
        if (pass_v>0) and (pass_phi=='R'):

            return True

    return False



def AV_with_alarm(file_root, N, Q_table, init_states, uncertainty_aware, observ_mode, delta_t, actions, deceleration_a, critical_d_list, t_phi_margin_list, alpha_N_list, v_max, rounds, epsilon, traj_count, std_d, std_v, std_t_phi, std_d_m, std_v_m, phi_obs_prob, max_phi_prob, min_phi_prob,\
                 d_lower, d_upper, v_lower, v_upper, T_Y, T_G, T_R, discrete_d, discrete_v, discrete_phi_t, phi_list, phi_prev_phi_dict, phi_T_dict, phi_next_phi_dict):

    action=0

    for critical_d, t_phi_margin, alpha_N in zip(critical_d_list, t_phi_margin_list, alpha_N_list):
        for rnd in range(rounds):
            traversed_trajs=[]
            traversed_trajs_estimated=[]
            traversed_actions=[]
            alarms=[]

            policy_trajs=[]
            policy_trajs_estimated=[]
            policy_actions=[]
            for i in range(traj_count):
                state=init_states[i]
                distance=state[0]
                velocity=state[1]
                observ = st.observation_generation([[state]], std_d_m, std_v_m, 'constant', phi_obs_prob, max_phi_prob, min_phi_prob, d_upper, phi_list)
                observ = observ[0][0]
                if observ_mode=='dphi':
                    observ=(observ[0], observ[2])
                if observ_mode=='vphi':
                    observ=(observ[1], observ[2])
                if observ_mode=='dv':
                    observ=(observ[0], observ[1])
                    
                d_particle, v_particle, phi_particle, t_phi_particle = ps.particle_initialization(N, d_lower, d_upper, v_lower, v_upper, T_Y, T_G, T_R)

                alarm=False
                action_set=False
                alarmed=0
                alarm_list=[]
                traversed_traj=[]
                traversed_traj_estimated=[]
                traversed_action=[]

                while distance>0:
                    if action_set and (velocity<=0):
                        break
                    traversed_traj.append(state)

                    #Weighing
                    weights = ps.particle_filter_weights(N, observ_mode, 'constant', observ, d_particle, v_particle, phi_particle, t_phi_particle, std_d_m, std_v_m, phi_obs_prob, max_phi_prob, min_phi_prob,d_upper,std_t_phi)
                    #normalizing the weights
                    normalized_weights = ps.normalize(weights)
                    #resampling
                    d_particle, v_particle, phi_particle, t_phi_particle = ps.particles_resampling(N, normalized_weights, d_particle, v_particle, phi_particle, t_phi_particle)
                    #state_estimation
                    estimated_state = ps.state_estimation(np.asarray(N*[1/N]), d_particle, v_particle, phi_particle, t_phi_particle)
                    traversed_traj_estimated.append(estimated_state)
#                             print(f'Estimate State: {estimated_state}')
                    
                    if uncertainty_aware:
                        alarm = dist_based_alarm_detection(estimated_state, action, Q_table, deceleration_a, critical_d, t_phi_margin, alpha_N, discrete_d,
                                                 discrete_v,discrete_phi_t, T_Y, T_R, phi_prev_phi_dict, phi_T_dict, phi_next_phi_dict,
                                                 std_d, std_v, std_t_phi, actions, epsilon+0.1, v_max)
                    elif not(uncertainty_aware):
                        alarm = point_based_alarm_detection(estimated_state, action, Q_table, deceleration_a, critical_d, t_phi_margin, alpha_N, discrete_d,
                                                 discrete_v,discrete_phi_t, T_Y, T_R, phi_prev_phi_dict, phi_T_dict, phi_next_phi_dict,
                                                 std_d, std_v, std_t_phi, actions, epsilon, v_max)
                    alarm_list.append(alarm)
                    if alarm:
                        alarmed+=1
                        action=-3
                        action_set=True
                        if alarmed==1:
                            alarmed_d_particle=d_particle.copy()
                            alarmed_v_particle=v_particle.copy()
                            alarmed_phi_particle=phi_particle.copy()
                            alarmed_t_phi_particle=t_phi_particle.copy()

                    else:
                        action_set=False

                    if not action_set:

                        if uncertainty_aware:
                            action = ql.PF_action_selection(Q_table, N, d_particle, v_particle, phi_particle, t_phi_particle, N*[1/N], actions, epsilon,discrete_d, discrete_v, discrete_phi_t, T_Y, T_R,v_max, phi_prev_phi_dict)
                        elif not(uncertainty_aware):
                            estimated_idx = st.map_state_to_idx(estimated_state, discrete_d, discrete_v, discrete_phi_t, T_Y, T_R, phi_prev_phi_dict)
                            action,_ = ql.action_selection(Q_table[estimated_idx], estimated_state[1], actions, epsilon, v_max, 0)

                    traversed_action.append(action)

                    state = st.update_state(state, action, delta_t, phi_T_dict, phi_next_phi_dict, std_d, std_v, std_t_phi, 0)
                    distance = state[0]
                    velocity = state[1]

                    #observation generation
                    observ = st.observation_generation([[state]], std_d_m, std_v_m, 'constant', phi_obs_prob, max_phi_prob, min_phi_prob, d_upper, phi_list)
                    observ = observ[0][0]
                    if observ_mode=='dphi':
                        observ=(observ[0], observ[2])
                    if observ_mode=='vphi':
                        observ=(observ[1], observ[2])
                    if observ_mode=='dv':
                        observ=(observ[0], observ[1])
                    
                    #state process for the particles
                    d_particle, v_particle, phi_particle, t_phi_particle = ps.state_process(N, 1, d_particle, v_particle, phi_particle, t_phi_particle, action, std_d, std_v, std_t_phi, phi_T_dict,phi_next_phi_dict,0)

                traversed_traj.append(state)

                #Weighing
                weights = ps.particle_filter_weights(N, observ_mode, 'constant', observ, d_particle, v_particle, phi_particle, t_phi_particle, std_d_m, std_v_m, phi_obs_prob, max_phi_prob, min_phi_prob,d_upper,std_t_phi)
                #normalizing the weights
                normalized_weights = ps.normalize(weights)
                #resampling
                d_particle, v_particle, phi_particle, t_phi_particle = ps.particles_resampling(N, normalized_weights, d_particle, v_particle, phi_particle, t_phi_particle)
                #state_estimation
                estimated_state = ps.state_estimation(np.asarray(N*[1/N]), d_particle, v_particle, phi_particle, t_phi_particle)
                traversed_traj_estimated.append(estimated_state)


                alarms.append(alarm_list)
                traversed_trajs.append(traversed_traj)
                traversed_trajs_estimated.append(traversed_traj_estimated)
                traversed_actions.append(traversed_action)

                if alarmed>0:
                    policy_traj, policy_traj_estimated, policy_action=st.generate_policy_traj(N, Q_table, observ_mode, traversed_traj, traversed_traj_estimated, traversed_action, alarm_list, alarmed_d_particle, alarmed_v_particle, alarmed_phi_particle, alarmed_t_phi_particle, uncertainty_aware, actions, epsilon, \
                                     discrete_d, discrete_v, discrete_phi_t, T_Y, T_R, v_max, delta_t, phi_T_dict, phi_prev_phi_dict, phi_next_phi_dict,\
                                    std_d, std_v, std_t_phi, std_d_m, std_v_m, phi_obs_prob, max_phi_prob, min_phi_prob, d_upper, phi_list)
                else:
                    policy_traj=traversed_traj.copy()
                    policy_traj_estimated=traversed_traj_estimated.copy()
                    policy_action=traversed_action.copy()

                policy_trajs.append(policy_traj)
                policy_trajs_estimated.append(policy_traj_estimated)
                policy_actions.append(policy_action)


            io.write_list(alarms, file_root+'epsilon '+str(epsilon)+'/alarms_d'+str(critical_d)+'_t'+str(t_phi_margin)+'_a'+str(alpha_N)+'_e'+str(epsilon)+'_rnd'+str(rnd)+'_unc'+str(uncertainty_aware))
            io.write_list(traversed_trajs, file_root+'epsilon '+str(epsilon)+'/traversed_trajs_d'+str(critical_d)+'_t'+str(t_phi_margin)+'_a'+str(alpha_N)+'_e'+str(epsilon)+'_rnd'+str(rnd)+'_unc'+str(uncertainty_aware))
            io.write_list(traversed_trajs_estimated, file_root+'epsilon '+str(epsilon)+'/etraversed_trajs_estimated_d'+str(critical_d)+'_t'+str(t_phi_margin)+'_a'+str(alpha_N)+'_e'+str(epsilon)+'_rnd'+str(rnd)+'_unc'+str(uncertainty_aware))
            io.write_list(traversed_actions, file_root+'epsilon '+str(epsilon)+'/traversed_actions_d'+str(critical_d)+'_t'+str(t_phi_margin)+'_a'+str(alpha_N)+'_e'+str(epsilon)+'_rnd'+str(rnd)+'_unc'+str(uncertainty_aware))
            io.write_list(policy_trajs, file_root+'epsilon '+str(epsilon)+'/policy_trajs_d'+str(critical_d)+'_t'+str(t_phi_margin)+'_a'+str(alpha_N)+'_e'+str(epsilon)+'_rnd'+str(rnd)+'_unc'+str(uncertainty_aware))
            io.write_list(policy_trajs_estimated, file_root+'epsilon '+str(epsilon)+'/policy_trajs_estimated_d'+str(critical_d)+'_t'+str(t_phi_margin)+'_a'+str(alpha_N)+'_e'+str(epsilon)+'_rnd'+str(rnd)+'_unc'+str(uncertainty_aware))
            io.write_list(policy_actions, file_root+'epsilon '+str(epsilon)+'/policy_actions_d'+str(critical_d)+'_t'+str(t_phi_margin)+'_a'+str(alpha_N)+'_e'+str(epsilon)+'_rnd'+str(rnd)+'_unc'+str(uncertainty_aware))

