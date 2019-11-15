# Train on RedGreen task and transfer to P
# coding: utf-8

# In[1]:

import sys
import micro_agent_class as ma
sys.path.append('../')
#import matplotlib.pyplot as plt
#import plotly
import numpy as np
import pandas as pd
#import plotly.express as px
import random
import hrr2 as hrr
import math
import state_machine_class as state_machine
import Gate_Class_v4_ as gate
import time
from IO_gate_model_forgetmodel_Transfer_functions import Transfer_Task
#from plotly.graph_objs import Scatter, Layout, Surface
#plotly.offline.init_notebook_mode(connected=True)
#import plotly.graph_objects as go

# In[2]:

# reward for transfer task
def reward(loc,cue,size):
    if cue == 'red':
        if loc[0] == 0 and loc[1] == 0:
            return 1
        else:
            return 0
    elif cue == 'green': 
        if loc[0] == 0 and loc[1] == size[1]-1:
            return 1
        else:
            return 0
    elif cue == 'purple': # middle
        if loc[0] == 0 and loc[1] == 2:
            return 1
        else:
            return 0


def reward_v2(loc,cue,size):
    if cue == 'red':
        if loc[0] == 2 and loc[1] == 0:
            return 1
        else:
            return 0
    elif cue == 'green': 
        if loc[0] == 2 and loc[1] == size[1]-1:
            return 1
        else:
            return 0
    elif cue == 'purple': # middle
        if loc[0] == 0 and loc[1] == 2:
            return 1
        else:
            return 0

# In[3]:

# Error for Transfer Task
def optimal_path_length_bottle(init_loc,cue):
    row,col = init_loc[0],init_loc[1]
    red = [ [0,1,2,3,4],
            [1,2,3,4,5],
            [6,5,4,5,6],
            [7,6,5,6,7],
            [8,7,6,7,8] ]
    
    green = [ [4,3,2,1,0],
              [5,4,3,2,1],
              [6,5,4,5,6],
              [7,6,5,6,7],
              [8,7,6,7,8] ]
    
    purple = [ [2,1,0,1,2],
               [3,2,1,2,3],
               [4,3,2,3,4],
               [5,4,3,4,5],
               [6,5,4,5,6] ]
    
    if cue == 'red': # red
        return red[row][col]
    elif cue == 'green':
        return green[row][col]
    elif cue == 'purple':
        return purple[row][col]

def optimal_path_length_bottle_v2(init_loc,cue):
    row,col = init_loc[0],init_loc[1]
    red = [ [2,3,4,5,6],
            [1,2,3,4,5],
            [0,1,2,3,4],
            [5,4,3,4,5],
            [6,5,4,5,6],
            [7,6,5,6,7] ]
    
    green = [ [6,5,4,3,2],
              [5,4,3,2,1],
              [4,3,2,1,0],
              [5,4,3,4,5],
              [6,5,4,5,6],
              [7,6,5,6,7] ]
    
    purple = [ [2,1,0,1,2],
               [3,2,1,2,3],
               [4,3,2,3,4],
               [5,4,3,4,5],
               [6,5,4,5,6],
               [7,6,5,6,7] ]
    
    if cue == 0: # red
        return red[row][col]
    elif cue == 1:
        return green[row][col]
    elif cue == 2:
        return purple[row][col]
    
# stores 1D list of weights in file
def store_weights(fname,datalist):
    f_obj = open(fname,"a")
    for item in datalist:
        f_obj.write(str(item))
        f_obj.write("\n")
    f_obj.close()

# store 1D list as csv in file
def store_weights_csv_row(fname,datalist):
    f_obj = open(fname,"a")
    for index in range(len(datalist)):
        f_obj.write(str(datalist[index]))
        if index < len(datalist)-1:
            f_obj.write(",")
        else:
            f_obj.write("\n")
    f_obj.close()
# In[12]:
# Train on the RG task
def IO_gate_model(state_size,nepisodes,stat_window,ep_0):
    #n = 512
    beginTime = time.time()
    num_agents = 100
    n = 1024
    hrr_size = n
    nagent_actions = 4
    gate_actions = 2
    #ncolors = 3
    nslots = 1
    nroles = 1
    nrolestates = 2
    Closed,Open = 0,1
    agent_actions = ['up','down','left','right']
    colors = ['red','green',''] # external colors
    #colors = ['red','green','purple',''] # external colors
    ncolors = len(colors)
    nothing = len(colors)-1
    #wm_colors = ['wm_red','wm_green',''] # internal colors
    roles = ['role_avail','role_unavail']
    gate_actions = ['closed','open']
    gateclosed = 'gateclosed'
    gateopen = 'gateopen'
    
    nrows,ncols = state_size
    state_table = state_machine.state_machine.generate_state_table_bottle(nrows,ncols)
    #state_table = state_machine.state_machine.generate_state_table_bottle_v2(nrows,ncols) # additional top row
    
    LTM_obj = hrr.LTM(n,True)
    #agent = gate.Q_learning(n,LTM_obj,epsilon=0.05,bias=1)
    agent = gate.Q_learning(n,LTM_obj,epsilon=0.1,bias=1)
    #i_gate = ma.micro_agents(num_agents,hrr_size,LTM_obj,bias=1,lrate=0.1,td_lambda=0.9,epsilon=0.01)
    i_gate = gate.Q_learning(n,LTM_obj,epsilon=0.1,bias=1)
    o_gate = gate.Q_learning(n,LTM_obj,epsilon=0.1,bias=1)
    #o_gate = ma.micro_agents(num_agents,hrr_size,LTM_obj,bias=1,lrate=0.1,td_lambda=0.9,epsilon=0.01)
    WM = gate.wm_content(colors,nslots,LTM_obj)
    #print('number of items',len(WM.wm_items))
    #print('items',WM.wm_items[0],WM.wm_items[1],WM.wm_items[2])
    s_mac = state_machine.state_machine(state_table)
    nsteps = 100
    goal = [ [0,0],[0,ncols-1] ]
    #goal = [ [0,0],[0,ncols-1],[0,ncols//2] ]
    #goal = [ [2,0],[2,ncols-1] ]
    #goal = [ [2,0],[2,ncols-1],[0,ncols//2] ] # make goals equal distance
    value_function = np.zeros((nrows,ncols)) # store Q values for statitics
     # track performance
    opt_array = []
    diff_array = []
    t1 = ""
    
    #handhold = 400000
    anneal = ep_0
    
    o_gate_correct = 0 # stats for output gate
    o_gate_total = 0 # stats for output gate
    o_gate_correct_shared = 0
    o_gate_correct_non = 0
    shared_total = 0
    non_total = 0
    o_gate_stat = []
    mean_o_gate = []
    accuracy = []
    acc_array = []
    i_gate_init_correct = []
    mean_acc = 0
    testing = False
    check = True
    r = 0
    episode = 1
    #for episode in range(nepisodes):
    memory_states = []
    memory_states_testing = []
    memory = ''
    correct_mem_use = 'n/a'
    diff_save = []
    diff_save_red, diff_red = [],[]
    diff_save_green, diff_green = [],[]
    diff_save_purple, diff_purple = [],[]
    test_flag = False
    while episode < nepisodes+1:
        never_opened_flag = True
        ATR = ''
        WM.flush_all_wm_maint() # flushes contents of working memory
        #print('episode:',episode)
        o_gate_correct_shared_local = 0
        shared_total_local = 0
        count = 0
        #t1 = str(count)
        t1 = ''
        row = random.randrange(2,nrows) # get random row for state (in shared area)
        col = random.randrange(0,ncols) # get random col for state
        cur_loc = [row,col] # current location of agent
        color_signal = random.randrange(0,ncolors-1) # get random color
        color = color_signal # used for reward because color signal may change
        '''
        if color_signal == nothing:
            color = random.randrange(0,ncolors-1)
        '''
        #r = reward(cur_loc,colors[color],state_size) if color!=nothing else 0
        '''
        if episode == anneal:
            agent.epsilon,i_gate.epsilon,o_gate.epsilon = 0,0,0
        '''
        if check and episode == anneal:
            agent.epsilon,i_gate.epsilon,o_gate.epsilon = 0,0,0
            testing = True
            check = False            
        
        if testing:
            nepisodes = 5*stat_window
            episode = 0
            testing = False
            test_flag = True
        
        #optimal_steps = optimal_path_length(cur_loc,color_signal) # get optimal steps to goal 
        optimal_steps = optimal_path_length_bottle(cur_loc,colors[color_signal]) # get optimal steps to goal 
        #optimal_steps = optimal_path_length_bottle_v2(cur_loc,color_signal) # get optimal steps to goal
        state = (state_table[row][col])["state"] # current state
        
        role_i = 0 if color_signal!=nothing else 1
        slot = 0
        igate_restrict = [0,1] # restricted set 
        ogate_restrict = [0,1] # restricted set 

        '''
        if episode < handhold: # hand holding
            ogate_restrict = [0] if row >= 2 else [1]
        else:
            ogate_restrict = [0,1]
        '''
        #print("cue:",colors[color_signal])
        #print("role_i:",roles[role_i])
        i_gate_input = roles[role_i]+'*'+state+'*'+t1
        LTM_obj.encode(i_gate_input)
        igate_restrict = [0] if color_signal == nothing else [0,1] # gate can only open if something is present
        i_gate_state,i_value,i_max,i_input,i_values = i_gate.action(i_gate_input,gate_actions,igate_restrict)
        WM.wm_in_flow(i_gate_state,slot,color_signal) # control flow of wm contents
        #print("i_gate:",i_gate_state)
        #print("wm_in:",WM.get_one_wm_maint(slot))
        role_o = 1 if WM.wm_maint_slot_is_empty(slot) else 0
        #print("role_o:",roles[role_o])
        o_gate_input = roles[role_o]+'*'+state+'*'+t1
        LTM_obj.encode(o_gate_input)
        ogate_restrict = [0] if WM.wm_maint_slot_is_empty(slot) else [0,1] # gate can only open if memory is present in slot
        o_gate_state,o_value,o_max,o_input,o_values = o_gate.action(o_gate_input,gate_actions,ogate_restrict)
        WM.wm_out_flow(o_gate_state,slot) # wm out contents for given slots
        wm_out = WM.get_one_wm_output(slot)
        #print("o_gate:",o_gate_state)
        #print("wm_out:",WM.get_one_wm_output(slot))
        #print("\n\n")
        '''
        if o_gate_state == Closed and not WM.wm_maint_slot_is_empty(slot):
            agent_input = gateclosed+'*'+colors[color_signal]+'*'+state+'*'+wm_out+'*'+t1
            LTM_obj.encode(agent_input)
        else:
            agent_input = colors[color_signal]+'*'+state+'*'+wm_out+'*'+t1
            LTM_obj.encode(agent_input)
            print("before:",LTM_obj.encode(WM.get_one_wm_maint(slot)))
            WM.update_wm_maint(slot,nothing)
            print("after:",LTM_obj.encode(WM.get_one_wm_maint(slot)))
        '''
        # context for it agent is remembering initial color que or not
        #print(colors[color_signal])
        #print(gate_actions[i_gate_state])
        #print(gate_actions[o_gate_state])
        
        if o_gate_state == Closed and never_opened_flag:
            if not WM.wm_maint_slot_is_empty(slot):
                ATR = 'remember'
            else:
                ATR = 'not_remember'
        
        if o_gate_state == Open:
            #pass
            WM.update_wm_maint(slot,nothing)
            memory_states.append(state)
            if test_flag:
                memory_states_testing.append(state)
            memory = state
            ATR = 'used'
            never_opened_flag = False
        
        #print(ATR)
        #print()
        agent_input = colors[color_signal]+'*'+state+'*'+wm_out+'*'+ATR
        LTM_obj.encode(agent_input)
        action_restrict = (state_table[row][col])["action_restrict"]# get restricted action set
        action,a_value,a_max,a_input,a_values = agent.action(agent_input,agent_actions,action_restrict) # agent action
        #print("agent_value: ",a_value)
        #print("input_value: ",i_value)
        #print("output_value:",o_value)
        
        o_gate_correct_shared += 1 if o_gate_state == Closed else 0
        o_gate_correct_shared_local += 1 if o_gate_state == Closed else 0
        shared_total += 1
        shared_total_local += 1
        if (color_signal != nothing and i_gate_state == Open) or (color_signal == nothing and i_gate_state == Closed):
            i_gate_init_correct.append(1)
        else:
            i_gate_init_correct.append(0)
        
        i_gate.set_eligibility_zero()
        o_gate.set_eligibility_zero()
        agent.set_eligibility_zero() # set eligibility trace to zero
        WM.flush_all_wm_output()
        count += 1
        for step in range(nsteps):
            #t1 = str(count)
            t1 = ''
            r = reward(cur_loc,colors[color],state_size) if color!=nothing else 0
            #r = reward_v2(cur_loc,colors[color],state_size) if color!=nothing else 0
            #r = punishment(cur_loc,colors[color],state_size) if color!=nothing else -1
            #print(state)
            if color!=nothing and (cur_loc == goal[color] or cur_loc in goal):
                i_gate.eligibility_trace_update(i_input)
                o_gate.eligibility_trace_update(o_input)
                agent.eligibility_trace_update(a_input)
                
                # if you get to a goal and it's 0, dont apply the delta
                i_gate.td_update_goal(r,i_value)
                o_gate.td_update_goal(r,o_value)
                agent.td_update_goal(r,a_value)
                #print('goal!!!')
                o_gate_correct_shared_local = 0
                shared_total_local = 0
                break
                
            p_a_input = agent_input
            p_i_input = i_gate_input
            p_o_input = o_gate_input
            p_a_action = action
            p_i_action = i_gate_state
            p_o_action = o_gate_state
            p_action_restrict = action_restrict
            p_igate_restrict = igate_restrict
            p_ogate_restrict = ogate_restrict
        
            p_loc = cur_loc
            p_i_value = i_value # Q val for input gate
            p_i_max = i_max
            p_o_value = o_value # Q val for output gate
            p_o_max = o_max
            p_a_value = a_value # Q val for agent
            p_a_max = a_max
            
            # update eligibility traces
            i_gate.eligibility_trace_update(i_input)
            o_gate.eligibility_trace_update(o_input)
            agent.eligibility_trace_update(a_input)
            
            # move agent
            state,cur_loc = s_mac.move(p_loc,action)
            row,col = cur_loc[0],cur_loc[1]
            color_signal = nothing # turn off cue
            
            role_i = 0 if color_signal!=nothing else 1
            igate_restrict = [0] # restricted set 
            ogate_restrict = [0,1] # restricted set
            #ogate_restrict = [1] if episode < handhold else [0,1] # hand holding
            '''
            if episode < handhold: # hand holding
                ogate_restrict = [0] if row >= 2 else [1]
            else:
                ogate_restrict = [0,1]
            '''
            #
            #print("cue:",colors[color_signal])
            #print("role_i:",roles[role_i])
            i_gate_input = roles[role_i]+'*'+state+'*'+t1
            LTM_obj.encode(i_gate_input)
            igate_restrict = [0] if color_signal == nothing else [0,1] # gate can only open if something is present
            i_gate_state,i_value,i_max,i_input,i_values = i_gate.action(i_gate_input,gate_actions,igate_restrict)
            #print("i_gate:",i_gate_state)
            WM.wm_in_flow(i_gate_state,slot,color_signal) # control flow of wm contents
            #print("wm_in:",WM.get_one_wm_maint(slot))
            role_o = 1 if WM.wm_maint_slot_is_empty(slot) else 0
            o_gate_input = roles[role_o]+'*'+state+'*'+t1
            LTM_obj.encode(o_gate_input)
            ogate_restrict = [0] if WM.wm_maint_slot_is_empty(slot) else [0,1] # gate can only open if memory is present in slot
            o_gate_state,o_value,o_max,o_input,o_values = o_gate.action(o_gate_input,gate_actions,ogate_restrict)
            #print("role_o:",roles[role_o])
            #print("o_gate:",o_gate_state)
            WM.wm_out_flow(o_gate_state,slot) # wm out contents for given slots
            wm_out = WM.get_one_wm_output(slot)
            #print("wm_out:",wm_out)
            #print("\n\n")
            '''
            if o_gate_state == Closed and not WM.wm_maint_slot_is_empty(slot):
                agent_input = gateclosed+'*'+colors[color_signal]+'*'+state+'*'+wm_out+'*'+t1
                LTM_obj.encode(agent_input)
            else:
                agent_input = colors[color_signal]+'*'+state+'*'+wm_out+'*'+t1
                LTM_obj.encode(agent_input)
                WM.update_wm_maint(slot,nothing)
            '''
            
            # context for it agent is remembering initial color que or not
            
            if o_gate_state == Closed and never_opened_flag:
                if not WM.wm_maint_slot_is_empty(slot):
                    ATR = 'remember'
                else:
                    ATR = 'not_remember'
            
            if o_gate_state == Open:
                #pass
                WM.update_wm_maint(slot,nothing)
                memory_states.append(state)
                if test_flag:
                    memory_states_testing.append(state)
                memory = state
                ATR = 'used'
                never_opened_flag = False
            
            agent_input = colors[color_signal]+'*'+state+'*'+wm_out+'*'+ATR
            LTM_obj.encode(agent_input)
            action_restrict = (state_table[row][col])["action_restrict"]# get restricted action set
            action,a_value,a_max,a_input,a_values = agent.action(agent_input,agent_actions,action_restrict) # agent action
            
            if row >= 2:
                o_gate_correct_shared += 1 if o_gate_state == Closed else 0
                o_gate_correct_shared_local += 1 if o_gate_state == Closed else 0
                shared_total_local += 1
                shared_total += 1
            else:
                o_gate_correct_non += 1 if o_gate_state == Open else 0
                non_total += 1
            '''
            error1 = agent.td_update(r,a_value,p_a_value)
            error2 = o_gate.td_update(r,a_value,p_o_value)
            error3 = i_gate.td_update(r,a_value,p_i_value)
            '''
            '''
            if error1 < -.5 or error2 < -.5 or error3 < -.5:
                if error1 < -.5:
                    #print('agent error triggered')
                    agent.set_eligibility_zero()
                    #agent.eligibility_trace_update(a_input)
                    agent.td_update_goal(r,a_value)
                if error2 < -.5:
                    #print('ogate error triggered')
                    o_gate.set_eligibility_zero()
                    #o_gate.eligibility_trace_update(a_input)
                    o_gate.td_update_goal(r,o_value)
                if error3 < -5:
                    #print('igate error triggered')
                    i_gate.set_eligibility_zero()
                    #i_gate.eligibility_trace_update(a_input)
                    i_gate.td_update_goal(r,i_value)
                
                break
            '''
            
            error1 = agent.td_update(r,a_max,p_a_value)
            error2 = o_gate.td_update(r,a_max,p_o_value)
            error3 = i_gate.td_update(r,a_max,p_i_value)
            
            WM.flush_all_wm_output()
            count += 1
        if test_flag:
            if color != nothing: 
                diff_save.append(abs(step-optimal_steps))
                if colors[color] == 'red':
                    diff_save_red.append(abs(step-optimal_steps))
                elif colors[color] == 'green':
                    diff_save_green.append(abs(step-optimal_steps))
                elif colors[color] == 'purple':
                    diff_save_purple.append(abs(step-optimal_steps))
            
        if color != nothing:    
            step_diff = abs(step-optimal_steps)
            #print(step_diff)
            opt_array.append(step_diff)
            if step_diff == 0 and cur_loc == goal[color]:
                acc_array.append(1)
            else:
                acc_array.append(0)
                
            if colors[color] == 'red':
                diff_red.append(abs(step-optimal_steps))
            elif colors[color] == 'green':
                diff_green.append(abs(step-optimal_steps))
            elif colors[color] == 'purple':
                diff_purple.append(abs(step-optimal_steps))
             
        if (episode)%stat_window == 0:
            # print performance
            agent.epsilon *= 0.99
            i_gate.epsilon *= 0.99
            o_gate.epsilon *= 0.99
            print('episode:',episode)
            mean_diff = sum(opt_array)/len(opt_array)
            diff_array.append(mean_diff)
            mean_acc = sum(acc_array)/len(acc_array)
            accuracy.append(mean_acc)
            
            mean_diff_red = sum(diff_red)/len(diff_red) if len(diff_red)>0 else 'N/A'
            mean_diff_green = sum(diff_green)/len(diff_green) if len(diff_green)>0 else 'N/A'
            mean_diff_purple = sum(diff_purple)/len(diff_purple) if len(diff_purple)>0 else 'N/A'
            #plt.plot(diff_array)
            #plt.show()
            #plt.plot(accuracy)
            #plt.show()
            opt_array = []
            acc_array = []
            diff_red,diff_green,diff_purple = [],[],[]
            #print("agent_value:",a_value)
            #print("input_value:",i_value)
            #print("output_value:",o_value)
            #print('episode:',episode)
            print("mean diff:",mean_diff)
            print("mean accuracy:",mean_acc)
            igate_acc = sum(i_gate_init_correct)/len(i_gate_init_correct)
            print("mean igate:",igate_acc)
            print("mean diff red",mean_diff_red)
            print("mean diff green",mean_diff_green)
            print("mean diff purple",mean_diff_purple)
            i_gate_init_correct = []
            ###################
            if shared_total > 0:
                shared = o_gate_correct_shared/shared_total
                #print("mean o_gate_shared:",o_gate_correct_shared/shared_total)
            if non_total > 0:
                pass
                #print("mean o_gate_non:",o_gate_correct_non/non_total)
            if len(memory_states) > 0:
                correct_mem_use = (memory_states.count('S12'))/len(memory_states)
            #print('correct memory_state use (S12):',correct_mem_use)
            #print('S10 memory:',(memory_states.count('S10'))/len(memory_states))
            #print('S11 memory:',(memory_states.count('S11'))/len(memory_states))
            #print('correct memory_state use (S12):',correct_mem_use)
            #print('S13 memory:',(memory_states.count('S13'))/len(memory_states))
            #print('S14 memory:',(memory_states.count('S14'))/len(memory_states))
                for x in range(nrows):
                    for y in range(ncols):
                        mem = 'S'+str(x)+str(y)
                        print(mem,'memory:',(memory_states.count(mem))/len(memory_states))
            memory_states = []
            print()
            o_gate_correct_shared = 0
            o_gate_correct_non = 0
            shared_total = 0
            non_total = 0
            o_gate_total = 0
        episode += 1
    print(mean_acc,mean_diff,sep=',')
    
    #WA,fname1 = agent.get_weights(), "IO_Model_agent_weights.txt" # agent weights
    #WO,fname2 = o_gate.get_weights(), "IO_Model_output_weights.txt"# output gate weights
    #WI,fname3 = i_gate.get_weights(), "IO_Model_input_weights.txt"# input gate weights
    #store_weights(fname1,WA) # store agent weights
    #store_weights(fname2,WO) # store output gate weights
    #store_weights(fname3,WI) # store input gate weights
    #fname = 'step_diff_RGP.dat'
    #fname2 = 'step_diff_red_RGP.dat'
    #fname3 = 'step_diff_green_RGP.dat'
    #fname4 = 'step_diff_purple_RGP.dat'
    #fname5 = 'memory_states_RGP.dat'
    #store_weights(fname,diff_save)
    #store_weights(fname2,diff_save_red)
    #store_weights(fname3,diff_save_green)
    #store_weights(fname4,diff_save_purple)
    #store_weights(fname5,memory_states_testing)
    
    endTime = time.time()
    #print(endTime-beginTime,"seconds wall time")
    return LTM_obj, agent, i_gate, o_gate, WM


# In[13]:

LTM_obj, agent, i_gate, o_gate, WM = IO_gate_model([5,5],100000,1000,100000)
WM_Objs = [LTM_obj, agent, i_gate, o_gate, WM]

# Transfer Task RedGreen
LTM_obj2, agent2, i_gate2, o_gate2, WM2 = Transfer_Task(WM_Objs,[5,5],100000,1000,100000,'RG')
# In[ ]:



