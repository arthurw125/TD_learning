import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import plotly
import numpy as np
import random
import hrr2 as hrr
import math
import state_machine_class as state_machine
import Gate_Class_v4 as gate
from plotly.graph_objs import Scatter, Layout, Surface
plotly.offline.init_notebook_mode(connected=True)

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
    
    if cue == 0: # red
        return red[row][col]
    elif cue == 1:
        return green[row][col]
    elif cue == 2:
        return purple[row][col]

def IO_gate_model(state_size,nepisodes,stat_window):
    n = 256
    #n = 1024
    nagent_actions = 4
    gate_actions = 2
    ncolors = 2
    nslots = 1
    nroles = 1
    nrolestates = 2
    Closed,Open = 0,1
    agent_actions = ['up','down','left','right']
    colors = ['red','green',''] # external colors
    nothing = len(colors)-1
    #wm_colors = ['wm_red','wm_green',''] # internal colors
    roles = ['role_avail','role_unavail']
    gate_actions = ['closed','open']
    gateclosed = 'gateclosed'
    gateopen = 'gateopen'
    
    nrows,ncols = state_size
    state_table = state_machine.state_machine.generate_state_table_bottle(nrows,ncols)
    
    LTM_obj = hrr.LTM(n,True)
    #agent = gate.Q_learning(n,LTM_obj,epsilon=0.05,bias=1)
    agent = gate.Q_learning(n,LTM_obj,epsilon=0.01,bias=1)
    i_gate = gate.Q_learning(n,LTM_obj,epsilon=0.01,bias=1)
    o_gate = gate.Q_learning(n,LTM_obj,epsilon=0.01,bias=1)
    WM = gate.wm_content(colors,nslots,LTM_obj)
    s_mac = state_machine.state_machine(state_table)
    nsteps = 100
    goal = [ [0,0],[0,ncols-1] ]
    value_function = np.zeros((nrows,ncols)) # store Q values for statitics
     # track performance
    opt_array = []
    diff_array = []
    t1 = ""
    
    handhold = 400000
    anneal = 490000
    
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
    shared = 0
    for episode in range(nepisodes):
        
        count = 0
        #t1 = str(count)
        t1 = "t1"
        row = random.randrange(2,nrows) # get random row for state (in shared area)
        col = random.randrange(0,ncols) # get random col for state
        cur_loc = [row,col] # current location of agent
        color_signal = random.randrange(0,ncolors) # get random color
        color = color_signal # used for reward because color signal may change
        #r = reward(cur_loc,colors[color],state_size) if color!=nothing else 0
        
        if episode == anneal:
            agent.epsilon,i_gate.epsilon,o_gate.epsilon = 0,0,0
        #optimal_steps = optimal_path_length(cur_loc,color_signal) # get optimal steps to goal 
        optimal_steps = optimal_path_length_bottle(cur_loc,color_signal) # get optimal steps to goal 
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
        
        WM.flush_all_wm_maint()
        #print("cue:",colors[color_signal])
        #print("role_i:",roles[role_i])
        i_gate_input = roles[role_i]+'*'+state+'*'+t1
        LTM_obj.encode(i_gate_input)
        i_gate_state,i_value,i_max,i_input,i_values = i_gate.action(i_gate_input,gate_actions,igate_restrict)
        WM.wm_in_flow(i_gate_state,slot,color_signal) # control flow of wm contents
        #print("i_gate:",i_gate_state)
        #print("wm_in:",WM.get_one_wm_maint(slot))
        role_o = 1 if WM.wm_maint_slot_is_empty(slot) else 0
        #print("role_o:",roles[role_o])
        o_gate_input = roles[role_o]+'*'+state+'*'+t1
        LTM_obj.encode(o_gate_input)
        o_gate_state,o_value,o_max,o_input,o_values = o_gate.action(o_gate_input,gate_actions,ogate_restrict)
        WM.wm_out_flow(o_gate_state,slot) # wm out contents for given slots
        wm_out = WM.get_one_wm_output(slot)
        #print("o_gate:",o_gate_state)
        #print("wm_out:",WM.get_one_wm_output(slot))
        #print("\n\n")
        
        if o_gate_state == Closed:
            agent_input = gateclosed+'*'+colors[color_signal]+'*'+state+'*'+wm_out+'*'+t1
            LTM_obj.encode(agent_input)
        else:
            agent_input = colors[color_signal]+'*'+state+'*'+wm_out+'*'+t1
            LTM_obj.encode(agent_input)
        
        #agent_input = colors[color_signal]+'*'+state+'*'+wm_out+'*'+t1
        action_restrict = (state_table[row][col])["action_restrict"]# get restricted action set
        action,a_value,a_max,a_input,a_values = agent.action(agent_input,agent_actions,action_restrict) # agent action
        #print("agent_value: ",a_value)
        #print("input_value: ",i_value)
        #print("output_value:",o_value)
        
        o_gate_correct_shared += 1 if o_gate_state == Closed else 0
        shared_total += 1
        
        i_gate.set_eligibility_zero()
        o_gate.set_eligibility_zero()
        agent.set_eligibility_zero() # set eligibility trace to zero
        WM.flush_all_wm_output()
        count += 1
        for step in range(nsteps):
            
            #t1 = str(count)
            t1 = ''
            r = reward(cur_loc,colors[color],state_size) if color!=nothing else 0
            #r = punishment(cur_loc,colors[color],state_size) if color!=nothing else -1
            #print(state)
            if color!=nothing and cur_loc == goal[color]:
                i_gate.eligibility_trace_update(i_input)
                o_gate.eligibility_trace_update(o_input)
                agent.eligibility_trace_update(a_input)
                i_gate.td_update_goal(r,i_value)
                o_gate.td_update_goal(r,o_value)
                agent.td_update_goal(r,a_value)
                #print('goal!!!')
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
            color_signal = 2 # turn off cue
            
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
            i_gate_state,i_value,i_max,i_input,i_values = i_gate.action(i_gate_input,gate_actions,igate_restrict)
            #print("i_gate:",i_gate_state)
            WM.wm_in_flow(i_gate_state,slot,color_signal) # control flow of wm contents
            #print("wm_in:",WM.get_one_wm_maint(slot))
            role_o = 1 if WM.wm_maint_slot_is_empty(slot) else 0
            o_gate_input = roles[role_o]+'*'+state+'*'+t1
            LTM_obj.encode(o_gate_input)
            o_gate_state,o_value,o_max,o_input,o_values = o_gate.action(o_gate_input,gate_actions,ogate_restrict)
            #print("role_o:",roles[role_o])
            #print("o_gate:",o_gate_state)
            WM.wm_out_flow(o_gate_state,slot) # wm out contents for given slots
            wm_out = WM.get_one_wm_output(slot)
            #print("wm_out:",wm_out)
            #print("\n\n")
            
            if o_gate_state == Closed:
                agent_input = gateclosed+'*'+colors[color_signal]+'*'+state+'*'+wm_out+'*'+t1
                LTM_obj.encode(agent_input)
            else:
                agent_input = colors[color_signal]+'*'+state+'*'+wm_out+'*'+t1
                LTM_obj.encode(agent_input)
            
            #agent_input = colors[color_signal]+'*'+state+'*'+wm_out
            action_restrict = (state_table[row][col])["action_restrict"]# get restricted action set
            action,a_value,a_max,a_input,a_values = agent.action(agent_input,agent_actions,action_restrict) # agent action
            # ----------------------------------
            # Bias gates to stay in a state for long periods of time
            
            a_beta = 0
            '''
            if a_value > 0:
                if p_a_action == action:
                    a_beta = 0
                else:
                    a_beta = 1
            else:
                if p_a_action == action:
                    a_beta = 1
                else:
                    a_beta = 0
            '''
            o_beta = 0
            '''
            if a_value > 0:
                if p_o_action == o_gate_state:
                    o_beta = 0
                else:
                    o_beta = 1
            else:
                if p_o_action == o_gate_state:
                    o_beta = 1
                else:
                    o_beta = 0
            '''       
            i_beta = 0
            '''
            if a_value > 0:
                if p_i_action == i_gate_state:
                    i_beta = 0
                else:
                    i_beta = 1
            else:
                if p_i_action == i_gate_state:
                    i_beta = 1
                else:
                    i_beta = 0
            '''
            # -------------------------------------
            if row >= 2:
                o_gate_correct_shared += 1 if o_gate_state == Closed else 0
                shared_total += 1
            else:
                o_gate_correct_non += 1 if o_gate_state == Open else 0
                non_total += 1
            
            
            '''
            agent.td_update(r,a_value,p_a_value)
            o_gate.td_update(r,a_value,p_o_value)
            i_gate.td_update(r,a_value,p_i_value)
            '''
            
            agent.td_update_transfer(r,a_value,p_a_value,a_beta)
            o_gate.td_update_transfer(r,a_value,p_o_value,o_beta)
            i_gate.td_update_transfer(r,a_value,p_i_value,i_beta)
            
            
            WM.flush_all_wm_output()
            count += 1
        if color != nothing:    
            step_diff = abs(step-optimal_steps)
            #print(step_diff)
            opt_array.append(step_diff)
            if step_diff == 0:
                acc_array.append(1)
            else:
                acc_array.append(0)
            
        if (episode+1)%stat_window == 0:
            # print performance
            mean_diff = sum(opt_array)/len(opt_array)
            diff_array.append(mean_diff)
            mean_acc = sum(acc_array)/len(acc_array)
            accuracy.append(mean_acc)
            #plt.plot(diff_array)
            #plt.show()
            #plt.plot(accuracy)
            #plt.show()
            opt_array = []
            acc_array = []
            #print("agent_value:",a_value)
            #print("input_value:",i_value)
            #print("output_value:",o_value)
            #print("mean diff:",mean_diff)
            #print("mean accuracy:",mean_acc)
        
            ###################
            if shared_total > 0:
                shared = o_gate_correct_shared/shared_total
                #print("mean o_gate_shared:",o_gate_correct_shared/shared_total)
                
            if non_total > 0:
                pass
                #print("mean o_gate_non:",o_gate_correct_non/non_total)
            o_gate_correct_shared = 0
            o_gate_correct_non = 0
            shared_total = 0
            non_total = 0
            o_gate_total = 0
    print(shared)
    #WA,fname1 = agent.get_weights(), "IO_Model_agent_weights.txt" # agent weights
    #WO,fname2 = o_gate.get_weights(), "IO_Model_output_weights.txt"# output gate weights
    #WI,fname3 = i_gate.get_weights(), "IO_Model_input_weights.txt"# input gate weights
    #store_weights(fname1,WA) # store agent weights
    #store_weights(fname2,WO) # store output gate weights
    #store_weights(fname3,WI) # store input gate weights
    return LTM_obj, agent, i_gate, o_gate, WM

LTM_obj2, agent2, i_gate2, o_gate2, WM2 = IO_gate_model([5,5],500000,1000)