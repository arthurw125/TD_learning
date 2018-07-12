#from plotly.graph_objs import Scatter, Layout
import matplotlib.pyplot as plt
import plotly
import numpy as np
import random
import hrr
import math
from plotly.graph_objs import Scatter, Layout, Surface
plotly.offline.init_notebook_mode(connected=True)


def softmax(arr,t=1.0):
    w = np.array(arr)
    e = np.exp(w / t)
    dist = e / np.sum(e)
    return dist

def identity_vector(n):
    hrr_i = np.zeros(n)
    hrr_i[0] = 1
    return hrr_i

def optimal_path(init_state,goal,nstates):
    #left = (num_states-goal) + init_state
    #right = (num_states+goal) - init_state
    #opt_steps = min(left,right)
    if init_state < goal:
        left = init_state + abs(goal - nstates)
        right = abs(goal - init_state)
    elif init_state == goal:
        left,right = 0,0
    else:
        left = abs(goal - init_state)
        right = abs(init_state - nstates) + goal
    opt_steps = min(left,right)
    #print(init_state,goal,nstates)
    return opt_steps


# Reinforcement Learning class
class RL_Obj:
    
    def __init__(self,n,nactions,bias=1,lrate=0.1,gamma=0.9,td_lambda=0.9,epsilon=0.01):
        self.n = n # vector length
        self.nactions = nactions
        self.W = hrr.hrr(n)
        self.actions = hrr.hrrs(n,nactions)
        self.eligibility = hrr.hrr(n)
        self.bias = bias
        self.lrate = lrate
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.epsilon = epsilon
        
    def action(self,state_space):
        mystate = hrr.convolve(state_space,self.actions)
        values = np.dot(mystate,self.W) + self.bias
        #print(values)
        sm = softmax(values)
        action = np.argmax(sm)
        if random.random() < self.epsilon:
            action = random.randrange(0,self.nactions)
        # force gate to be closed
        #if close:
        #    action = 0
        #x = hrr.convolve(mystate,self.actions[action]) # input
        x = mystate[action] # input
        #return action, values[action], my_state2[action]
        return action, values[action], x
   
    def eligibility_trace_update(self,my_input):
        #s_a = hrr.convolve(state_vec,action_vec)
        self.eligibility = my_input + self.td_lambda*self.eligibility
    
    def set_eligibility_zero(self):
        self.eligibility = np.zeros(self.n)
    
    def td_update(self,r,value,pvalue):
        error = (r+self.gamma*value) - pvalue
        self.W += self.lrate*error*self.eligibility
        
    def td_update_goal(self,r,value):
        error = r - value
        self.W += self.lrate*error*self.eligibility

class Gate(RL_Obj):
    
    def action(self,state_space,forced_gate_state):
        mystate = hrr.convolve(state_space,self.actions)
        values = np.dot(mystate,self.W) + self.bias
        #print(values)
        sm = softmax(values)
        action = np.argmax(sm)
        if random.random() < self.epsilon:
            action = random.randrange(0,self.nactions)
        # force gate to be closed or open
        if forced_gate_state == 'Open':
            action = 1
        elif forced_gate_state == 'Closed':
            action = 0
        #x = hrr.convolve(mystate,self.actions[action]) # input
        x = mystate[action] # input
        #return action, values[action], my_state2[action]
        return action, values[action], x

class wm_content:
    
    def __init__(self,n,nitems,n_wm_slots):
        self.n = n
        self.nitems = nitems
        self.n_wm_slots = n_wm_slots
        self.wm_maint = [identity_vector(n)]*n_wm_slots # init wm maint slots
        self.wm_output = [identity_vector(n)]*n_wm_slots # init wm output slots
        self.wm_maint_statistics = [-1]*n_wm_slots # view of what's in wm_maint
        self.wm_output_statistics = [-1]*n_wm_slots # view of what's in wm_output
        self.wm_items = hrr.hrrs(n,nitems) # encode available internal representations
        self.wm_items = np.row_stack((self.wm_items,identity_vector(n) ))
        #self.wm_maint_states = 
        #self.wm_maint_num = len(self.wm_slots)-1
        #self.wm_output_num = len(self.wm_slots)-1
    '''
    def update_wm_maint(self,gate_state,cue):
        if gate_state:
            self.wm_maint[0] = self.wm_slots[cue]
            # tracks wm maint level for empty content
            self.wm_maint_num = cue
            
            if cue == len(self.wm_slots)-1:
                #print(len(self.wm_slots))
                wm_maint_state = 1 # empty
            else:
                wm_maint_state = 0 # has role
    '''
    def update_wm_maint(self,slot_num,item_num):
        self.wm_maint[slot_num] = self.wm_items[item_num]
        self.wm_maint_statistics[slot_num] = item_num # updates stats for wm
    
    '''
    def update_wm_output(self,gate_state):
        if gate_state:
            self.wm_output[0] = self.wm_slots[self.wm_maint_num]
            self.wm_output_num = self.wm_maint_num
    '''
    def update_wm_output(self,slot_num,item_num):
        self.wm_output[slot_num] = self.wm_items[item_num]
        self.wm_output_statistics[slot_num] = item_num
        
    def get_all_wm_maint(self):
        return self.wm_maint # returns matrix of wm_maint contents
    
    def get_all_wm_output(self):
        return self.wm_output # returns matrix of wm_output contents
    
    def get_one_wm_maint(self,slot_num):
        return self.wm_maint[slot_num] # returns vector
    
    def get_one_wm_output(self,slot_num):
        return self.wm_output[slot_num] # returns vector
    
    def flush_all_wm_maint(self):
        self.wm_maint = [identity_vector(self.n)]*self.n_wm_slots
        
    def flush_all_wm_output(self):
        self.wm_output = [identity_vector(self.n)]*self.n_wm_slots
        
    def get_wm_maint_statistics(self):
        return self.wm_maint_statistics
    
    def get_wm_output_statistics(self):
        return self.wm_output_statistics
    
    def wm_maint_slot_is_empty(self,slot_num):
        return np.array_equal(identity_vector(self.n),self.wm_maint[slot_num])
    
    def wm_output_slot_is_empty(self,slot_num):
        return np.array_equal(identity_vector(self.n),self.wm_output[slot_num])
    
    # controls the flow of wm contents from wm_maint layer to wm_output layer
    
    def wm_in_flow(self,i_gate_state,slot_num,item_num):
        if i_gate_state:
            self.update_wm_maint(slot_num,item_num)
            
    def wm_out_flow(self,o_gate_state,slot_num):
        if o_gate_state:
            item_num = self.get_wm_maint_statistics()[slot_num]
            self.update_wm_output(slot_num,item_num)
    


class working_memory_system:
    
    def __init__(self,input_gate_obj,output_gate_obj,agent_obj,wm_content_obj):
        self.input_gate = input_gate_obj
        self.output_gate = output_gate_obj
        self.agent = agent_obj
        self.wm_contents = wm_contents_obj

def flow_control_test(nstates,nepisodes):
    n = 8
    agent_actions = 2
    gate_actions = 2
    ncolors = 2
    nslots = 1
    nroles = 1
    
    states = hrr.hrrs(n,nstates) 
    colors = hrr.hrrs(n,ncolors) # external cue
    colors = np.row_stack((colors,identity_vector(n) ))
    roles = hrr.hrrs(n,nroles)
    roles = np.row_stack((roles,identity_vector(n) ))
    
    # preconvolve states
    role_state = hrr.oconvolve(roles,states)
    role_state = np.reshape(role_state,(nroles+1,nstates,n))
    cue_state = hrr.oconvolve(colors,states)
    cue_state = np.reshape(cue_state,(ncolors+1,nstates,n))
    #print(role_state.shape)
    #print(cue_state.shape)
    # create objects
    agent = RL_Obj(n,agent_actions)
    i_gate = RL_Obj(n,gate_actions)
    o_gate = RL_Obj(n,gate_actions)
    WM = wm_content(n,ncolors,nslots)
    
    for episode in range(nepisodes):
        state = random.randrange(0,nstates)
        color_signal = random.randrange(0,ncolors)
        
        role_i = 0 # role is available
        slot = 0 # slot number in use
        i_gate_input = role_state[role_i,state,:] # input for in_gate
        i_gate_state,i_value,i_input = i_gate.action(i_gate_input)
        print('color_cue:',color_signal)
        print('i_gate_state:',i_gate_state)
        WM.wm_in_flow(i_gate_state,slot,color_signal) # control flow of wm maint contents
        print('wm_maint:',WM.get_wm_maint_statistics()[slot])
        print('wm_maint_contents:',WM.get_one_wm_maint(slot))
        role_o = 1 if WM.wm_maint_slot_is_empty(slot) else 0
        print('role_o:',role_o)
        o_gate_input = role_state[role_o,state,:]   # input for out_gate
        o_gate_state,o_value,o_input = i_gate.action(o_gate_input)
        print('o_gate_state:',o_gate_state)
        WM.wm_out_flow(o_gate_state,slot) # control flow of wm out contents
        print('wm_output:',WM.get_wm_output_statistics()[slot])
        wm_out = WM.get_one_wm_output(slot) # wm out contents for given slot
        print(wm_out)
        agent_input = hrr.convolve(cue_state[color_signal,state],wm_out)
        action,a_value,a_input = agent.action(agent_input)
        print('action:',action)
        print()
        print()
        

def color_maze_task(nstates,nepisodes,stat_window):
    n = 128
    agent_actions = 2
    gate_actions = 2
    ncolors = 2
    nslots = 1
    nroles = 1
    
    # goals and rewards
    goal = [0,nstates//2,None]
    
    reward = np.zeros((ncolors+1,nstates))
    for x in range(ncolors):
        reward[x,goal[x]] = 1
    
    #####
    # punishment based reward
    '''
    reward = np.ones((ncolors+1,nstates))
    reward *= -1
    for x in range(ncolors):
        reward[x,goal[x]] = 0
    '''
    
    states = hrr.hrrs(n,nstates) 
    colors = hrr.hrrs(n,ncolors) # external cue
    colors = np.row_stack((colors,identity_vector(n) ))
    roles = hrr.hrrs(n,nroles)
    roles = np.row_stack((roles,identity_vector(n) ))
    
    # preconvolve states
    role_state = hrr.oconvolve(roles,states)
    role_state = np.reshape(role_state,(nroles+1,nstates,n))
    cue_state = hrr.oconvolve(colors,states)
    cue_state = np.reshape(cue_state,(ncolors+1,nstates,n))
    
    agent = RL_Obj(n,agent_actions)
    i_gate = Gate(n,gate_actions)
    o_gate = Gate(n,gate_actions)
    WM = wm_content(n,ncolors,nslots)
    
    nsteps = 100
    opt_array = []
    diff_sum = 0
    mycount = 0
    for episode in range(nepisodes):
        print('episode:',episode)
        mycount += 1
        state = random.randrange(0,nstates)
        color_signal = random.randrange(0,ncolors)
        color = color_signal
        optimal_steps = optimal_path(state,goal[color_signal],nstates) # tracks number of optimal steps
        role_i = 0 # role is available
        slot = 0 # slot number in use
        forced_igate_state = None
        forced_ogate_state = None
        WM.flush_all_wm_maint()
        i_gate_input = role_state[role_i,state,:] # input for in_gate
        i_gate_state,i_value,i_input = i_gate.action(i_gate_input,forced_igate_state)
        
        WM.wm_in_flow(i_gate_state,slot,color_signal) # control flow of wm maint contents
        
        role_o = 1 if WM.wm_maint_slot_is_empty(slot) else 0
        o_gate_input = role_state[role_o,state,:]   # input for out_gate
        o_gate_state,o_value,o_input = o_gate.action(o_gate_input,forced_ogate_state)
        
        WM.wm_out_flow(o_gate_state,slot) # control flow of wm out contents

        wm_out = WM.get_one_wm_output(slot) # wm out contents for given slot
        agent_input = hrr.convolve(cue_state[color_signal,state],wm_out)
        action,a_value,a_input = agent.action(agent_input)
        
        i_gate.set_eligibility_zero()
        o_gate.set_eligibility_zero()
        agent.set_eligibility_zero()
        
        # clear wm output
        WM.flush_all_wm_output()
        
        for step in range(nsteps):
            r = reward[color,state]
            
            if state == goal[color]:
                i_gate.eligibility_trace_update(i_input)
                o_gate.eligibility_trace_update(o_input)
                agent.eligibility_trace_update(a_input)
                
                i_gate.td_update_goal(r,i_value)
                o_gate.td_update_goal(r,o_value)
                agent.td_update_goal(r,a_value)
                break
            
            pstate = state # maze state
            p_i_value = i_value # Q val for input gate
            p_o_value = o_value # Q val for output gate
            p_a_value = a_value # Q val for agent
            
            # update eligibility traces
            i_gate.eligibility_trace_update(i_input)
            o_gate.eligibility_trace_update(o_input)
            agent.eligibility_trace_update(a_input)
            
            # change state in maze by taking action
            state = ((state+np.array([-1,1]))%nstates)[action]
            
            # turn off cue
            color_signal = 2
            role_i = 1 # role is unavailable
            forced_igate_state = 'Closed'
            forced_ogate_state = 'Open'
            
            i_gate_input = role_state[role_i,state,:] # input for in_gate
            i_gate_state,i_value,i_input = i_gate.action(i_gate_input,forced_igate_state)
            # 
            WM.wm_in_flow(i_gate_state,slot,color_signal) # control flow of wm maint contents

            role_o = 1 if WM.wm_maint_slot_is_empty(slot) else 0 # checks if role is available in wm_maint
            o_gate_input = role_state[role_o,state,:]   # input for out_gate
            o_gate_state,o_value,o_input = o_gate.action(o_gate_input,forced_ogate_state)

            WM.wm_out_flow(o_gate_state,slot) # control flow of wm out contents

            wm_out = WM.get_one_wm_output(slot) # wm out contents for given slot
            agent_input = hrr.convolve(cue_state[color_signal,state],wm_out)
            action,a_value,a_input = agent.action(agent_input)
            
            # td update
            i_gate.td_update(r,i_value,p_i_value)
            o_gate.td_update(r,o_value,p_o_value)
            agent.td_update(r,a_value,p_a_value)
            
            # clear wm output
            WM.flush_all_wm_output()
            
        step_diff = abs(step - optimal_steps)
        diff_sum += step_diff
        if episode%stat_window==0:
            mean_diff = diff_sum/mycount
            opt_array.append(mean_diff)
            mycount = 0
            diff_sum = 0
            # optimal steps
            plotly.offline.plot({
            "data": [Scatter(x=[x for x in range(len(opt_array))], y=opt_array)]
            })

def testing_maze(nstates,nepisodes,stat_window):
    n = 128
    nactions = 2
    goal = 0
    reward = np.zeros(nstates)
    reward[goal] = 1
    states = hrr.hrrs(n,nstates)
    
    agent = RL_Obj(n,nactions)
    nsteps = 100
    opt_array = []
    diff_sum = 0
    mycount = 0
    for episode in range(nepisodes):
        mycount += 1
        print('episode:',episode)
        state = random.randrange(0,nstates)
        #print('state:',state)
        action,value,my_input = agent.action(states[state])
        agent.set_eligibility_zero()
        optimal_steps = optimal_path(state,goal,nstates)
        #print('optimal steps',optimal_steps)
        for step in range(nsteps):
            r =  reward[state]
            if state == goal:
                agent.eligibility_trace_update(my_input)
                agent.td_update_goal(r,value)
                break
                
            pstate = state
            pvalue = value
            #paction = action
            agent.eligibility_trace_update(my_input)
            state = ((state+np.array([-1,1]))%nstates)[action]
            action,value,my_input = agent.action(states[state])
            agent.td_update(r,value,pvalue)
            
        step_diff = abs(step - optimal_steps)
        print('step_dif:',step_diff)
        diff_sum += step_diff   
        if episode%stat_window==0:
            mean_diff = diff_sum/mycount
            opt_array.append(mean_diff)
            mycount = 0
            diff_sum = 0
            
            V1 = list(map(lambda x: np.dot(x,agent.W)+agent.bias, hrr.convolve(states,agent.actions[0]) ))
            V2 = list(map(lambda x: np.dot(x,agent.W)+agent.bias, hrr.convolve(states,agent.actions[1]) ))

            plotly.offline.plot({
            "data": [Scatter(x=[x for x in range(len(V1))] , y=V1,name='left'),
                    Scatter(x=[x for x in range(len(V2))] , y=V2,name='right')],
            "layout": Layout(title="",xaxis=dict(title="state"),yaxis=dict(title="Q(s,a)"))
            })
            
    plt.plot(opt_array)
    plt.show()
            
def maze_task(nstates,nepisodes,stat_window):
    n = 128
    nactions = 2 # left right
    ncolors = 2 # red/blue
    #ngates = 1
    ngate_states = 2 # open/close
    nmaint_states = 2 # empty/full
    nroles = 1 # number of roles
    
    #goal for red is at 0, green at middle
    goal = [0,nstates//2,None]
    reward = np.zeros((ncolors+1,nstates))
    #reward = np.ones((ncolors+1,nstates)) # punishment based
    #reward = reward*-10
    # reward matrix for each context
    
    for x in range(ncolors):
        reward[x,goal[x]] = 1
     
    # punishment based reward
    '''
    for x in range(ncolors):
        reward[x,goal[x]] = 0
    '''
    # basic actions are left and right
    states = hrr.hrrs(n,nstates)
    actions = hrr.hrrs(n,nactions)

    # identity vector
    hrr_i = np.zeros(n)
    hrr_i[0] = 1

    # external color
    external = hrr.hrrs(n,ncolors)
    external = np.row_stack((external,hrr_i))

    # Internal Representations
    wm_slots = hrr.hrrs(n,ncolors)
    wm_slots = np.row_stack((wm_slots,hrr_i))
    
    # working memory contents
    wm_maint = [hrr_i]
    wm_output = [hrr_i]
    
    # wm maint state (empty,full)
    #wm_maint_state = hrr.hrrs(n,nmaint_states)
    
    # Gate
    #gates = hrr.hrrs(n,ngates)
    roles = hrr.hrrs(n,nroles)
    roles = np.row_stack((roles,hrr_i))

    # Gate state (open/closed)
    gate_states = hrr.hrrs(n,ngate_states)

    # Weight vectors
    IGate_W = hrr.hrr(n) # Input gate
    OGate_W = hrr.hrr(n) # Output gate
    AGate_W = hrr.hrr(n) # Agent

    IBias,OBias,ABias = 1,1,1 # Bias for gates and agent
    eligibility = np.zeros(n)
    epsilon = 0.1
    nsteps = 100
        
    i_gate = input_gate(IGate_W,gate_states,eligibility)
    o_gate = output_gate(OGate_W,gate_states,eligibility)
    myagent = agent(AGate_W,actions,eligibility)
    wm_cont = wm_content(wm_maint,wm_output,wm_slots)
    counter = 0
    opt_array = []
    diff_sum = 0
    mycount = 0
    for episode in range(nepisodes):
        print('episode:',counter)
        counter += 1
        mycount += 1 # used for mean diff
        state = random.randrange(0,nstates)
        color_signal = random.randrange(0,ncolors)
        
        optimal_steps = optimal_path(state,goal[color_signal],nstates)
        #print(optimal_steps)
        # set external cue
        cue = color_signal
        color = cue
        #################
        close = False
        open = False
        wm,wm_num = wm_cont.get_wm_maint()
        i_gate_state,i_value,i_state = i_gate.gate_action(roles[0],states[state],close)
        wm_cont.update_wm_maint(i_gate_state,cue)
        wm_state = wm_cont.get_wm_maint_state()
        o_gate_state,o_value,o_state = o_gate.gate_action(roles[wm_state],states[state],open)
        wm_cont.update_wm_output(o_gate_state)
        wm_o,wm_out_num = wm_cont.get_wm_output()
        action,a_value,a_state = myagent.agent_action(external[cue],states[state],wm_o)
        #print(i_gate_state,o_gate_state)
        i_gate.set_eligibility_zero(n)
        o_gate.set_eligibility_zero(n)
        myagent.set_eligibility_zero(n)
        
        #testing purpose
        wm,wm_num = wm_cont.get_wm_maint()
        
        #print(wm_num,wm_out_num)
        # set output wm to identity vector
        wm_cont.set_wm_output(2)
        
        # decay epsilon
        if episode == 10000:
            i_gate.epsilon,o_gate.epsilon,myagent.epsilon = 0,0,0
            
        for step in range(nsteps):
            r = reward[color,state]
            
            if state == goal[color]:
                i_gate.eligibility_trace_update(i_state)
                o_gate.eligibility_trace_update(o_state)
                myagent.eligibility_trace_update(a_state)
                
                i_gate.td_update_goal(r,i_value)
                o_gate.td_update_goal(r,o_value)
                myagent.td_update_goal(r,a_value)
                #print('Made it to goal')
                break
            
            pstate = state # maze state
            p_i_value = i_value # Q val for input gate
            p_o_value = o_value # Q val for output gate
            p_a_value = a_value # Q val for agent
            
            # update eligibility traces
            i_gate.eligibility_trace_update(i_state)
            o_gate.eligibility_trace_update(o_state)
            myagent.eligibility_trace_update(a_state)
            
            # change state in maze by taking action
            state = ((state+np.array([-1,1]))%nstates)[action]
            
            # turn off cue
            cue = 2
            
            close = True
            open = False
            wm,wm_num = wm_cont.get_wm_maint()
            i_gate_state,i_value,i_state = i_gate.gate_action(roles[1],states[state],close)
            wm_cont.update_wm_maint(i_gate_state,cue)
            wm_state = wm_cont.get_wm_maint_state()
            o_gate_state,o_value,o_state = o_gate.gate_action(roles[wm_state],states[state],open)
            wm_cont.update_wm_output(o_gate_state)
            wm_o,wm_out_num = wm_cont.get_wm_output()
            action,a_value,a_state = myagent.agent_action(external[cue],states[state],wm_o)
            
            #testing 
            #print(wm_num,wm_out_num)
            # compute errors and update weights
            i_gate.td_update(r,i_value,p_i_value)
            o_gate.td_update(r,o_value,p_o_value)
            myagent.td_update(r,a_value,p_a_value)
            
            # set output wm to identity vector
            wm_cont.set_wm_output(2)
        #print('step:',step)
        # check for optimal steps being learned
        step_diff = abs(step - optimal_steps)
        diff_sum += step_diff
        if episode%stat_window==0:
            mean_diff = diff_sum/mycount
            opt_array.append(mean_diff)
            mycount = 0
            diff_sum = 0
            # iGate
            '''
            one = hrr.convolve(states[:],hrr.convolve(roles[0],gate_states[0]))
            two = hrr.convolve(states[:],hrr.convolve(roles[0],gate_states[1]))
            three = hrr.convolve(states[:],hrr.convolve(roles[1],gate_states[0]))
            four = hrr.convolve(states[:],hrr.convolve(roles[1],gate_states[1]))
            V1 = list(map(lambda x: np.dot(x,i_gate.W)+IBias, one))
            V2 = list(map(lambda x: np.dot(x,i_gate.W)+IBias, two))
            V3 = list(map(lambda x: np.dot(x,i_gate.W)+IBias, three))
            V4 = list(map(lambda x: np.dot(x,i_gate.W)+IBias, four))
            '''
            # oGate
            '''
            one = hrr.convolve(states[:],hrr.convolve(roles[0],gate_states[0]))
            two = hrr.convolve(states[:],hrr.convolve(roles[0],gate_states[1]))
            three = hrr.convolve(states[:],hrr.convolve(roles[1],gate_states[0]))
            four = hrr.convolve(states[:],hrr.convolve(roles[1],gate_states[1]))
            V1 = list(map(lambda x: np.dot(x,o_gate.W)+OBias, one))
            V2 = list(map(lambda x: np.dot(x,o_gate.W)+OBias, two))
            V3 = list(map(lambda x: np.dot(x,o_gate.W)+OBias, three))
            V4 = list(map(lambda x: np.dot(x,o_gate.W)+OBias, four))
            '''
            # agent
            '''
            one = hrr.convolve(states[:],hrr.convolve(wm_slots[0],actions[0]))
            two = hrr.convolve(states[:],hrr.convolve(wm_slots[0],actions[1]))
            three = hrr.convolve(states[:],hrr.convolve(wm_slots[1],actions[0]))
            four = hrr.convolve(states[:],hrr.convolve(wm_slots[1],actions[1]))
            V1 = list(map(lambda x: np.dot(x,myagent.W)+ABias, one))
            V2 = list(map(lambda x: np.dot(x,myagent.W)+ABias, two))
            V3 = list(map(lambda x: np.dot(x,myagent.W)+ABias, three))
            V4 = list(map(lambda x: np.dot(x,myagent.W)+ABias, four))
            #V5 = list(map(lambda x: np.dot(x,W)+bias, s_s_a_wm[0,2,:,0,:]))
            #V6 = list(map(lambda x: np.dot(x,W)+bias, s_s_a_wm[0,2,:,1,:]))
            #V7 = list(map(lambda x: np.dot(x,W)+bias, s_s_a_wm[1,2,:,0,:]))
            #V8 = list(map(lambda x: np.dot(x,W)+bias, s_s_a_wm[1,2,:,1,:]))
            '''
            #plotly.offline.iplot
            '''
            plotly.offline.plot({
            "data": [Scatter(x=[x for x in range(len(V1))] , y=V1),
                    Scatter(x=[x for x in range(len(V2))] , y=V2),
                    Scatter(x=[x for x in range(len(V2))] , y=V3),
                    Scatter(x=[x for x in range(len(V2))] , y=V4)],
                    #Scatter(x=[x for x in range(len(V2))] , y=V5),
                    #Scatter(x=[x for x in range(len(V2))] , y=V6),
                    #Scatter(x=[x for x in range(len(V2))] , y=V7),
                    #Scatter(x=[x for x in range(len(V2))] , y=V8)],
            "layout": Layout(title="",xaxis=dict(title="state"),yaxis=dict(title="V(s)"))
            })
            '''
            # optimal steps
            plotly.offline.plot({
            "data": [Scatter(x=[x for x in range(len(opt_array))], y=opt_array)]
            })
    plt.plot(opt_array)
    plt.show()
        
    
        
    


#testing_maze(25,100000,20)

#maze_task(10,20000,100)

#flow_control_test(10,5)

color_maze_task(10,100000,100)

