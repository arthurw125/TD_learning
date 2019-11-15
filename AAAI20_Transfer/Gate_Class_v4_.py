import matplotlib.pyplot as plt
import plotly
import numpy as np
import random
import hrr2 as hrr
import math

#from plotly.graph_objs import Scatter, Layout, Surface
#plotly.offline.init_notebook_mode(connected=True)

def argmax2(arr_2d,wm_restrict,action_restrict):
    max_row = wm_restrict[0]
    max_col = action_restrict[0]
    #max_value = arr_2d[0,0]
    max_value = arr_2d[wm_restrict[0],action_restrict[0]]
    for row in range(arr_2d.shape[0]):
        if row not in wm_restrict:
            continue
        for col in range(arr_2d.shape[1]):
            if col not in action_restrict:
                continue
            if arr_2d[row,col] > max_value:
                max_value = arr_2d[row,col]
                max_row,max_col = row,col
    return list((max_row,max_col))

def softmax(arr,t=1.0):
    w = np.array(arr)
    e = np.exp(w / t)
    dist = e / np.sum(e)
    return dist

def log_transform(error):
    return math.copysign(1.0,error)*math.log(math.fabs(error)+1,2)

def identity_vector(n):
    hrr_i = np.zeros(n)
    hrr_i[0] = 1
    return hrr_i

def argmax(arr,restrict):
    max_col = restrict[0]
    max_val = arr[restrict[0]]
    for x in range(len(arr)):
        if x not in restrict:
            continue
        if arr[x] > max_val:
            max_val = arr[x]
            max_col = x
    return max_col

def get_action_set(row,col,arr):
    if row==0 and col==0: # top left corner
        pass
    elif row==0 and col==len(arr[row])-1: # top right corner
        pass
    elif row==len(arr)-1 and col==0: # bottom left corner
        pass
    elif row==len(arr)-1 and col==len(arr[row])-1: # bottom right corner
        pass
'''
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
'''

# Reinforcement Learning class
### changes from version3 to version4 ######
# added 'action_restict' parameter to 'action' method
# removed 'action_list' parameter from init method
# added 'action_list' parameter to 'action' method
# no need for Gate classes anymore
class RL_Obj:
    
    def __init__(self,n,LTM_obj,bias=1,lrate=0.1,gamma=0.9,td_lambda=0.9,epsilon=0.1,W=0):
        self.n = n # vector length
        #self.actions = action_list
        #self.nactions = len(action_list)
        self.LTM = LTM_obj # long term memory object
        self.W = hrr.hrr(n) if isinstance(W,int) else W # network weights
        #self.actions = hrr.hrrs(n,nactions)
        self.eligibility = hrr.hrr(n)
        self.bias = bias
        self.lrate = lrate
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.epsilon = epsilon
        
    def action(self,state_space,action_list,action_restrict):
        #mystate = hrr.convolve(state_space,self.actions)
        nactions = len(action_list)
        mystate = []
        for action in action_list:
            state = self.LTM.encode(action+'*'+state_space)
            mystate.append(state)
        values = np.dot(mystate,self.W) + self.bias
        #print(values)
        sm = softmax(values)
        #action = np.argmax(sm)
        restrict = action_restrict # restricted set of actions
        action = argmax(sm,restrict)
        if random.random() < self.epsilon:
            #action = random.randrange(0,nactions)
            action = random.choice(action_restrict)
        
        x = mystate[action] # input
        #return action, values[action], my_state2[action]
        return action, values[action], x
    
    def WM_action(self,state_space,action_list,wm_list,wm_restrict,action_restrict):
        #mystate = hrr.convolve(state_space,self.actions)
        nactions = len(action_list)
        nwm = len(wm_list)
        mystate = []
        wm_state = []
        for wm in wm_list:
            mystate = []
            for action in action_list:
                state = self.LTM.encode(wm+'*'+action+'*'+state_space)
                mystate.append(state)
            
            wm_state.append(np.array(mystate))
        wm_state = np.array(wm_state)
        values = np.dot(wm_state,self.W) + self.bias
        #print(values)
        sm = softmax(values)
        #action = np.argmax(sm)
        
        #action = argmax(sm,restrict)
        wm_action = argmax2(sm,wm_restrict,action_restrict)
        wm = wm_action[0]
        action = wm_action[1]
        #print(wm,action)
        if random.random() < self.epsilon:
            #wm = random.randrange(0,nwm)
            #action = random.randrange(0,nactions)
            wm = random.choice(wm_restrict)
            action = random.choice(action_restrict)
        
        x = wm_state[wm,action] # input
        #return action, values[action], my_state2[action]
        #print(values)
        return wm,action, values[wm,action], x
   
    def eligibility_trace_update(self,my_input):
        #s_a = hrr.convolve(state_vec,action_vec)
        self.eligibility = my_input + self.td_lambda*self.eligibility
    
    def set_eligibility_zero(self):
        self.eligibility = np.zeros(self.n)
    
    def td_update(self,r,value,pvalue):
        error = (r+self.gamma*value) - pvalue
        self.W += self.lrate*log_transform(error)*self.eligibility
        return error
        
    def td_update_star(self,value,pvalue):
        error = value - pvalue
        self.W += self.lrate*log_transform(error)*self.eligibility
        return error
    
    def td_update_transfer(self,r,value,pvalue,beta):
        error = (r+self.gamma*(self.gamma**beta)*value) - pvalue
        self.W += self.lrate*log_transform(error)*self.eligibility
        
    def td_update_goal(self,r,value):
        error = r - value
        self.W += self.lrate*log_transform(error)*self.eligibility
        return error
    
    def get_weights(self):
        return self.W

class Q_learning(RL_Obj):
    
    def action(self,state_space,action_list,action_restrict):
        #mystate = hrr.convolve(state_space,self.actions
        nactions = len(action_list)
        mystate = []
        for action in action_list:
            state = self.LTM.encode(action+'*'+state_space)
            mystate.append(state)
        values = np.dot(mystate,self.W) + self.bias
        #print(values)
        sm = softmax(values)
        #action = np.argmax(sm)
        action = argmax(sm,action_restrict)
        max_action = action
        if random.random() < self.epsilon:
            #action = random.randrange(0,nactions)
            action = random.choice(action_restrict)
        # force gate to be closed
        #if close:
        #    action = 0
        #x = hrr.convolve(mystate,self.actions[action]) # input
        x = mystate[action] # input
        #return action, values[action], my_state2[action]
        return action, values[action], values[max_action], x, values

class wm_content:
    
    def __init__(self,item_list,n_wm_slots,LTM_obj):
        #self.n = n
        #self.nitems = nitems
        # encode working memory items
        self.wm_items = []
        for item in item_list:
            if item != '':
                LTM_obj.encode('WM_'+item)
                self.wm_items.append('WM_'+item)
        
        #LTM_obj.encode('')
        self.wm_items.append('')
        self.items = item_list
        self.n_wm_slots = n_wm_slots
        self.wm_maint = ['']*n_wm_slots # init wm maint slots
        self.wm_output = ['']*n_wm_slots # init wm output slots
    
    def update_wm_maint(self,slot_num,item_num):
        self.wm_maint[slot_num] = self.wm_items[item_num]
        #self.wm_maint_statistics[slot_num] = item_num # updates stats for wm
    
    def update_wm_output(self,slot_num,item_num):
        self.wm_output[slot_num] = self.wm_items[item_num]
        #self.wm_output_statistics[slot_num] = item_num
        
    def get_all_wm_maint(self):
        return self.wm_maint # returns matrix of wm_maint contents
    
    def get_all_wm_output(self):
        return self.wm_output # returns matrix of wm_output contents
    
    def get_one_wm_maint(self,slot_num):
        return self.wm_maint[slot_num] # returns vector
    
    def get_one_wm_output(self,slot_num):
        return self.wm_output[slot_num] # returns vector
    
    def flush_all_wm_maint(self):
        self.wm_maint = ['']*self.n_wm_slots
        #self.wm_maint_statistics = [-1]*self.n_wm_slots 
        
    def flush_all_wm_output(self):
        self.wm_output = ['']*self.n_wm_slots
        #self.wm_output_statistics = [-1]*self.n_wm_slots

    def wm_maint_slot_is_empty(self,slot_num):
        return np.array_equal('',self.wm_maint[slot_num])
    
    def wm_output_slot_is_empty(self,slot_num):
        return np.array_equal('',self.wm_output[slot_num])
    
    # controls the flow of wm contents from wm_maint layer to wm_output layer
    
    def wm_in_flow(self,i_gate_state,slot_num,item_num):
        if i_gate_state:
            self.update_wm_maint(slot_num,item_num)
            
    def wm_out_flow(self,o_gate_state,slot_num):
        if o_gate_state:
            target = self.wm_maint[slot_num]
            item_num = self.wm_items.index(target)
            #item_num = self.get_wm_maint_statistics()[slot_num]
            self.update_wm_output(slot_num,item_num)
