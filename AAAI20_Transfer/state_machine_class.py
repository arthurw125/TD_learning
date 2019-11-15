class state_machine:
    def __init__(self,state_table):
        self.state_table = state_table
        
    def move(self,cur_loc,action):
        new_state = ((self.state_table[cur_loc[0]][cur_loc[1]])["transitions"])[action]
        new_loc = [int(new_state[1]),int(new_state[2])] # state must have format SRowCol example S12
        return new_state, new_loc
    
    @staticmethod
    def generate_state_table(nrows,ncols):
        mat = []
        for x in range(nrows):
            temp = []
            for y in range(ncols):
                table = {"state":"S"+str(x)+str(y),"actions":["up","down","left","right"],"action_restrict":[0,1,2,3]}
                temp.append(table)
            mat.append(temp)

        for row in range(nrows):
            for col in range(ncols):
                up = ((row-1)+len(mat))%len(mat)
                down = ((row+1)+len(mat))%len(mat)
                left = ((col-1)+len(mat[row]))%len(mat[row])
                right = ((col+1)+len(mat[row]))%len(mat[row]) 
                up = "S"+str(up)+str(col)
                down = "S"+str(down)+str(col)
                left = "S"+str(row)+str(left)
                right = "S"+str(row)+str(right)
                (mat[row][col])["transitions"] = [up,down,left,right]
                
        for mycol in range(ncols):
            (mat[0][mycol])["action_restrict"] = [1,2,3] # top
            (mat[nrows-1][mycol])["action_restrict"] = [0,2,3] # bottom
        for myrow in range(nrows):
            (mat[myrow][0])["action_restrict"] = [0,1,3] # left
            (mat[myrow][ncols-1])["action_restrict"] = [0,1,2] # right
            
        (mat[0][0])["action_restrict"] = [1,3] # top left corner
        (mat[0][ncols-1])["action_restrict"] = [1,2] # top right corner
        (mat[nrows-1][0])["action_restrict"] = [0,3] # bottom left corner
        (mat[nrows-1][ncols-1])["action_restrict"] = [0,2] # bottom right corner
        
        return mat
                
    @staticmethod
    def generate_state_table_bottle(nrows,ncols):
        mat = []
        for x in range(nrows):
            temp = []
            for y in range(ncols):
                table = {"state":"S"+str(x)+str(y),"actions":["up","down","left","right"],"action_restrict":[0,1,2,3]}
                temp.append(table)
            mat.append(temp)

        for row in range(nrows):
            for col in range(ncols):
                up = ((row-1)+len(mat))%len(mat)
                down = ((row+1)+len(mat))%len(mat)
                left = ((col-1)+len(mat[row]))%len(mat[row])
                right = ((col+1)+len(mat[row]))%len(mat[row]) 
                up = "S"+str(up)+str(col)
                down = "S"+str(down)+str(col)
                left = "S"+str(row)+str(left)
                right = "S"+str(row)+str(right)
                (mat[row][col])["transitions"] = [up,down,left,right]
                
        for mycol in range(nrows):
            (mat[0][mycol])["action_restrict"] = [1,2,3] # top
            (mat[4][mycol])["action_restrict"] = [0,2,3] # bottom
            (mat[mycol][0])["action_restrict"] = [0,1,3] # left
            (mat[mycol][4])["action_restrict"] = [0,1,2] # right
            (mat[1][mycol])["action_restrict"] = [0,2,3] # barrier no down
            (mat[2][mycol])["action_restrict"] = [1,2,3] # barrier no up
        
        (mat[0][0])["action_restrict"] = [1,3] # top left corner
        (mat[0][4])["action_restrict"] = [1,2] # top right corner
        (mat[4][0])["action_restrict"] = [0,3] # bottom left corner
        (mat[4][4])["action_restrict"] = [0,2] # bottom right corner
        (mat[1][0])["action_restrict"] = [0,3] # barrier no down no left
        (mat[2][0])["action_restrict"] = [1,3] # barrier no up no left
        (mat[1][4])["action_restrict"] = [0,2] # barrier no down no right
        (mat[2][4])["action_restrict"] = [1,2] # barrier no up no right
        (mat[1][2])["action_restrict"] = [0,1,2,3] # barrier opening
        (mat[2][2])["action_restrict"] = [0,1,2,3] # barrier opening
        
        
        return mat

    @staticmethod
    def generate_state_table_bottle_v2(nrows,ncols):
        mat = []
        bar_row = None
        for x in range(nrows):
            temp = []
            for y in range(ncols):
                table = {"state":"S"+str(x)+str(y),"actions":["up","down","left","right"],"action_restrict":[0,1,2,3]}
                temp.append(table)
            mat.append(temp)

        for row in range(nrows):
            for col in range(ncols):
                up = ((row-1)+len(mat))%len(mat)
                down = ((row+1)+len(mat))%len(mat)
                left = ((col-1)+len(mat[row]))%len(mat[row])
                right = ((col+1)+len(mat[row]))%len(mat[row]) 
                up = "S"+str(up)+str(col)
                down = "S"+str(down)+str(col)
                left = "S"+str(row)+str(left)
                right = "S"+str(row)+str(right)
                (mat[row][col])["transitions"] = [up,down,left,right]
                
        for mycol in range(ncols):
            (mat[0][mycol])["action_restrict"] = [1,2,3] # top
            (mat[nrows-1][mycol])["action_restrict"] = [0,2,3] # bottom
            
            (mat[2][mycol])["action_restrict"] = [0,2,3] # barrier no down
            (mat[3][mycol])["action_restrict"] = [1,2,3] # barrier no up
            
        for myrow in range(nrows):
            (mat[myrow][0])["action_restrict"] = [0,1,3] # left
            (mat[myrow][ncols-1])["action_restrict"] = [0,1,2] # right
            
        (mat[0][0])["action_restrict"] = [1,3] # top left corner
        (mat[0][ncols-1])["action_restrict"] = [1,2] # top right corner
        (mat[nrows-1][0])["action_restrict"] = [0,3] # bottom left corner
        (mat[nrows-1][ncols-1])["action_restrict"] = [0,2] # bottom right corner
        
        (mat[2][0])["action_restrict"] = [0,3] # barrier no down no left
        (mat[3][0])["action_restrict"] = [1,3] # barrier no up no left
        (mat[2][4])["action_restrict"] = [0,2] # barrier no down no right
        (mat[3][4])["action_restrict"] = [1,2] # barrier no up no right
        (mat[2][2])["action_restrict"] = [0,1,2,3] # barrier opening
        (mat[3][2])["action_restrict"] = [0,1,2,3] # barrier opening
        
    @staticmethod
    def generate_state_table_multi_layer(nrows,ncols):
        mat = []
        bar_row = None
        for x in range(nrows):
            temp = []
            for y in range(ncols):
                table = {"state":"S"+str(x)+str(y),"actions":["up","down","left","right"],"action_restrict":[0,1,2,3]}
                temp.append(table)
            mat.append(temp)

        for row in range(nrows):
            for col in range(ncols):
                up = ((row-1)+len(mat))%len(mat)
                down = ((row+1)+len(mat))%len(mat)
                left = ((col-1)+len(mat[row]))%len(mat[row])
                right = ((col+1)+len(mat[row]))%len(mat[row]) 
                up = "S"+str(up)+str(col)
                down = "S"+str(down)+str(col)
                left = "S"+str(row)+str(left)
                right = "S"+str(row)+str(right)
                (mat[row][col])["transitions"] = [up,down,left,right]
                
        for mycol in range(ncols):
            (mat[0][mycol])["action_restrict"] = [1,2,3] # top
            (mat[nrows-1][mycol])["action_restrict"] = [0,2,3] # bottom
            
            (mat[1][mycol])["action_restrict"] = [0,2,3] # barrier1 no down 
            (mat[2][mycol])["action_restrict"] = [1,2,3] # barrier1 no up
            (mat[3][mycol])["action_restrict"] = [0,2,3] # barrier2 no down
            (mat[4][mycol])["action_restrict"] = [1,2,3] # barrier2 no up
            
        for myrow in range(nrows):
            (mat[myrow][0])["action_restrict"] = [0,1,3] # left
            (mat[myrow][ncols-1])["action_restrict"] = [0,1,2] # right
            
        (mat[0][0])["action_restrict"] = [1,3] # top left corner
        (mat[0][ncols-1])["action_restrict"] = [1,2] # top right corner
        (mat[nrows-1][0])["action_restrict"] = [0,3] # bottom left corner
        (mat[nrows-1][ncols-1])["action_restrict"] = [0,2] # bottom right corner
        
        (mat[1][0])["action_restrict"] = [0,3] # barrier no down no left
        (mat[2][0])["action_restrict"] = [1,3] # barrier no up no left
        (mat[1][2])["action_restrict"] = [0,2] # barrier no down no right
        (mat[0][2])["action_restrict"] = [1,2] # barrier G right corner
        (mat[0][4])["action_restrict"] = [1,3] # barrier B left corner
        (mat[1][4])["action_restrict"] = [0,3] # barrier no down no left
        (mat[1][6])["action_restrict"] = [0,2] # barrier no down no right
        (mat[2][6])["action_restrict"] = [1,2] # barrier no up no right
        (mat[3][0])["action_restrict"] = [0,3] # barrier no down no left
        (mat[4][0])["action_restrict"] = [1,3] # barrier no up no left
        (mat[3][6])["action_restrict"] = [0,2] # barrier no down no right
        (mat[4][6])["action_restrict"] = [1,2] # barrier no up no right
        
        (mat[3][3])["action_restrict"] = [0,1,2,3] # barrier opening
        (mat[4][3])["action_restrict"] = [0,1,2,3] # barrier opening
        (mat[1][1])["action_restrict"] = [0,1,2,3] # barrier opening
        (mat[2][1])["action_restrict"] = [0,1,2,3] # barrier opening
        (mat[1][5])["action_restrict"] = [0,1,2,3] # barrier opening
        (mat[2][5])["action_restrict"] = [0,1,2,3] # barrier opening
        
        return mat