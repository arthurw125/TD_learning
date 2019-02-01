from threading import Thread
import os
#import _thread


def run_system(tname,cmd):
    print(tname)
    os.system(cmd)
    
n = 50
cmd1 = "python3 OutputGate_s256.py >> o_gate_stats_256.dat"
cmd2 = "python3 OutputGate_s1024.py >> o_gate_stats_1024.dat"

for x in range(n):
    # Create two threads as follows
    try:
        #_thread.start_new_thread( run_system, ("Thread-"+x, 2, ) )
        #_thread.start_new_thread( run_system, ("Thread-"+x+1, 4, ) )
        t1 = Thread(target=run_system, args=("thread"+str(x)+"_s256",cmd1,))
        t2 = Thread(target=run_system, args=("thread"+str(x)+"_s1024",cmd2,))
        
        t1.start()
        t2.start()
        
    except:
       print ("Error: unable to start thread")

t1.join()
t2.join()
print("Finished -- Exiting")