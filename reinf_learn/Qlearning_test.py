#-*- coding:utf-8 –*-
import numpy as np
import pandas as pd
import time

np.random.seed(2)
N_STATES =6
ACTION =['left','right']
EPSILON=0.9
ALPHA = 0.1
LAMBDA=0.9
MAX_EPISODES =13#最多的回合数
FRESH_TIME=0.01#每一步更新时间

def build_q_table(n_states,actions):
    table=pd.DataFrame(
        np.zeros((n_states,len(actions))),
        columns = actions,
    )
    #print(table)
    return table


#build_q_table(N_STATES,ACTION)



def choose_action(states,q_table):
    states_actions = q_table.iloc[states,:]
    if(np.random.uniform()>EPSILON)or(states_actions.all()==0):
        action_name=np.random.choice(ACTION)
    else:
        action_name=states_actions.argmax()
    return action_name

def get_env_reward(S,A):
    if A=='right':
        if S==N_STATES-2:
            S_='terminal'
            R=1
        else:
            S_=S+1
            R=0
    else:
        R=0
        if S==0:
            S_=S
        else:
            S_=S-1
    return S_,R

def update_env(S,episode,step_counter):
    env_list=['-']*(N_STATES-1)+['T']
    if S =='terminal':
        interation = '\t Episode %s: total_steps=%s'%(episode+1,step_counter)
        print('\r{}'.format(interation),end='')
        time.sleep(2)
        print('\r                 ',end='')
    else:
        env_list[S]='o'
        interation=''.join(env_list)
        print('\r{}'.format(interation),end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table=build_q_table(N_STATES,ACTION)
    for episode in range(MAX_EPISODES):
        step_counter=0
        S=0
        is_terminal =False
        update_env(S,episode,step_counter)
        while not is_terminal:
            A=choose_action(S,q_table)
            S_,R=get_env_reward(S,A)
            q_predict =q_table.ix[S,A]
            if S_!='terminal':
                q_target=R+LAMBDA*q_table.iloc[S_,:].max()
            else:
                q_target=R
                is_terminal=True

            q_table.ix[S,A]+=ALPHA*(q_target-q_predict)
            S=S_

            update_env(S,episode,step_counter+1)
            step_counter+=1
    return q_table

if __name__=="__main__":
    q_table=rl()
    print('\r\nQ-table:\n')
    print(q_table)


