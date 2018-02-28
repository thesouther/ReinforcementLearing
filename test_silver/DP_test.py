#-*- coding:utf-8 –*-
#初始化格子状态
states = [i for i in range(16)]
#初始化状态值
values = [0 for _ in range(16)]
#行为
actions = ["n","e","s","w"]
#行为对应的状态改变量
ds_actions = {"n":-4, "e":1,"s":4,"w":-1}
#折损因子
gamma = 1.00

#根据当前状态和采取的行为计算下一个状态id 以及得到的临时奖励
def nextStates(s, a):
    next_states= s
    if(s%4==0 and a=="w")or(s<4 and a=="n")or\
            ((s+1)%4==0 and a== "e")or(s>11 and a=="s"):
        pass
    else:
        ds =ds_actions[a]
        next_states = s+ds
    return next_states

#计算奖励
def rewardOf(s):
    return 0 if s in [0,15] else -1

def isTerminalStates(s):
    return s in [0,15]

def getSuccessors(s):
    successor =[]
    if isTerminalStates(s):
        return successor
    for a in actions:
        next_states =nextStates(s, a)
        successor.append(next_states)
    return successor

#更新s的value
def updateValue(s):
    successors = getSuccessors(s)
    newValue = 0
    num =4
    reward = rewardOf(s)
    for next_states in successors:
        newValue   +=1.00/num*(reward+gamma*values[next_states])
    return newValue
#进行一步迭代
def PerformOneIteration():
    newValues =[0 for _ in range(16)]
    for s in states:
        newValues[s]=updateValue(s)
    global values
    values = newValues
    PrintValue(values)

#输出
def PrintValue(v):
    for i in range(16):
        print('{0:>6.2f}'.format(v[i]),end="")
        if (i+1)%4==0:
            print(" ")
    print()

def test():
    PrintValue(states)
    PrintValue(values)
    for s in states:
        reward =rewardOf(s)
        for a in actions:
            next_states =nextStates(s, a)
            print("({0},{1})-> {2},with reward {3}".format(s,a,next_states,reward))

    for i in range(200):
        PerformOneIteration()
        PrintValue(values)
def main():
    max_iterate_times =160
    cur_iterate_times=0
    while cur_iterate_times<=max_iterate_times:
        print("Iteration No.{0}".format(cur_iterate_times))
        PerformOneIteration()
        cur_iterate_times+=1
    PrintValue(values)

if __name__ == '__main__':
    main()