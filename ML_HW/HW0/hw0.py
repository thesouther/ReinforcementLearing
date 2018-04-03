# -*- coding: utf-8 -*-

# Q1：
# method 1
# from collections import OrderedDict
# with open('words.txt', 'r') as f:
#     instr = f.read()
#     instr = instr.split()
#     od = OrderedDict()
#     for i in instr:
#         if i in od:
#             od[i] += 1
#         else:
#             od[i] = 1
#     cnt = 0
#     for k, v in od.items():
#        print(k, cnt, v)
#        cnt += 1
#     with open('./Hw0_Q1.txt', 'w') as w:
#         cnt = 0
#         for k, v in od.items():
#             tmp = str(k) + " " + str(cnt) + " " + str(v) + "\n"
#             cnt +=1
#             w.write(tmp)

# # method 2
# import numpy as np
# with open('words.txt', 'r') as wd:
#     wdlist = wd.read()
#     wdlist = wdlist.split()
#     tem = [[], []]
#     no_tem = 0
#     for i in wdlist:
#         if i in tem[0]:
#             tem[1][tem[0].index(i)] +=1
#         else:
#             tem[0].append(i)
#             tem[1].append(1)
#     with open('Hw0_Q1.txt', 'w') as ans:
#         for j in range(len(tem[0])):
#             opq = str(tem[0][j]) + " " + str(j) + " " + str(tem[1][j]) + "\n"
#             ans.write(opq)

# Q2:
from PIL import Image
im = Image.open('westbrook.jpg')
pix = im.load()
print(pix)
w,h = im.size
newim = Image.new("RGB", (w,h))
for i in range(w):
    for j in range(h):
        r,g,b = pix[i,j]
        newim.putpixel((i,j),(r//2,g//2,b//2))
newim.save('Q2.jpg', 'jpeg')