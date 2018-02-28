
#-*- coding:utf-8 –*-
import tkinter as tk
import tkinter.messagebox
window =tk.Tk()
window.title('test')
window.geometry('200x400')

# tk.Label(window,text='me').pack(side='top')
# tk.Label(window,text='me').pack(side='bottom')
# tk.Label(window,text='me').pack(side='right')
# tk.Label(window,text='me').pack(side='left')

# for i in range(4):
#     for j in range(3):
#         tk.Label(window,text=1).grid(row=i,column=j,ipadx=10,ipady=10)

tk.Label(window,text='me').place(x=10,y=100,anchor='e')
window.mainloop()