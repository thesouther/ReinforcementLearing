#-*- coding:utf-8 –*-
import tkinter as tk

window=tk.Tk()
window.title('test')
window.geometry('200x400')

l=tk.Label(window,bg='red',width=10,text='empty')
l.pack()
var1=tk.IntVar()
var2=tk.IntVar()
def print_selection():
    if (var1.get()==1)and (var2.get()==1):
        l.config(text='hhhhh both')
    elif (var1.get()==1)and (var2.get()==0):
        l.config(text='hhhhh python')
    elif (var1.get()==0)and (var2.get()==1):
        l.config(text='hhhhh c++')
    else:
        l.config(text='nnnnnn both')

c1=tk.Checkbutton(window,text='python',variable=var1,onvalue=1,offvalue=0,
                  command=print_selection)
c2=tk.Checkbutton(window,text='C++',variable=var2,onvalue=1,offvalue=0,
                  command=print_selection)
c1.pack()
c2.pack()
window.mainloop()