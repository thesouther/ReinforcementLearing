#-*- coding:utf-8 –*-
import tkinter as tk

window=tk.Tk()
window.title('test')
window.geometry('200x200')

e=tk.Entry(window,show=None)
e.pack()

t=tk.Text(window,height=2)
t.pack()

def insert():
    var=e.get()
    t.insert('insert',var)
def append():
    var=e.get()
    t.insert('end',var)
b1=tk.Button(window,text='insert',command=insert)
b1.pack()
b2=tk.Button(window,text='append',command=append)
b2.pack()
window.mainloop()