#-*- coding:utf-8 –*-
import tkinter as tk

window=tk.Tk()
window.title('test')
window.geometry('200x400')

l=tk.Label(window,bg='red',width=10,text='empty')
l.pack()

def print_selection(v):
    l.config(text=v)

s=tk.Scale(window,label='try me',from_=0,to=10,orient=tk.HORIZONTAL,length=200,showvalue=0,
           tickinterval=3,resolution=0.01,command=print_selection)
s.pack()
window.mainloop()