#-*- coding:utf-8 –*-
import tkinter as tk

window=tk.Tk()
window.title('test')
window.geometry('200x400')

var=tk.StringVar()
l=tk.Label(window,bg='red',width=10,text='empty')
l.pack()

def print_selection():
    l.config(text=var.get())

rb1=tk.Radiobutton(window,text='A',variable=var,value='A',command=print_selection)
rb1.pack()
rb2=tk.Radiobutton(window,text='B',variable=var,value='B',command=print_selection)
rb2.pack()
rb3=tk.Radiobutton(window,text='C',variable=var,value='C',command=print_selection)
rb3.pack()
window.mainloop()