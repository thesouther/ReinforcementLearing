#-*- coding:utf-8 –*-
import tkinter as tk

window=tk.Tk()
window.title('test')
window.geometry('200x400')

var1=tk.StringVar()
l=tk.Label(window,textvariable=var1,bg='red',width=4)
l.pack()

def print_selection():
    value =lb.get(lb.curselection())
    var1.set(value)

b1=tk.Button(window,text='print_selection',command=print_selection)
b1.pack()

var2=tk.StringVar()
var2.set((11,22,33,44))
lb=tk.Listbox(window,listvariable=var2)
list_item=[1,2,3,4]
for item in list_item:
    lb.insert('end',item)
lb.insert(0,'first')
lb.insert(1,'second')
lb.delete(1)
lb.pack()

window.mainloop()