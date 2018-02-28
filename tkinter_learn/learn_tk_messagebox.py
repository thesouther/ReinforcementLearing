#-*- coding:utf-8 –*-
import tkinter as tk
import tkinter.messagebox
window =tk.Tk()
window.title('test')
window.geometry('200x400')

def hit_me():
    #tk.messagebox.showinfo(title='Hi',message='hahahaha')
    #tk.messagebox.showwarning(title='Hi',message='hahahaha')
    #tk.messagebox.showerror(title='Hi',message='hahahaha')
    # print(tk.messagebox.askquestion(title='Hi',message='hahahaha'))
    #print(tk.messagebox.askyesno(title='Hi', message='hahahaha'))
    print(tk.messagebox.askretrycancel(title='Hi', message='hahahaha'))
    print(tk.messagebox.askyesnocancel(title='Hi', message='hahahaha'))
    print(tk.messagebox.askokcancel(title='Hi', message='hahahaha'))
b=tk.Button(window,text='me',command=hit_me).pack()

window.mainloop()