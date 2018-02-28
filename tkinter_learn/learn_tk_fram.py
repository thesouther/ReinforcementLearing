#-*- coding:utf-8 –*-
import tkinter as tk
window =tk.Tk()
window.title('test')
window.geometry('200x400')

tk.Label(window,text='on the window').pack()

frm=tk.Frame(window)
frm.pack()
frm_l=tk.Frame(frm,bg='red')
frm_r=tk.Frame(frm,bg='green')
frm_l.pack(side='left')
frm_r.pack(side='right')
tk.Label(frm_l,text='on the frm_l1',bg='red').pack()
tk.Label(frm_l,text='on the frm_l2',bg='green').pack()
tk.Label(frm_r,text='on the frm_r1',bg='yellow').pack()
tk.Label(frm_r,text='on the frm_r1',bg='yellow').pack()

window.mainloop()


