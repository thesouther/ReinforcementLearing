#-*- coding:utf-8 –*-
import tkinter as tk
import pickle
import tkinter.messagebox
window=tk.Tk()
window.title('test')
window.geometry('450x300')
#加载图片
canvas=tk.Canvas(window,width=400,height=100)
image_file=tk.PhotoImage(file='kk.gif')
image=canvas.create_image(150,0,anchor='nw',image=image_file)
canvas.pack(side='top')
#输入框
tk.Label(window,text='name:').place(x=50,y=150)
tk.Label(window,text='password:').place(x=50,y=190)
usr_name_var=tk.StringVar()
usr_name_var.set("12323@python.com")
usr_pwd_var=tk.StringVar()
usr_name=tk.Entry(window,textvariable=usr_name_var)
usr_name.place(x=160,y=150)
usr_pwd=tk.Entry(window,textvariable=usr_pwd_var,show='*')
usr_pwd.place(x=160,y=190)

def login():
    usr_name=usr_name_var.get()
    usr_pwd=usr_pwd_var.get()
    try:
        with open('usr_info.pickle','rb') as usr_file:
            usr_info=pickle.load(usr_file)
    except FileNotFoundError:
        with open('usr_info.pickle','wb') as usr_file:
            usr_info={'admin':'admin'}
            pickle.dump(usr_info,usr_file)

    if usr_name in usr_info:
        if usr_pwd ==usr_info[usr_name]:
            tk.messagebox.showinfo(title='Hello',message="hahahaha")
        else:
            tk.messagebox.showerror(message="Error!")
    else:
        is_sign_up=tk.messagebox.askyesno(title='Welcome',message='Would you like to sign up now?')

        if is_sign_up:
            signup()

def signup():
    def sign_upto_python():
        nn=new_name.get()
        np=new_pwd.get()
        npf=new_pwd_conf.get()
        with open('usr_info.pickle','rb') as usr_file:
            exist_usr_info=pickle.load(usr_file)
        if np!=npf:
            tk.messagebox.showerror(title="error",message='password and confirm must be same!')
        elif nn in exist_usr_info:
            tk.messagebox.showerror(title='error',message='the user name has been signed!')
        else:
            exist_usr_info[nn]=np
            with open('usr_info.pickle','wb')as usr_file:
                pickle.dump(exist_usr_info,usr_file)
            tk.messagebox.showinfo(title='Welcome',message='welcome')
            window_signup.destroy()


    window_signup=tk.Toplevel(window)
    window_signup.title('Sign up')
    window_signup.geometry('350x200')

    new_name = tk.StringVar()
    new_name.set("12323@python.com")
    tk.Label(window_signup, text='name:').place(x=10, y=30)
    usr_name = tk.Entry(window_signup, textvariable=new_name)
    usr_name.place(x=90, y=30)

    new_pwd = tk.StringVar()
    tk.Label(window_signup, text='password:').place(x=10, y=70)
    usr_pwd = tk.Entry(window_signup, textvariable=new_pwd, show='*')
    usr_pwd.place(x=90, y=70)

    new_pwd_conf = tk.StringVar()
    tk.Label(window_signup, text='conf:').place(x=10, y=110)
    usr_pwd_conf = tk.Entry(window_signup, textvariable=new_pwd_conf, show='*')
    usr_pwd_conf.place(x=90, y=110)

    btn_conf_signup = tk.Button(window_signup, text='sign up', command=sign_upto_python)
    btn_conf_signup.place(x=110, y=130)

#登录注册
btn_login=tk.Button(window,text='login',command=login)
btn_login.place(x=100,y=210)
btn_signup=tk.Button(text='sign up',command=signup)
btn_signup.place(x=180,y=210)

window.mainloop()