"""
Author: xufeng shen
"""
import tkinter
import tkinter.messagebox

class MyDialog(object): #定义对话框类

    def __init__(self, root): #对话框初始化
        self.root = root
        self.top = tkinter.Toplevel(root) #生成TopLevel组件
        self.top["bg"] = 'pink'
        self.top.geometry('568x231+5+5')
        label = tkinter.Label(self.top, text='输入衣服数据：') # 生成标签组件
        label.pack()
        self.entry = tkinter.Entry(self.top) # 生成文本框组件
        self.entry.pack()
        self.entry["bg"] = 'pink'
        self.entry.focus() # 文本框获得焦点
        button = tkinter.Button(self.top, text='确认', command=self.Ok)  #生成按钮，设置按钮处理函数
        button.pack()

    def Ok(self): # 定义按钮事件处理函数
        self.input = self.entry.get()  # 获取文本框中的内容，保存为input
        self.top.destroy() #销毁对话框

    def get(self): # 返回在文本框中内容
        return self.input


class MyButton(object):
    def __init__(self, root, type):
        self.root = root # 保持父窗口引用
        self.root.title("热舒适评估软件V1.0")
        self.root.geometry('1068x681+10+10')
        self.root["bg"] = "pink"
        if type == 0: # 类不同创建不同按钮
            self.button = tkinter.Button(root, text='开始预测', command=self.Create)
        else:
            self.button = tkinter.Button(root, text='退出软件', command=self.Quit)
        self.button.pack()

    def Create(self):
        d = MyDialog(self.root) # 生成一个对话框
        self.button.wait_window(d.top) # 等待对话框结束
        x = float(d.get())
        y = 0.60832566 * x + 0.20735498

        tkinter.messagebox.showinfo('预测结果', '预测PMV值为:\n' + str(y))  # 输出输入值，重点语句

    def Quit(self):  # 退出主窗口
        self.root.quit()


root = tkinter.Tk()
tkinter.Label(root, text='作者：沈栩烽', font=('Arial', 50), ).place(x=400, y=300, anchor='nw')
MyButton(root, 0)
MyButton(root, 1)

root.mainloop()


