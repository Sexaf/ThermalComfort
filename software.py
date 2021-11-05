from flask import *
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080)   # 0.0.0.0地址可给本机所有的IP地址访问


# def GetDiss(a):
#     pass
#
#
# def TimeDiff(r):
#     list = []
#     c = 300  # 修改为声音在空气中传播速速
#     for i in range(len(r)):
#         for j in range(len(r)):
#             list.append((r[i] - r[j]) / c)
#
# def CosFunc(l):
#
#
#
# if __name__ == "__mian__":
#     a = []  # 修改为真实距离(传感器获取，答辩展示时自行模拟数据)
#     b = GetDiss(a)
#     c = TimeDiff(b)