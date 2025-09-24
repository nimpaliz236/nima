import numpy as np

#ساخت ارایه یک بعدی با اعداد 10 تا 20
a=np.array([10,11,12,13,14,15,16,17,18,19,20])

#ساخت ماتریس رندوم 2در 3 بین صفر و یک
b=np.random.random((3,4))

#ساخت ماتریس پر 2در 3 بین اعداد 5 تا 15
c=np.random.randint(5,15,(2,3))

#ساخت ماتریس همانی 4در4
d=np.identity(4)

#ساخت دو ماتریس 2در 3 و 3در 2 و انجام ضرب ماتریسی
e=np.array([[1,2,3],[4,5,6]])
f=np.array([[7,8],[9,10],[11,12]])
g=np.matmul(e,f)
print("g:\n", g)

#ساخت ارایه با عناصر یک و جمع با عددی
j=np.full((3,3),1)
h=8
l=j+h
print("l:\n", l)

#ساخت ماتریس و سه بار تکرار کردن آن
p=np.array([1,2,3])
k=np.array([4,5,6])
z=np.repeat(p,3)
x=np.hstack((k,z))
x1=np.vstack((k,p))
print("x1:\n", x1)
print("x:\n", x)
print("z:\n", z)

#کار با csv
data = np.genfromtxt('C:\\Users\\elecomp\\Desktop\\nima.txt', delimiter=',', filling_values=0)
data.astype(int)
print("data: \n",data)


