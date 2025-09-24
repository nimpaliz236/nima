import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])

b = np.full_like(a, 9)

print("a:\n", a)
print("b:\n", b)

x=np.random.random((3,4))
print("x:\n", x)

c=np.random.random_sample((a.shape))
print("c:\n", c)

v=np.random.randint(7 ,size=(3,4))
print("v:\n", v)

b=np.identity(3)
print("b:\n", b)

arr=np.array([[1, 2, 3]])
r1=np.repeat(arr,3,axis=1)
print("r1:\n", r1)

m=np.array([1,2,3])
b=m.copy()
b[0]=200
print("m:\n", m)
print("b:\n", b)

q=np.ones((2,3))
w=np.full((3,2),2)
e=np.matmul(q,w)
print("e:\n", e)

v1=np.array([1,2,3,4])
v2=np.array([5,6,7,8])
v3=np.vstack([v1,v2,v2,v2])
v4=np.hstack((v1,v2))
print("v4:\n", v4)
print("v3:\n", v3)



data = np.genfromtxt('C:\\Users\\elecomp\\Desktop\\nima.txt', delimiter=',', filling_values=0)
data.astype(int)
print("data: \n",data)


