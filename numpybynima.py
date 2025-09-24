import numpybynima as np

#a=np.array([1,2,3])
#b=np.array([4,5,6])

#c = np.array([[7,8,9,10,11,12] ,[13,14,15,16,17,18] ,[19,20,21,22,23,24]])
#print(a.shape)
#print (c.ndim)
#print(c.itemsize)
#print(c.nbytes)
#print(a.size)
#print(c[0,:])
#print(c[1,:])
#print(c[2,:])
#print(c[:,0])
#print(c[:,1])
#print(c[:,2])
#slice_c = c[0, 2:6:2]
#print(slice_c)
#c[1,4]=80
#print(c)

#n = np.array([[1,2],
              #[3,4],
              #[5,6],
              #[7,8]])
#N = n[0, 1]
#print(N)
#print(n[2, :])
#print(n[:, 0])

#z=np.zeros((2,3,3),)
#print(z)

#o=np.ones((2,3,3),dtype=int)
#print(o)

#k=np.full((2,2),99,dtype=int)
#print(k)



a = np.array([[1, 2, 3],
              [4, 5, 6]])

b = np.full_like(a, 9)

print("a:\n", a)
print("b:\n", b)









