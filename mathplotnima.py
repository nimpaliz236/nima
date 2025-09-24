import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'B Nazanin'

x=[2,4,6,8,10]
y=[10,30,50,70,90]
#رسم نمودار خطی
#plt.plot(x,y,color='red',linestyle='--',marker='o')
#plt.title("نمودارخطی")
#plt.xlabel("x")
#plt.ylabel("y")
#plt.show()

#رسم دو خط روی یک نمودار
y2=[1,3,5,7,9]
#plt.plot(x,y,label='خط اول')
#plt.plot(x,y2,label='خط دوم')
#plt.title('چند نمودار روی یک شکل')
#plt.legend()
#plt.show()

#رسم نمودار میله ای
categories=['A','B','C']
values=[10,20,15]
#plt.bar(categories, values, color=['red','blue','green'])
#plt.title('نمودار میله ای')
#plt.show()

#رسم نمودار پراکندگی
#plt.scatter(x,y,color='black',marker='o')
#plt.title('نمودار پراکندگی')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

#هیستوگرام
data=[1,2,2,3,3,3,4,4,4,4,5,5,6,7,8,8,9]
plt.hist(data,bins=5,color='blue',edgecolor='black')
plt.title('هیستوگرام')
plt.show()
plt.savefig('hist.png')
