import os

for i in range(1,60):
    #os.rename(str(i)+'.jpg',str(i)+'_0.jpg')
    for j in range(1,8):
        os.remove(str(i)+'_'+str(j)+'.jpg');
