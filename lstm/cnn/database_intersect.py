'''
Created on Oct 12, 2014

@author: liuzz
'''
import sqlite3

db=sqlite3.connect("./data/database791-821.db")
leng=open("D:/eclipse/project/save/generating/kmedoid/length.txt")
le=leng.readline().split(" ")
cu=db.cursor()
wh=db.execute("Select SignSample.SignID, SignSample.index_ID, SignSample.Signer from SignSample where Intersected = 0 ORDER BY SignSample.SignID")
id_inter=open("D:/eclipse/project/save/generating/kmedoid/id_inter.txt","w")
b=cu.fetchall()
index=0
i=-1

for signid,indexid,signer in wh:
    if(signid=="HKG_049_d_0016" and signer=="Micheal"):
        continue
    i+=1
    if(le[i]==str(0)):
        continue
    index+=1
    id_inter.write(signid+" "+str(index)+signer+"\n")
id_inter.close()
'''
for i in range(0,len(b)):
    tmp=b[i][0]
    tmp2=b[i][2]
    l=le[i]
    if(l==str(0)):
        continue
    index+=1
    id_inter.write(str(tmp)+" "+str(index)+str(tmp2)+"\n")'''