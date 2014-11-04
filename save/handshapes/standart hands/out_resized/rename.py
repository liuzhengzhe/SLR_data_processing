import os

f = open('hog_template.txt');
fo = open('hog.txt','w')
for i in range(0,480):
   line = f.readline()
   if i % 8 == 0:
      fo.write(line)
