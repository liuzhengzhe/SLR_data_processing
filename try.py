import json
file = open('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/project/src/EducationwithDiff.json','rb')
p=json.load(file)
print p['HKG_001_a_0013'][1]['diff']
'''import csv
with open('/media/lzz/HD11/kinect/Aaron/HKG_001_a_0001 Aaron 11/feature.csv','rb') as Label1:
    reader = csv.reader(Label1)
    labelArr1 = []
    last=-999
    #print self.path
    for row in reader:
        print len(row)'''

