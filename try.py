import json
file = open('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/project/src/EducationNew.json','rb')
p=json.load(file)
print p['HKG_001_b_0023']

