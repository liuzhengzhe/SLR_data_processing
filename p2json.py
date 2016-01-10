import jsonpickle
import pickle
import json
 
if __name__ == "__main__":
    #my_car = Car()
    picklef = pickle.load( open( "/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/project/src/EducationFy.pickle", "rb" ) )
    data = jsonpickle.encode(picklef)
    with open('/home/lzz/education_fy.txt', 'w') as outfile:
        outfile.write(data)
    #print serialized
 
    #my_car_obj = jsonpickle.decode(serialized)
    #print my_car_obj.drive()