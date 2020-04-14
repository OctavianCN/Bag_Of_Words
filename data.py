import numpy as np
class data:

    def __init__(self):
        self.id = []
        self.lista=[]

    def config_data(self,path):
        if isinstance(path, str):
            AllData=open(path, 'r',encoding="utf8").readlines()
            for val in AllData:
                self.lista.append(val.split())
            for val in self.lista:
                self.id.append(val[0])
                del(val[0])
        else:
            print(path, "is not a string")



