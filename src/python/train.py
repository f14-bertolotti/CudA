
from lenia.lenia import Lenia

def compile_cpp(x)->str:
    cppstr = ""
    for i,r in enumerate(x):
        for j,v in enumerate(r):
            cppstr += f"ptr[{i}*size+{j}]={v}f;"
        cppstr += "\n"
    return cppstr


lenia = Lenia(128)

from geneticalgorithm import geneticalgorithm as ga
import numpy as np
import torch

def f(x):
    x = torch.tensor(x).cuda().reshape(10,10)
    lenia.grid.fill_(0.0)
    lenia.grid[:10,:10] = x
    lenia.run(20)
    l = (((lenia.grid[:10,30:40] - x)**2).sum()/2).item()
    return l 

varbound= np.array([[0,1]]*100)
model=ga(function=f,dimension=100,variable_type='real',variable_boundaries=varbound)
model.run()
print(model.report)
print(model.ouput_dict)


