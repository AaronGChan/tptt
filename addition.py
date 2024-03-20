"""
source: https://github.com/nmanchev/tptt/blob/master/addition.py
author: Hitesh Vaidya
"""

import numpy as np

class AddTask(object):
    def __init__(self, rng, floatX) -> None:
        self.rng = rng
        self.floatX = floatX
        self.nin = 2
        self.nout = 1
        self.classifType = 'lastLinear'

    def generate(self, batchsize, length):
        l = self.rng.randint(int(length*0.1)) + length
        p0 = self.rng.randint(int(l*0.1), size=(batchsize,))
        p1 = self.rng.randint(int(l*0.4), size=(batchsize,)) + int(l*0.1)
        data = self.rng.uniform(size=(l, batchsize, 2)).astype(self.floatX)
        data[:,:,0] = 0.
        data[p0, np.arange(batchsize), np.zeros((batchsize,),
                                                dtype='int32')] = 1.
        data[p1, np.arange(batchsize), np.zeros((batchsize,),
                                                dtype='int32')] = 1.
        
        targs = (data[p0, np.arange(batchsize),
                      np.ones((batchsize,), dtype='int32')]) + \
                    data[p1, np.arange(batchsize),
                         np.ones((batchsize,), dtype='int32')] / 2.
        return data, targs.reshape((-1, 1)).astype(self.floatX)
    

if __name__ == "__main__":
    print("Testing add task generator ..")
    addtask = AddTask(np.random.RandomState(42), "float32")
    seq, targ = addtask.generate(3,10)
    assert seq.dtype == 'float32'
    assert seq.dtype == 'float32'
    

    print("Seq_0")
    print(seq[:,0,:])
    print("Targ0", targ[0])
    print()

    print('Seq_1')
    print(seq[:,1,:])
    print('Targ1', targ[1])
    print()
    
    print('Seq_2')
    print(seq[:,2,:])
    print('Targ2', targ[2])