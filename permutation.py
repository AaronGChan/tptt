"""
source: https://github.com/nmanchev/tptt/blob/master/permutation.py
author: Hitesh Vaidya
"""

import numpy

class PermTask(object):
    def __init__(self, rng, floatX):
        self.rng = rng
        self.floatX = floatX
        self.nin = 100
        self.nout = 100
        self.classifType = 'lastSoftmax'
        self.report = 'last'

    def generate(self, batchsize, length):
        randvals = self.rng.randint(98, size=(length+1, batchsize)) + 2
        val = self.rng.randint(2, size=(batchsize,))
        randvals[numpy.zeros((batchsize,), dtype='int32'),
                 numpy.arange(batchsize)] = val
        randvals[numpy.ones((batchsize,), dtype='int32')*length,
                 numpy.arange(batchsize)] = val
        _targ = randvals[1:]
        _inp = randvals[:-1]
        inp = numpy.zeros((length, batchsize, 100), dtype=self.floatX)
        # targ = numpy.zeros((length, batchsize, 100), dtype=self.floatX)
        targ = numpy.zeros((1, batchsize, 100), dtype=self.floatX)
        inp.reshape((length*batchsize, 100))[\
                numpy.arange(length*batchsize),
                _inp.flatten()] = 1.
        #targ.reshape((length*batchsize, 100))[\
        #        numpy.arange(batchsize),
        #        _targ[-1].flatten()] = 1.
        targ.reshape((batchsize, 100))[\
                numpy.arange(batchsize),
                _targ[-1].flatten()] = 1.
        return inp, targ.reshape((batchsize, 100))


if __name__ == '__main__':
    print('Testing permutation task generator ..')
    task = PermTask(numpy.random.RandomState(123), 'float32')
    seq, targ = task.generate(3, 25)
    assert seq.dtype == 'float32'
    assert targ.dtype == 'float32'
    print('Seq_0')
    print(seq[:,0,:].argmax(axis=1))
    print('Targ0')
    print(targ[0].argmax(axis=0))
    print
    print('Seq_1')
    print(seq[:,1,:].argmax(axis=1))
    print('Targ1')
    print(targ[1].argmax(axis=0))
    print
    print('Seq_2')
    print(seq[:,2,:].argmax(axis=1))
    print('Targ2')
    print(targ[2].argmax(axis=0))