import scipy.io as sio
test = sio.loadmat("../SynthText/SynthText/gt.mat")
print(type(test))
print(test.items())