import numpy as np
# 1
lst = [[1, 3, 5], [2, 4, 6]]
print(type(lst))
np_lst = np.array(lst)
print(type(np_lst))
np_lst = np.array(lst, dtype=np.float)
print(np_lst.shape)
print(np_lst.ndim)
print(np_lst.dtype)
print(np_lst.itemsize)
print(np_lst.size)
# 2
print(np.zeros([2, 4]))
print(np.ones([3, 5]))
print("Rand:")
print(np.random.rand(2, 4))
print(np.random.rand())
print("RandInt:")
print(np.random.randint(1, 10, 3))
print("Randn:")
print(np.random.randn(2, 4))
print("Choice:")
print(np.random.choice([10, 20, 3, 4, 56]))
print("Distribute:")
print(np.random.beta(1, 10, 100))
# 3
lst = np.arange(1, 11).reshape((2, -1))
print(lst)
print("Exp:")
print(np.exp(lst))
print("Exp2:")
print(np.exp2(lst))
print("Sqrt:")
print(np.sqrt(lst))
print("Sin:")
print(np.sin(lst))
print("Log:")
print(np.log(lst))

lst = np.array([[[1, 2, 3, 4],
                 [5, 6, 7, 8]],
                [[9, 10, 11, 12],
                 [13, 14, 15, 16]],
                [[17, 18, 19, 20],
                 [21, 22, 23, 24]],
                ])
print(lst.sum(axis=2))
print(lst.max(axis=1))
print(lst.min(axis=0))

lst1 = np.array([10, 20, 30, 40])
lst2 = np.array([4, 3, 21, 1])
print("Add:")
print(lst1+lst2)
print("Sub:")
print(lst1-lst2)
print("Mul:")
print(lst1*lst2)
print("Div")
print(lst1/lst2)
print("Square")
print(lst1**2)
print("Dot:")
print(np.dot(lst1.reshape([2, 2]), lst2.reshape([2, 2])))
print("Cancatenate")
print(np.concatenate((lst1, lst2), axis=0))
print(np.vstack((lst1, lst2)))
print(np.hstack((lst1, lst2)))
print(np.split(lst1, 4))
print(np.copy(lst1))
# 4
from numpy.linalg import *
print(np.eye(3))
lst = np.array([[1, 2], [3, 4]])
print("Inv:")
print(inv(lst))
print("T：")
print(lst.transpose())
print("Det:")
print(det(lst))
print("Eig:")
print(eig(lst))
y = np.array([[5.], [7.]])
print("Solve:")
print(solve(lst, y))
# 5
print("FFT:")