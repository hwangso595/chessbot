import concurrent.futures
from time import time
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool


def f(x):
    return x * x


def inc(inc, test):
    try:
        print(test)
        test.increment(inc)
        return test.num
    except TypeError as err:
        print(err)


class Test:
    def __init__(self):
        self.num = 0

    def increment(self, inc):
        self.num += inc
        return self.num


def speed1():
    s = time()
    a = np.array([1,2,3,4,5]*100)
    b = np.array([1,2,3,4,5]*100)
    for i in range(25000):
        a = np.append(a,b,0)
    print(time()-s)


def speed2():
    s = time()
    a = [1,2,3,4,5]*100
    b = [1, 2, 3, 4, 5]*100
    for i in range(25000):
        a.extend(b)
    print(time() - s)

if __name__ == '__main__':
    incs = [5, 4, 3, 2, 1]
    # speed1()
    # speed2()
    test = Test()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = []
        for i in range(len(incs)):
            future.append(executor.submit(test.increment, i))
        print(list(map(lambda x: x.result(), future)))
    # pool = ThreadPool(4)
    # print(pool.map(test.increment, incs))
    print(test.num)
