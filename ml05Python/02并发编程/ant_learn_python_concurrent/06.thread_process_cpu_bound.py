import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import multiprocessing

PRIMES = [1122725350952] * 10

def is_prime(n):
    print("Process ID:", multiprocessing.current_process().pid)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def single_thread():
    for number in PRIMES:
        is_prime(number)


def multi_thread():
    with ThreadPoolExecutor() as pool:
        pool.map(is_prime, PRIMES)

def multi_process_():
    with ProcessPoolExecutor() as pool:
        pool.map(is_prime, PRIMES)

def multi_process():
    pool = ProcessPoolExecutor()
    results = pool.map(is_prime, PRIMES)
    for result in results:
        print(result)


if __name__ == "__main__":
    start = time.time()
    single_thread()
    end = time.time()
    print("single_thread, cost:", end - start, "seconds")

    start = time.time()
    multi_thread()
    end = time.time()
    print("multi_thread, cost:", end - start, "seconds")

    start = time.time()
    multi_process()
    end = time.time()
    print("multi_process, cost:", end - start, "seconds")











