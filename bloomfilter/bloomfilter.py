import base64
import math
import random
import struct
import timeit
from typing import List, Optional, Union
import uuid

import numba
import numpy as np

class BloomFilter:        
    def __init__(self, data: Optional[bytearray] = None, seeds: Optional[List[int]] = None, arr: Optional[bytearray] = None,
                 size_bytes: Optional[int] = None, num_hashes: Optional[int] = None):
        """
        Recreate a bloomfilter from a list of uint32 seeds and an array of bytes.
        """
        if data is None:
            # Build a single block with seeds and filter array
            if seeds is None and num_hashes is not None:
                seeds = [random.getrandbits(32) for _ in range(num_hashes)]
                #seeds = [random.randint(0, 0xFFFFFFFF) for _ in range(num_hashes)]
            if seeds:
                seeds = np.array(seeds, dtype='uint32')
                num_hashes = len(seeds)
            else:
                raise ValueError("Specify either seeds or num_hashes")

            if arr is not None:
                arr = np.array(arr, dtype='uint8')
                size_bytes = len(arr)
            elif size_bytes is None:
                raise ValueError("Specify either arr or size_bytes")
    
            data = bytearray(4 * (1 + len(seeds)) + size_bytes)
            if arr is None:
                struct.pack_into(f'<I{len(seeds)}I', data, 0, len(seeds), *seeds)
            else:
                struct.pack_into(f'<I{len(seeds)}I{size_bytes}s', data, 0, len(seeds), *seeds, arr)
        # Now unpack it with the arr as a writeable memoryview.
        # This means the full state of the bloomfilter is in self.data.
        self.data = data
        self.num_hashes = int.from_bytes(data[0:4], byteorder='little')
        self.seeds = np.ndarray(self.num_hashes, dtype='uint32', buffer=data[4: 4 * (self.num_hashes + 1)])
        self.arr = memoryview(data)[4 * (self.num_hashes + 1):]

    def __getstate__(self):
        return self.data

    def __setstate__(self, state):
        self.__init__(data=state)
    
    def capacity(self, false_positive_rate: float) -> int:
        "Max number of items in the filter to be within a designated false positive rate"
        bits_in_array = len(self.arr) * 8
        return int(bits_in_array * (math.log(2) ** 2) / abs(math.log(false_positive_rate)))
        
    def false_positive_rate(self, items_in_array: int) -> float:
        #bits_in_array = len(self.arr) * 8
        #math.exp(num_hashes)/math.exp(2) = max(math.floor(math.log2(1 / error_rate)), 1)
        #return math.exp(-1 * bits_in_array / items_in_array * log(math.pow(2, math.log(2))))
        return 0.0
    
    def __repr__(self):
        return f"<BloomFilter size {len(self.arr)} with {len(self.seeds)} seeds. {self.count_bits()} bits set>"
    
    @property
    def base64(self) -> str:
        return base64.b64encode(self.data)
    
    def add_string(self, s: str):
        "Add a string to the bloomfilter"
        return self._add(self.arr, s.encode(), self.seeds)

    def add_bytes(self, b: bytes):
        "Add a bytes string to the bloomfilter array"
        return self._add(self.arr, b, self.seeds)

    def add_uuid(self, u: Union[str, uuid.UUID]):
        "Add a UUID string of format 'xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx' to the bloomfilter array"
        # Simple string conversion is 2.5x faster than via UUID object 
        #return self._add(self.arr, uuid.UUID(uuid_string).bytes, self.seeds)
        return self._add(self.arr, str(u).lower().encode(), self.seeds)
        
    def has_string(self, s: str) -> bool:
        "True if the given string is in the bloomfilter"
        return self._contains(self.arr, s.encode(), self.seeds)

    def has_bytes(self, b: bytes) -> bool:
        "True if the given bytes object is in the bloomfilter"
        return self._contains(self.arr, b, self.seeds)
    
    def has_uuid(self, u: Union[str, uuid.UUID]) -> bool:
        "True if the given UUID string (independent of upper/lower case) is in the bloomfilter"
        # Simple string conversion is 2.5x faster than via UUID object 
        #return self._contains(self.arr, uuid.UUID(u).bytes, self.seeds)
        return self._contains(self.arr, str(u).lower().encode(), self.seeds)
        
    add = add_string
    __contains__ = has_string
    
    def count_bits(self) -> int:
        "Number of bits in the array that are set"
        byte_counts = np.bincount(self.arr, None, 256)
        return self._countbits(byte_counts)

    def clear(self):
        "Empty the bloomfilter array"
        self.arr.fill(0)
    
    def similar_copy(self):
        "A new Bloomfilter with the same size array and same seeds"
        return self.__class__(self.seeds.copy(), np.zero_like(self.arr))

    def copy(self):
        "A copy of this Bloomfilter with the array initialised with the same values"
        return self.__class__(self.seeds.copy(), self.arr.copy())

    @staticmethod
    @numba.njit
    def _add(arr, key, seeds):
        num_bits = len(arr) * 8
        for seed in seeds:
            loc = murmurhash(key, seed) % num_bits
            offset, shift = divmod(loc, 8)
            arr[offset] |= (1 << shift)
        
    @staticmethod
    @numba.njit
    def _contains(arr, key, seeds):
        num_bits = len(arr) * 8
        for seed in seeds:
            loc = murmurhash(key, seed) % num_bits
            offset, shift = divmod(loc, 8)
            if not (arr[offset] & (1 << shift)):
                return False
        return True

    @staticmethod
    @numba.njit
    def _countbits(byte_counts) -> int:
        num_bits = 0
        for i, cnt in enumerate(byte_counts):
            if cnt > 0:
                while i:
                    i &= i - 1
                    num_bits += cnt
        return num_bits
    

@numba.njit
def murmurhash(key, seed) -> int:
    """
    Numba-accelerated 32-bit murmurhash.
    """
    length = len(key)
    n, t = divmod(length, 4)
    
    h = seed
    c1 = 0xcc9e2d51
    c2 = 0x1b873593

    # Process whole blocks of 4 bytes
    for i in range(n):
        k1 = (key[4*i] << 24) + (key[4*i + 1] << 16) + (key[4*i + 2] << 8) + key[4*i + 3]  
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF  # ROTL32
        h ^= (k1 * c2) & 0xFFFFFFFF
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF     # ROTL32
        h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF

    # Process tail of 1-3 bytes if present
    if t > 0:
        k1 = (key[4*n] << 16)
        if t > 1:
            k1 += key[4*n + 1] << 8
        if t > 2:
            k1 += key[4*n + 2]
        k1  = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF # ROTL32
        k1 = (k1 * c2) & 0xFFFFFFFF
        h ^= k1

    h ^= length   # Include length to give different values for 1-3 tails of \0 bytes

    # Finalise by mixing the bits
    x = h
    x ^= (x >> 16)
    x = (x * 0x85ebca6b) & 0xFFFFFFFF
    x ^= (x >> 13)
    x = (x * 0xc2b2ae35) & 0xFFFFFFFF
    x ^= (x >> 16)
    return x


def test_bloomfilter():
    s = 'hello'
    b = b'hello'
    b2 = b'hello1'
    u     = '12345678-1234-1234-1234-abcdefABCDEF'
    is_u  = '12345678-1234-1234-1234-aBcDeFaBcDeF'
    not_u = '02345678-1234-1234-1234-abcdefABCDEF'

    bf = BloomFilter(num_hashes=6, size_bytes=1000)
    print(repr(bf))
    bf.add_string(s)
    bf.add_bytes(b)
    bf.add_uuid(u)
    bf.add_uuid(is_u)
    res = bf.has_string(s), bf.has_bytes(b), bf.has_bytes(b2), bf.has_uuid(u), bf.has_uuid(is_u), bf.has_uuid(not_u)
    assert res == (True, True, False, True, True, False)
    
    bf = BloomFilter(num_hashes=6, size_bytes=50)
    bf.add('hello')
    print(bf.base64)
    
    statements = [
        'bf.add_string(s)', 
        'bf.add_bytes(b)', 
        'bf.add_uuid(u)', 
        'bf.has_string(s)', 
        'bf.has_bytes(b)', 
        'bf.has_uuid(u)', 
        'bf.has_uuid(is_u)', 
        'bf.has_uuid(not_u)',
    ]

    # Timing of common operations
    # Approx 1/3 speed of pybloomfiltermmap3, which is written in C and Cython.
    # Which shows how good numba is!
    bf = BloomFilter(num_hashes=6, size_bytes=100000)
    variables = dict(bf=bf, s=s, b=b, u=u, is_u=is_u, not_u=not_u)
    for stmt in statements:
        timeit.timeit(stmt, number=10, globals=variables) # Don't count JIT time first time through
        secs = timeit.timeit(stmt, number=10000, globals=variables)
        print(f"{stmt:30} -> {secs * 100:0.3f} µs")