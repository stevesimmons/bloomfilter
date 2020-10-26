# To be completed
import pickle

from bloomfilter import BloomFilter


def test():
    bf = BloomFilter(num_hashes=10, size_bytes=100)
    bf.add('hello')
    s = pickle.dumps(bf)

    bf2 = pickle.loads(s)
    assert 'hi' not in bf2
    assert 'hello' in bf2
    assert (bf.seeds == bf2.seeds).all()