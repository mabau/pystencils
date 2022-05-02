from pystencils.cache import sharedmethodcache


class Fib:

    def __init__(self):
        self.fib_rec_called = 0
        self.fib_iter_called = 0

    @sharedmethodcache("fib_cache")
    def fib_rec(self, n):
        self.fib_rec_called += 1
        return 1 if n <= 1 else self.fib_rec(n-1) + self.fib_rec(n-2)

    @sharedmethodcache("fib_cache")
    def fib_iter(self, n):
        self.fib_iter_called += 1
        f1, f2 = 0, 1
        for i in range(n):
            f2 = f1 + f2
            f1 = f2 - f1
        return f2


def test_fib_memoization_1():
    fib = Fib()

    assert "fib_cache" not in fib.__dict__

    f13 = fib.fib_rec(13)
    assert fib.fib_rec_called == 14
    assert "fib_cache" in fib.__dict__
    assert fib.fib_cache[(13,)] == f13

    for k in range(14):
        #   fib_iter should use cached results from fib_rec
        fib.fib_iter(k)
    
    assert fib.fib_iter_called == 0


def test_fib_memoization_2():
    fib = Fib()

    f11 = fib.fib_iter(11)
    f12 = fib.fib_iter(12)

    assert fib.fib_iter_called == 2

    f13 = fib.fib_rec(13)

    #   recursive calls should be cached
    assert fib.fib_rec_called == 1


class Triad:
    
    def __init__(self):
        self.triad_called = 0

    @sharedmethodcache("triad_cache")
    def triad(self, a, b, c=0):
        """Computes the triad a*b+c."""
        self.triad_called += 1
        return a * b + c


def test_triad_memoization():
    triad = Triad()

    assert triad.triad.__doc__ == "Computes the triad a*b+c."

    t = triad.triad(12, 4, 15)
    assert triad.triad_called == 1
    assert triad.triad_cache[(12, 4, 15)] == t

    t = triad.triad(12, 4, c=15)
    assert triad.triad_called == 2
    assert triad.triad_cache[(12, 4, 'c', 15)] == t

    t = triad.triad(12, 4, 15)
    assert triad.triad_called == 2

    t = triad.triad(12, 4, c=15)
    assert triad.triad_called == 2
