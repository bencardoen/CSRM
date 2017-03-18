import logging

logger = logging.getLogger('global')

class DistributeSpreadPolicy:
    def spread(buf, n):
        """
        Divide buffer over n parts, with the last part taking a remainder.
        If len buffer <= n, return n instances of buffer
        """
        if not n:
            raise ValueError("n is zero")
        bl = len(buf)
        d, m = divmod(bl, n)
        if d:
            slicesize = round(bl/n)
            bf =  [buf[ i*slicesize : (i+1)*slicesize] if i!=n-1 else buf[(n-1)*slicesize:] for i in range(n)]
            assert(len(bf) == n)
            return bf
        else:
            return [buf for _ in range(n)] # list of 3 references, but the contents are copied in any case

class CopySpreadPolicy:
    def spread(buf, n):
        return [buf for _ in range(n)]
