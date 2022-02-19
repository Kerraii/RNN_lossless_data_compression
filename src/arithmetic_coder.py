# Code based on the Github repo below, but modified to improve speed and ease of use. Modifications include:
# - Removed frequency table classes in favor of numpy arrays
# - Included bit stream functionality in encoder/decoder classes instead of using an external class
# - Removed asserts, unnecessary functions and configuration fields that only got used in assertions
# - General code cleanup
# - Added default parameters to some functions
# - Added context manager for convenience
#
# Reference arithmetic coding
# Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
# 

import numpy as np
from contextlib import contextmanager

# ---- Arithmetic coding core classes ----

# Provides the state and behaviors that arithmetic coding encoders and decoders share.
class ArithmeticCoderBase:

    # Constructs an arithmetic coder, which initializes the code range.
    def __init__(self, numbits):

        # -- Configuration fields --
        # Number of bits for the 'low' and 'high' state variables. Must be at least 1.
        # - Larger values are generally better
        self.num_state_bits = numbits
        # Maximum range (high+1-low) during coding (trivial), which is 2^num_state_bits = 1000...000.
        self.full_range = 1 << self.num_state_bits
        # The top bit at width num_state_bits, which is 0100...000.
        self.half_range = self.full_range >> 1  # Non-zero
        # The second highest bit at width num_state_bits, which is 0010...000. This is zero when num_state_bits=1.
        self.quarter_range = self.half_range >> 1  # Can be zero
        # Bit mask of num_state_bits ones, which is 0111...111.
        self.state_mask = self.full_range - 1

        # -- State fields --
        # Low end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 0s.
        self.low = 0
        # High end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 1s.
        self.high = self.state_mask


    # Updates the code range (low and high) of this arithmetic coder as a result
    # of processing the given symbol with the given frequency table.
    def update(self, intervals, symbol):
        # State check
        low = self.low
        high = self.high
        range = high - low + 1

        # Frequency table values check
        total = intervals.item(-1)
        symlow = intervals.item(symbol)
        symhigh = intervals.item(symbol+1)

        # Update range
        newlow  = low + symlow  * range // total
        newhigh = low + symhigh * range // total - 1
        self.low = newlow
        self.high = newhigh

        # While low and high have the same top bit value, shift them out
        while ((self.low ^ self.high) & self.half_range) == 0:
            self.shift()
            self.low  = ((self.low  << 1) & self.state_mask)
            self.high = ((self.high << 1) & self.state_mask) | 1
        # Now low's top bit must be 0 and high's top bit must be 1

        # While low's top two bits are 01 and high's are 10, delete the second highest bit of both
        while (self.low & ~self.high & self.quarter_range) != 0:
            self.underflow()
            self.low = (self.low << 1) ^ self.half_range
            self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1


    # Called to handle the situation when the top bit of 'low' and 'high' are equal.
    def shift(self):
        raise NotImplementedError()


    # Called to handle the situation when low=01(...) and high=10(...).
    def underflow(self):
        raise NotImplementedError()


    def probs_to_intervals(self, probs):
        intervals = np.zeros(probs.size+1, dtype=np.uint64)
        intervals[1:] = np.cumsum(probs*10000000 + 1)
        return intervals



# Encodes symbols and writes to an arithmetic-coded bit stream.
class ArithmeticEncoder(ArithmeticCoderBase):

    # Constructs an arithmetic coding encoder based on the given bit output stream.
    def __init__(self, outputfile, numbits=32):
        super(ArithmeticEncoder, self).__init__(numbits)
        # Number of saved underflow bits. This value can grow without bound.
        self.num_underflow = 0
        # The underlying byte stream to write to
        self.output = open(outputfile, "wb")
        # The accumulated bits for the current byte, always in the range [0x00, 0xFF]
        self.currentbyte = 0
        # Number of accumulated bits in the current byte, always between 0 and 7 (inclusive)
        self.numbitsfilled = 0


    # Encodes the given symbol based on the given frequency table.
    # This updates this arithmetic coder's state and may write out some bits.
    def encode_symbol(self, probs, symbol):
        intervals = self.probs_to_intervals(probs)
        self.update(intervals, symbol)


    # Terminates the arithmetic coding by flushing any buffered bits, so that the output can be decoded properly.
    # It is important that this method must be called at the end of the each encoding process.
    def finish(self):
        self.write(1)
        self.close()


    def shift(self):
        bit = self.low >> (self.num_state_bits - 1)
        self.write(bit)

        # Write out the saved underflow bits
        for _ in range(self.num_underflow):
            self.write(bit ^ 1)
        self.num_underflow = 0


    def underflow(self):
        self.num_underflow += 1


    # Writes a bit to the stream. The given bit must be 0 or 1.
    def write(self, b):
        self.currentbyte = (self.currentbyte << 1) | b
        self.numbitsfilled += 1
        if self.numbitsfilled == 8:
            towrite = bytes((self.currentbyte,))
            self.output.write(towrite)
            self.currentbyte = 0
            self.numbitsfilled = 0


    # Closes this stream and the underlying output stream. If called when this
    # bit stream is not at a byte boundary, then the minimum number of "0" bits
    # (between 0 and 7 of them) are written as padding to reach the next byte boundary.
    def close(self):
        while self.numbitsfilled != 0:
            self.write(0)
        self.output.close()


# Reads from an arithmetic-coded bit stream and decodes symbols.
class ArithmeticDecoder(ArithmeticCoderBase):

    # Constructs an arithmetic coding decoder based on the
    # given bit input stream, and fills the code bits.
    def __init__(self, inputfile, numbits=32):
        super(ArithmeticDecoder, self).__init__(numbits)
        # The underlying byte stream to read from
        self.input = open(inputfile, "rb")
        # Either in the range [0x00, 0xFF] if bits are available, or -1 if end of stream is reached
        self.currentbyte = 0
        # Number of remaining bits in the current byte, always between 0 and 7 (inclusive)
        self.numbitsremaining = 0
        # The current raw code bits being buffered, which is always in the range [low, high].
        self.code = 0
        for _ in range(self.num_state_bits):
            self.code = self.code << 1 | self.read_code_bit()


    # Decodes the next symbol based on the given frequency table and returns it.
    # Also updates this arithmetic coder's state and may read in some bits.
    def decode_symbol(self, probs):
        # Translate from coding range scale to frequency table scale
        intervals = self.probs_to_intervals(probs)
        total = intervals.item(-1)
        range = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range

        # A kind of binary search. Find highest symbol such that freqs.get_low(symbol) <= value.
        start = 0
        end = probs.size
        while end - start > 1:
            middle = (start + end) >> 1
            if intervals[middle] > value:
                end = middle
            else:
                start = middle

        symbol = start
        self.update(intervals, symbol)
        return symbol


    def finish(self):
        self.close()


    def shift(self):
        self.code = ((self.code << 1) & self.state_mask) | self.read_code_bit()


    def underflow(self):
        self.code = (self.code & self.half_range) | ((self.code << 1) & (self.state_mask >> 1)) | self.read_code_bit()


    # Returns the next bit (0 or 1) from the input stream. The end
    # of stream is treated as an infinite number of trailing zeros.
    def read_code_bit(self):
        temp = self.read()
        if temp == -1:
            temp = 0
        return temp


    # Reads a bit from this stream. Returns 0 or 1 if a bit is available, or -1 if
    # the end of stream is reached. The end of stream always occurs on a byte boundary.
    def read(self):
        if self.currentbyte == -1:
            return -1
        if self.numbitsremaining == 0:
            temp = self.input.read(1)
            if len(temp) == 0:
                self.currentbyte = -1
                return -1
            self.currentbyte = temp[0]
            self.numbitsremaining = 8
        assert self.numbitsremaining > 0
        self.numbitsremaining -= 1
        return (self.currentbyte >> self.numbitsremaining) & 1


    # Closes this stream and the underlying input stream.
    def close(self):
        self.input.close()
        self.currentbyte = -1
        self.numbitsremaining = 0


@contextmanager
def finishing(coder):
    try:
        yield coder
    finally:
        coder.finish()