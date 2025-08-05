# coding: utf-8
from binascii import hexlify
from itertools import combinations
from CompactFIPS202 import SHA3_256, SHAKE256

class Vision:
    def __init__( self ):
        self.security_level = 128
        self.n = 512
        self.m = 4
        # self.rate = floor(m/2)
        # self.capacity = m - self.rate
        self.Nb = max(10, 2*ceil((1.0 * self.security_level + self.m + 8) / (8*self.m)))

        # Choose field to be GHASH, GF(2)[X] / (X^128 + X^7 + X^2 + X + 1)
        GF2 = FiniteField(2)
        GF2x.<y> = PolynomialRing(GF2, "y")
        F.<x> = FiniteField(2^128, "x", modulus=y^128 + y^7 + y^2 + y + 1)
        self.F = F

        # Choose MDS matrix to have the same structure as the AES MixColumns matrix
        self.MDS = Matrix(
            self.F,
            [
                [x, x + 1, 1, 1], # 2, 3, 1, 1
                [1, x, x + 1, 1], # 1, 2, 3, 1
                [1, 1, x, x + 1], # 1, 1, 2, 3
                [x + 1, 1, 1, x], # 3, 1, 1, 2
            ]
        )

        print("Testing MDS property of the matrix...")
        assert Vision.is_mds_matrix(self.MDS)

        # Sample linearized polynomial coefficients pseudo-randomly
        self.B, self.Binv, self.initial_constant, self.constants_matrix, self.constants_constant = Vision.sample_parameters(self.F, self.m)

    # get the parameters B, Binv, and the constants; extract them
    # from SHAKE seeded with a memorable string
    @staticmethod
    def sample_parameters( F, m ):
        seed = "biniusiscoming"
        seed = bytearray(seed.encode('utf-8'))

        buf_len = 1 # bytes
        have_params = False
        while have_params == False:
            buf_len = buf_len * 2
            randomness = SHAKE256(seed, buf_len)
            try:
                constants = Vision.sample_constants(F, m, randomness)
                break
            except BufferError:
                continue

        B = constants[0:4]
        Binv = Vision.affine_inverse(B)
        initial_constant = matrix([[constants[4+i]] for i in range(0, m)])
        constants_matrix = matrix([[constants[4+m+i*m+j] for j in range(0,m)] for i in range(0,m)])
        constants_constant = matrix([[constants[4+m+m^2+i]] for i in range(0,m)])
        return B, Binv, initial_constant, constants_matrix, constants_constant

    # sample the given number of constants randomly from the given
    # field using the randomness provided; if there is not enough
    # randomness, throw a buffer error
    @staticmethod
    def sample_constants( F, m, randomness ):
        counter = 0
        constants = []
        chunk_size = ceil(F.modulus().degree() / 8 )
        while len(constants) != 4 + 2*m + m^2:
            have_constant = False
            while have_constant == False:
                if counter+chunk_size > len(randomness):
                    raise BufferError
                constant = Vision.field_element_from_bytes(F, randomness[counter:(counter+chunk_size)])
                counter += chunk_size

                # only if the sampled constant is not in any
                # subfields do we accept it
                if Vision.is_generator(F, constant):
                    constants.append(constant)
                    break

            # if constants 1,2,3 do not define an invertible
            # linearized polynomial, reject them
            if len(constants) == 4:
                if not Vision.linearized_is_invertible(constants[1:]):
                    constants = [constants[0]]

            # if the m^2 constants after the affine polynomial B do
            # define an invertible matrix, reject them
            if len(constants) == 4 + m^2:
                mat = matrix([[constants[4+i*m+j] for j in range(0,m)] for i in range(0,m)])
                if not mat.is_invertible():
                    constants = constants[0:4]
        return constants

    # determine if the given list of coefficients represents an
    # invertible linearized transform
    @staticmethod
    def linearized_is_invertible( coefficients ):
        F = coefficients[0].parent()
        deg = F.modulus().degree()
        mat = copy(MatrixSpace(F, deg, deg, sparse=True).zero())
        for i in range(0, min(deg, len(coefficients))):
            mat[0,i] = coefficients[i]
        for i in range(1, deg):
            for j in range(1, deg):
                mat[i,j] = mat[i-1,j-1]
            mat[i,0] = mat[i-1,deg-1]
        for i in range(1, deg):
            for j in range(0, deg):
                mat[i,j] = mat[i,j]^(2^i)
        return mat.is_invertible()

    # turn a string of bytes into an element from the given field;
    # assuming there are enough bytes
    @staticmethod
    def field_element_from_bytes( F, randomness ):
        x = F.gen()
        integer = sum(randomness[i] * 256^i for i in range(0, len(randomness)))
        bits = bin(integer)[2:]
        element = sum(x^i for i in range(0, min(F.modulus().degree(), len(bits))) if bits[i] == '1')
        return element

    # decide if a given element generates the entire field (and not
    # just a subfield)
    @staticmethod
    def is_generator( F, constant ):
        deg = F.modulus().degree()
        factorization = factor(deg)
        # for all possible subfield orders, try raising the power order
        for fact, multiplicity in factorization:
            subfield_power = int(2^(deg/fact))
            if constant^subfield_power == constant: # if that power sends the element to itself, it generates only the subfield
                return False
        # all tests succeed
        return True

    # get the inverse of an affine transform, as represented by a
    # list of coefficients
    @staticmethod
    def affine_inverse( B ):
        F = B[0].parent()
        deg = F.modulus().degree()
        MS = MatrixSpace(F, deg, deg)
        mat = copy(MS.zero())
        mat[0,0] = B[1]
        mat[0,1] = B[2]
        mat[0,2] = B[3]

        for i in range(1,deg):
            for j in range(1, deg):
                mat[i,j] = mat[i-1,j-1]
            mat[i,0] = mat[i-1,deg-1]
        for i in range(1, deg):
            for j in range(0, deg):
                mat[i,j] = mat[i,j]^(2^i)
        mat = mat.inverse()
        coefficients = [mat[0,i] for i in range(0,deg)]
        constant = sum(coefficients[i] * B[0]^(2^i) for i in range(0, deg))
        return [constant] + coefficients

    # test if a matrix is Maximum Distance Separable (MDS)
    # by checking that all square submatrices are non-singular
    @staticmethod
    def is_mds_matrix( matrix ):
        """
        Test if a matrix is Maximum Distance Separable (MDS).

        A matrix is MDS if every square submatrix is non-singular (has non-zero determinant).
        This function uses brute force to check all possible square submatrices.

        Args:
            matrix: A matrix to test

        Returns:
            True if the matrix is MDS, False otherwise
        """
        rows = matrix.nrows()
        cols = matrix.ncols()
        max_size = min(rows, cols)
        print(matrix)

        # Check all square submatrices of all sizes
        for size in range(1, max_size + 1):
            # Get all combinations of row indices
            for row_indices in combinations(range(rows), size):
                # Get all combinations of column indices
                for col_indices in combinations(range(cols), size):
                    # Extract the submatrix
                    submatrix = matrix[list(row_indices), list(col_indices)]

                    # Check if determinant is zero (matrix is singular)
                    if size == 1:
                        # For 1x1 matrices, just check if element is non-zero
                        if submatrix[0, 0] == 0:
                            print(row_indices, col_indices)
                            return False
                    else:
                        det = submatrix.determinant()
                        if det == 0:
                            print(row_indices, col_indices)
                            return False

        return True

    # print parameters
    def Parameters( self ):
        print("# Parameters")
        print("security level:", self.security_level)
        print("nm: %i \t m: %i \t Nb: %i" % (self.n, self.m, self.Nb))
        # print"rate:", self.rate, " \t capacity:", self.capacity
        print("field:", self.F, "... with modulus:", self.F.modulus())
        print("B[0]:", self.B[0])
        print("B[1]:", self.B[1])
        print("B[2]:", self.B[2])
        print("B[3]:", self.B[3])
        print("initial constant:")
        print(self.initial_constant)
        print("constants matrix:")
        print(self.constants_matrix)
        print("constants constant:")
        print(self.constants_constant)

    # evaluate the block cipher
    def BlockCipher( self, key, ptxt ):
        key_state = key
        data_state = ptxt

        key_injection = self.initial_constant
        key_state += key_injection
        data_state += key_state
        for r in range(0, 2*self.Nb):
            # constants
            key_injection = self.constants_matrix * key_injection + self.constants_constant

            # key schedule
            key_state = matrix([[key_state[i,0]^int(2^(self.n/self.m)-2)] for i in range(0,self.m)])
            if r % 2 == 0:
                key_state = matrix([[self.Binv[0] + sum(key_state[i,0]^(2^(j-1)) * self.Binv[j] for j in range(1, len(self.Binv)))] for i in range(0, self.m)])
            else:
                key_state = matrix([[self.B[0] + sum(key_state[i,0]^(2^(j-1)) * self.B[j] for j in range(1, len(self.B)))] for i in range(0, self.m)])
            key_state = self.MDS * key_state
            key_state += key_injection

            # data path
            data_state = matrix([[data_state[i,0]^int(2^(self.n/self.m)-2)] for i in range(0,self.m)])
            if r % 2 == 0:
                data_state = matrix([[self.Binv[0] + sum(data_state[i,0]^(2^(j-1)) * self.Binv[j] for j in range(1, len(self.Binv)))] for i in range(0, self.m)])
            else:
                data_state = matrix([[self.B[0] + sum(data_state[i,0]^(2^(j-1)) * self.B[j] for j in range(1, len(self.B)))] for i in range(0, self.m)])
            data_state = self.MDS * data_state
            data_state += key_state

        return data_state

    # evaluate the sponge function
    def Sponge( self, inputs, rate ):
        key = matrix(self.F, [[self.F.zero()]] * self.m)
        state = matrix(self.F, [[self.F.zero()]] * self.m)
        for i in range(0, len(inputs), rate):
            for j in range(min(rate, len(inputs)-i)):
                state[j,0] += inputs[j+i]
            state = self.BlockCipher(key, state)

        outputs = []
        for i in range(0, rate):
            outputs.append(state[i%rate,0])

        return outputs


vision = Vision()
print("Vision instance: ")
vision.Parameters()
