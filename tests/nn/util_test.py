# pylint: disable=invalid-name,no-self-use,too-many-public-methods
import numpy
import torch
from allennlp.common.testing import AllenNlpTestCase
from numpy.testing import assert_array_almost_equal
from torch.autograd import Variable

from scitail.nn import util


class TestNnUtil(AllenNlpTestCase):
    def test_masked_mean_no_mask(self):
        # Testing the general unmasked 1D case.
        vector_1d = Variable(torch.FloatTensor([[1.0, 2.0, 3.0]]))
        dim = 1
        vector_1d_mean = util.masked_mean(vector_1d, dim, None).data.numpy()
        assert_array_almost_equal(vector_1d_mean,
                                  numpy.array([2.0]))

        # Testing the unmasked 1D case where the input is all 0s.
        vector_zero = Variable(torch.FloatTensor([[0.0, 0.0, 0.0]]))
        vector_zero_mean = util.masked_mean(vector_zero, dim, None).data.numpy()
        assert_array_almost_equal(vector_zero_mean,
                                  numpy.array([0.0]))

        # Testing the unmasked batched case where one of the inputs are all 0s.
        matrix = Variable(torch.FloatTensor([[1.0, 2.0, 5.0], [0.0, 0.0, 0.0]]))
        masked_matrix_mean = util.masked_mean(matrix, dim, None).data.numpy()
        assert_array_almost_equal(masked_matrix_mean,
                                  numpy.array([2.666666, 0.0]))

    def test_masked_softmax_masked(self):
        # Testing the general masked 1D case.
        vector_1d = Variable(torch.FloatTensor([[1.0, 2.0, 5.0]]))
        mask_1d = Variable(torch.FloatTensor([[1.0, 0.0, 1.0]]))
        dim = 1
        vector_1d_mean = util.masked_mean(vector_1d, dim, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_mean,
                                  numpy.array([3.0]))

        vector_1d = Variable(torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]]))
        mask_1d = Variable(torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]]))
        vector_1d_mean = util.masked_mean(vector_1d, dim, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_mean,
                                  numpy.array([2.333333]))

        # Testing the masked 1D case where the input is all 0s and the mask
        # is not all 0s.
        vector_1d = Variable(torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]]))
        mask_1d = Variable(torch.FloatTensor([[0.0, 0.0, 0.0, 1.0]]))
        vector_1d_mean = util.masked_mean(vector_1d, dim, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_mean,
                                  numpy.array([0]))

        # Testing the masked 1D case where the input is not all 0s
        # and the mask is all 0s.
        vector_1d = Variable(torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]]))
        mask_1d = Variable(torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]]))
        vector_1d_mean = util.masked_mean(vector_1d, dim, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_mean,
                                  numpy.array([0.0]))

        # Testing the masked 1D case where the input is all 0s and
        # the mask is all 0s.
        vector_1d = Variable(torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]]))
        mask_1d = Variable(torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]]))
        vector_1d_mean = util.masked_mean(vector_1d, dim, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_mean,
                                  numpy.array([0.0]))

        # Testing the general masked batched case.
        matrix = Variable(torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]]))
        mask = Variable(torch.FloatTensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))
        masked_matrix_mean = util.masked_mean(matrix, 1, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_mean,
                                  numpy.array([3.0, 2.0]))
        masked_matrix_mean = util.masked_mean(matrix, 0, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_mean,
                                  numpy.array([1.0, 2.0, 4.0]))

        # Testing the masked batch case where one of the inputs is all 0s but
        # none of the masks are all 0.
        matrix = Variable(torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        mask = Variable(torch.FloatTensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))
        masked_matrix_mean = util.masked_mean(matrix, 1, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_mean,
                                  numpy.array([0.0, 2.0]))
        masked_matrix_mean = util.masked_mean(matrix, 0, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_mean,
                              numpy.array([0.5, 2.0, 1.5]))

        # Testing the masked batch case where one of the inputs is all 0s and
        # one of the masks are all 0.
        matrix = Variable(torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        mask = Variable(torch.FloatTensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
        masked_matrix_mean = util.masked_mean(matrix, 1, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_mean,
                                  numpy.array([0.0, 0.0]))
        masked_matrix_mean = util.masked_mean(matrix, 0, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_mean,
                              numpy.array([0.0, 0.0, 0.0]))

        matrix = Variable(torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        mask = Variable(torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]))
        masked_matrix_mean = util.masked_mean(matrix, 1, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_mean,
                                  numpy.array([0.0, 2.0]))
        masked_matrix_mean = util.masked_mean(matrix, 0, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_mean,
                              numpy.array([1.0, 0.0, 3.0]))
