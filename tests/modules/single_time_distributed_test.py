import torch
from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.matrix_attention import MatrixAttention
from numpy.testing import assert_almost_equal
from torch.autograd import Variable

from scitail.modules.single_time_distributed import SingleTimeDistributed


class TestTimeDistributed(AllenNlpTestCase):
    def test_single_time_distributed_reshapes_correctly(self):
        matrix_attention = MatrixAttention()
        distributed_embedding = SingleTimeDistributed(matrix_attention, 0)
        input_matrix1 = Variable(torch.rand([2, 3, 5, 4]))
        input_matrix2 = Variable(torch.rand([2, 6, 4]))
        output = distributed_embedding(input_matrix1, input_matrix2)
        assert output.size() == torch.Size([2, 3, 5, 6])

        distributed_embedding = SingleTimeDistributed(matrix_attention, 1)
        output = distributed_embedding(input_matrix2, input_matrix1)
        assert output.size() == torch.Size([2, 3, 6, 5])

    def test_single_time_distributed_computes_correctly(self):
        matrix_attention = MatrixAttention()
        distributed_embedding = SingleTimeDistributed(matrix_attention, 0)
        input_matrix1 = Variable(torch.rand([2, 3, 5, 4]))
        input_matrix2 = Variable(torch.rand([2, 6, 4]))
        output = distributed_embedding(input_matrix1, input_matrix2)
        # Check the value in output[i, j, k, l] == similarity(input_matrix1[i, j, k],
        # input_matrix[i, l])
        for batch in range(output.size()[0]):
            for distributed in range(output.size()[1]):
                for row in range(output.size()[2]):
                    for col in range(output.size()[3]):
                        assert_almost_equal(output[batch, distributed, row, col].data.numpy(),
                                            matrix_attention._similarity_function(
                                                input_matrix1[batch, distributed, row].data,
                                                input_matrix2[batch, col].data).numpy())
