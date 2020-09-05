import unittest
import torch
from pretrain_VAE import VAE


class test_image_process(unittest.TestCase):
    
    def setUp(self):
        self.x = torch.rand(4,3,491,600)

    
    
    def tearDown(self):
        pass


    def test_VAE_output_size(self):
        self.assertEqual(VAE(16).forward(self.x)[0].size(), torch.Size([4,3,491,600]))
        print("DONE PRETRAIN TESTS")
    






if __name__ == '__main__':
    unittest.main()





