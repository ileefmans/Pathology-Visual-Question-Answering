import unittest
from preprocess import image_process
from PIL import Image
import requests


class test_image_process(unittest.TestCase):
    
    def setUp(self):
        response1 = requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/LACMTA_Square_Orange_Line.svg/1200px-LACMTA_Square_Orange_Line.svg.png", stream=True).raw
        
        response2 = requests.get("http://www.math.uwaterloo.ca/~hdesterc/websiteW/personal/pictures/argentina2003/200311-set8/images/200311-set8_4_400x600.jpg", stream=True).raw
        #response3 = requests.get("insert path here")
    
        self.image1 = Image.open(response1)
        self.image2 = Image.open(response2)
        #self.image3 = Image.open(BytesIO(response3.content))
    
    
    
    def tearDown(self):
        pass

    def test_expand_square(self):

        self.assertEqual(image_process((400,600)).expand(self.image1).size, (400,400))
        self.assertEqual(image_process((600,600)).expand(self.image1).size, (600,600))
        self.assertEqual(image_process((600,400)).expand(self.image1).size, (400,400))
    


    def test_expand_tall_rectangle(self):

        self.assertEqual(image_process((300,700)).expand(self.image2).height, 300)
        self.assertEqual(image_process((700,700)).expand(self.image1).height, 700)
        self.assertEqual(image_process((700,300)).expand(self.image1).height, 300)










if __name__ == '__main__':
    unittest.main()





