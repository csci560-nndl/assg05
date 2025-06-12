import numpy as np
import pandas as pd
import sklearn
#import unittest
from twisted.trial import unittest
#from assg_tasks import calc_new_height_width
#from assg_tasks import pool_forward
#from assg_tasks import zero_pad
#from assg_tasks import conv_single_step
#from assg_tasks import conv_forward


class test_calc_new_height_width(unittest.TestCase):

    def setUp(self):
        pass

    def test_basic_case(self):
        new_height, new_width = calc_new_height_width(5, 5, 3, 1, 0)
        self.assertIsInstance(new_height, int)
        self.assertIsInstance(new_width, int)
        self.assertEqual((new_height, new_width), (3, 3))

    def test_other_example_case(self):
        new_height, new_width = calc_new_height_width(12, 21, 4, 2, 2)
        self.assertIsInstance(new_height, int)
        self.assertIsInstance(new_width, int)
        self.assertEqual((new_height, new_width), (7, 11))

    def test_big_example(self):
        new_height, new_width = calc_new_height_width(1024, 768, 7, 3, 4)
        self.assertIsInstance(new_height, int)
        self.assertIsInstance(new_width, int)
        self.assertEqual((new_height, new_width), (342, 257))


class test_pool_forward(unittest.TestCase):

    def setUp(self):
        pass

    def test_given_case_max(self):
        np.random.seed(1)
        A = np.random.randint(-5, 5, size=(3, 4, 4, 4))
        kernel_size = 2
        Z = pool_forward(A, kernel_size, mode='max')

        self.assertEqual(Z.shape, (3, 2, 2, 4))
        expected_sample2_channel2 = np.array(
            [[ 4.,  3.],
             [ 3., -2.]]
        )
        self.assertTrue(np.allclose(Z[2, :, :, 2], expected_sample2_channel2))
        expected_sample1_channel3 = np.array(
            [[ 4.,  4.],
             [-1.,  4.]]            
        )
        self.assertTrue(np.allclose(Z[1, :, :, 3], expected_sample1_channel3))

    def test_given_case_avg(self):
        np.random.seed(1)
        A = np.random.randint(-5, 5, size=(3, 4, 4, 4))
        kernel_size = 2
        Z = pool_forward(A, kernel_size, mode='avg')

        self.assertEqual(Z.shape, (3, 2, 2, 4))
        expected_sample2_channel2 = np.array(
            [[ 2.5,  -0.75],
             [ 0.0, -3.25]]
        )
        self.assertTrue(np.allclose(Z[2, :, :, 2], expected_sample2_channel2))
        expected_sample1_channel3 = np.array(
            [[-1.  ,  0.75],
             [-2.75, -0.25]]        )
        self.assertTrue(np.allclose(Z[1, :, :, 3], expected_sample1_channel3))


    def test_bigger_case_max(self):
        np.random.seed(2)
        A = np.random.randint(-100, 100, size=(20, 55, 88, 16))
        kernel_size = 5
        Z = pool_forward(A, kernel_size, mode='max')

        self.assertEqual(Z.shape, (20, 11, 17, 16))
        expected = np.array(
            [[96., 96., 96., 94., 84., 94., 88., 94., 80., 99., 92., 92., 97.,
                88., 86., 94., 99.],
            [94., 97., 93., 98., 93., 85., 84., 83., 98., 97., 96., 96., 99.,
                99., 83., 95., 94.],
            [96., 93., 98., 98., 96., 92., 95., 97., 98., 89., 96., 94., 96.,
                95., 76., 97., 96.],
            [92., 82., 92., 95., 91., 97., 94., 91., 86., 94., 93., 90., 72.,
                93., 78., 94., 95.],
            [93., 71., 89., 96., 84., 96., 80., 91., 93., 99., 98., 95., 92.,
                97., 92., 95., 91.],
            [93., 98., 82., 89., 93., 88., 87., 96., 98., 80., 98., 66., 98.,
                99., 97., 99., 98.],
            [80., 79., 97., 66., 79., 80., 93., 96., 99., 97., 86., 83., 90.,
                90., 94., 99., 93.],
            [97., 92., 98., 97., 99., 87., 95., 93., 98., 93., 97., 96., 98.,
                85., 99., 90., 94.],
            [96., 88., 97., 89., 99., 99., 99., 93., 91., 86., 99., 90., 95.,
                89., 98., 86., 88.],
            [90., 92., 99., 98., 98., 89., 99., 99., 98., 96., 98., 92., 93.,
                92., 99., 99., 96.],
            [97., 99., 94., 99., 92., 98., 90., 82., 95., 89., 92., 99., 73.,
                71., 80., 79., 96.]]
        )
        self.assertTrue(np.allclose(Z[10, :, :, 8], expected))

    def test_bigger_case_avg(self):
        np.random.seed(2)
        A = np.random.randint(-100, 100, size=(20, 55, 88, 16))
        kernel_size = 5
        Z = pool_forward(A, kernel_size, mode='avg')

        self.assertEqual(Z.shape, (20, 11, 17, 16))
        expected = np.array(
            [[  0.68,  10.12,  -0.04,  -0.6 ,   7.32,  17.6 ,  -2.24,   3.48,
                11.16,   8.6 ,   9.72, -11.52,  -5.6 ,  -8.64, -17.24, -20.76,
                -13.96],
            [-13.  , -18.  ,  17.  ,  12.12, -18.  ,  -4.84,   1.92,   0.24,
                1.36,  11.92,   9.52,  -2.48,  -1.2 ,   7.24, -12.88,  12.52,
                6.32],
            [ -1.08,  13.64,   3.44,  -4.56,  -2.2 ,   3.76,   9.36,  -4.84,
                14.16,  -2.16,   9.4 , -14.28,  -8.76, -16.12, -17.52,  19.28,
                -4.96],
            [  3.48,  18.2 ,  -2.24,   3.68,  -6.32,  -4.52,  11.68,  -9.16,
                -9.96,  -2.08,   3.72, -21.24,   9.96,   2.84,   0.12,  22.08,
                0.24],
            [-10.92,  -7.36,   9.36,   2.  ,  -2.04,  27.24, -15.04,  16.76,
                6.64,  10.84,   0.08,  -5.92,   4.44,  16.04, -22.52,  17.4 ,
                15.4 ],
            [  5.96,  -4.  ,   3.68,  -9.56,  -3.44,  -6.68,  13.2 , -12.68,
                4.28, -24.64, -15.2 , -10.92,   1.72,   7.88,  21.32,  16.36,
                15.28],
            [ -0.24,   2.2 ,  -3.96, -13.16, -11.92,  -7.8 , -11.96,  -0.4 ,
                -3.8 ,  -2.68, -13.28,  -1.76, -13.4 , -15.36, -10.76,  11.56,
                1.4 ],
            [  5.84,   2.48,  -6.72, -13.56,  -5.24, -13.2 ,   2.6 ,   0.2 ,
                -7.12,  -4.44,  -1.52,  -4.16,  -4.2 ,  -1.32,   1.  ,   1.  ,
                17.64],
            [ 21.4 ,  -5.24,   7.76,  -7.52,   4.64,  -3.6 ,  23.16,  -8.16,
                9.76, -29.2 ,  -8.76,  -6.72,   9.92,  -7.24,  -5.36,  -1.24,
                2.24],
            [ -0.24,  -1.  ,   7.84,   8.2 ,   8.44,   1.12,   9.84, -11.92,
                11.08, -14.76,  -8.24,  -2.48, -14.2 ,   8.84,  15.2 ,   4.12,
                -6.16],
            [-11.32,   0.4 ,   5.6 ,  -1.24,   3.16,  -0.68,   1.48,  18.4 ,
                21.92,  -7.76,  -4.52,   2.08,  -4.72, -36.24, -25.12,  -0.28,
                10.6 ]]
        )
        self.assertTrue(np.allclose(Z[10, :, :, 8], expected))


class test_zero_pad(unittest.TestCase):

    def setUp(self):
        pass

    def test_default_padding(self):
        np.random.seed(2)
        X = np.random.randint(0, 256, size=(4, 3, 3, 3))
        X_pad = zero_pad(X)
        self.assertEqual(X_pad.shape, (4, 5, 5, 3))
        expected_pad = np.array(
            [[  0,   0,   0,   0,   0],
             [  0,  63,  39, 136,   0],
             [  0, 250, 237, 145,   0],
             [  0, 250, 185, 252,   0],
             [  0,   0,   0,   0,   0]]            
        )
        self.assertTrue((X_pad[2, :, :, 1] == expected_pad).all())

    def test_bigger_padding(self):
        np.random.seed(3)
        X = np.random.randint(0, 256, size=(5, 2, 2, 1))
        X_pad = zero_pad(X, 3)
        self.assertEqual(X_pad.shape, (5, 8, 8, 1))
        expected_pad = np.array(
            [[  0,   0,   0,   0,   0,   0,   0,   0],
             [  0,   0,   0,   0,   0,   0,   0,   0],
             [  0,   0,   0,   0,   0,   0,   0,   0],
             [  0,   0,   0, 249, 169,   0,   0,   0],
             [  0,   0,   0, 138, 149,   0,   0,   0],
             [  0,   0,   0,   0,   0,   0,   0,   0],
             [  0,   0,   0,   0,   0,   0,   0,   0],
             [  0,   0,   0,   0,   0,   0,   0,   0]]            
        )
        self.assertTrue((X_pad[3, :, :, 0] == expected_pad).all())

    # base case, should work if called with pad=0
    def test_no_padding(self):
        np.random.seed(4)
        X = np.random.randint(0, 256, size=(5, 5, 5, 5))
        X_pad = zero_pad(X, 0)
        self.assertEqual(X_pad.shape, (5, 5, 5, 5))
        expected_pad = np.array(
            [[ 49,  96,  16,  16,  91],
             [152, 169, 144,  33,  79],
             [185,  37,  94,  13,  46],
             [241,  28, 105,  41, 100],
             [142,  42, 160,  87, 137]]       
        )
        self.assertTrue((X_pad[4, :, :, 4] == expected_pad).all())


class test_conv_single_step(unittest.TestCase):

    def setUp(self):
        pass

    def test_example_case(self):
        np.random.seed(1)
        slice = np.random.randn(4, 4, 3)
        W = np.random.randn(4, 4, 3)
        b = np.random.randn(1, 1, 1)
        c = conv_single_step(slice, W, b)
        self.assertIsInstance(c, np.float64)
        self.assertAlmostEqual(c, -6.999089450680221)

    def test_more_channels(self):
        np.random.seed(1)
        slice = np.random.randn(5, 5, 32)
        W = np.random.randn(5, 5, 32)
        b = np.random.randn(1, 1, 1)
        c = conv_single_step(slice, W, b)
        self.assertIsInstance(c, np.float64)
        self.assertAlmostEqual(c, -13.779734001272715)

    def test_one_kernel_one_channel(self):
        np.random.seed(1)
        slice = np.random.randn(1, 1, 1)
        W = np.random.randn(1, 1, 1)
        b = np.random.randn(1, 1, 1)
        c = conv_single_step(slice, W, b)
        self.assertIsInstance(c, np.float64)
        self.assertAlmostEqual(c, -1.5218754464672077)


class test_conv_forward(unittest.TestCase):

    def setUp(self):
        pass

    def test_given_case(self):
        np.random.seed(1)
        # input activations have 10 samples, 5x7 shape, with 4 channels coming in
        A = np.random.randn(10,5,7,4)
        # the settings of the kernel W/b parameters.  This convolution creates
        # 8 channel/filter outputs. The kernel size is 3, and because
        # we use stride=2 pad=1 the new heightxwidth are 4x8
        W = np.random.randn(3,3,4,8)
        b = np.random.randn(1,1,1,8)
        stride = 2
        pad = 1
        Z = conv_forward(A, W, b, stride, pad)

        self.assertEqual(Z.shape, (10, 3, 4, 8))
        self.assertAlmostEqual(Z.mean(), 0.6923608807576933)
        expected_channel_means = np.array(
            [ 1.5086006 , -0.1743384 ,  2.68375965, -0.35835072,  2.76845058, -1.12789915, -0.41870826,  0.65737275]
        )
        self.assertTrue(np.allclose(Z.mean(axis=0).mean(axis=0).mean(axis=0), expected_channel_means))
        expected_channel_maxs = np.array(
            [11.12355835, 23.48197533, 19.73741108, 10.69033276, 19.89262537, 9.97188609, 13.00689405, 12.78954327]
        )
        self.assertTrue(np.allclose(Z.max(axis=0).max(axis=0).max(axis=0), expected_channel_maxs))
        expected_sample_3_channel_2 = np.array(
            [[ 4.54315363,  6.3188805 ,  9.48395578,  1.95844304],
             [ 8.07643821,  9.59542022, 19.73741108, -1.3748106 ],
             [ 3.36457258,  6.61941931,  9.9232075 ,  8.78183548]]            
        )
        self.assertTrue(np.allclose(Z[3,:,:,2], expected_sample_3_channel_2))


    def test_bigger_case(self):
        np.random.seed(3)
        # input activations have 10 samples, 256x128 shape, with 32 channels coming in
        A = np.random.randn(10,256,128,32)
        # the settings of the kernel W/b parameters.  This convolution creates
        # 64 channel/filter outputs. The kernel size is 5x5, and because
        # we use stride=5 pad=2 the new heightxwidth are 52, 26
        W = np.random.randn(5,5,32,64)
        b = np.random.randn(1,1,1,64)
        stride = 5
        pad = 2
        Z = conv_forward(A, W, b, stride, pad) 

        self.assertEqual(Z.shape, (10, 52, 26, 64))
        self.assertAlmostEqual(Z.mean(), 0.15105004696407845)
        expected_channel_means = np.array(
            [-0.00741596,  0.68182022, -1.36350957,  0.4179139 , -0.14052165,
                -0.42126877, -1.89696316, -1.16590733, -0.40760975, -0.71610292,
                -0.04495145,  1.54779134, -0.63113727,  0.95482755, -0.31910474,
                0.83765407, -0.7572652 , -0.45725869, -0.00655267,  0.1058567 ,
                0.31154808, -1.28845451,  2.0089342 , -0.21956557, -1.80861776,
                1.05966193, -0.91023077,  1.15595428,  0.3483273 ,  0.65143956,
                -1.60382935, -0.72719538, -0.59242545,  0.92045631,  1.66558851,
                1.59003021, -0.51793901,  0.83746719, -0.11521672, -0.69861321,
                0.20533031,  1.95667617, -0.49446255, -0.41953427,  3.57685909,
                -0.07897408,  0.08833971, -2.25148895,  0.55593264,  1.09278501,
                0.37219338,  0.86229026, -0.21402997,  0.59059918,  0.12032715,
                0.89085808,  2.41404201,  0.39891743,  0.27899598,  0.74835513,
                0.19576521,  1.97554297, -0.75221058, -0.72352079]
        )
        self.assertTrue(np.allclose(Z.mean(axis=0).mean(axis=0).mean(axis=0), expected_channel_means))
        expected_channel_maxs = np.array(
            [100.10024806, 114.62449138, 103.86979818, 110.91171259,
            111.80926405, 128.22994548, 110.58970932, 106.95928374,
            110.29469988, 107.71473686, 107.13873786, 117.79028242,
            105.55003848, 100.61882203, 108.14782826, 103.87966718,
            111.37978821, 100.39949451, 110.49106549, 119.5513399 ,
            99.17251553, 141.04659507, 108.22088495, 139.8916281 ,
            97.13379833,  97.37758609, 115.27931771, 125.53905274,
            148.42974628, 118.92025271, 126.43777448, 107.06074801,
            129.18649731, 112.59469544, 104.8127489 , 123.04021046,
            106.49481245, 119.16960395, 132.5650402 , 115.29397963,
            133.02612112, 118.82040671, 128.80341724, 103.50879621,
            113.029461  , 115.96830518, 105.78652728, 123.14433444,
            110.30203251, 107.01231832, 104.8632183 , 103.62567384,
            109.40576141, 105.7943842 ,  92.44484377, 118.27309963,
            120.40140326,  95.12552265, 107.26580984, 111.68265747,
            109.68069552, 106.18446743, 114.43224051, 123.54680322]
        )
        self.assertTrue(np.allclose(Z.max(axis=0).max(axis=0).max(axis=0), expected_channel_maxs))
        expected_sample_8_channel_42 = np.array(
        [[-4.88865648e+01, -1.35409806e+01, -2.41640805e+01,
            -3.37117505e+01, -2.20711297e+01,  1.20981230e+01,
            -9.91913944e+00, -9.71441619e+00, -3.34328301e+00,
            -8.20231533e+00],
        [ 1.48479209e+01, -6.65535532e-01, -3.16560298e+01,
            -1.38948413e+01,  2.89322504e+01, -4.99402342e+01,
            -1.49626590e+01, -6.26878843e-01,  2.68911681e+01,
            1.92670501e+01],
        [-1.98049400e+01, -3.39620747e+01, -1.58308703e+01,
            -2.45231610e+01,  8.92946796e+00,  9.13325345e+00,
            3.59063402e+01,  1.26607712e+00,  8.46948540e+00,
            -3.35257343e+01],
        [-4.34066541e+01, -1.87155101e+00,  1.24967904e+01,
            6.32043116e+00,  1.45936974e+01,  3.50105009e+01,
            -6.41290971e-02,  1.73258908e+01,  2.06080111e+01,
            -1.70096061e+01],
        [-8.40842950e+00,  1.04186489e+01, -3.38887945e+01,
            -4.18711870e+01, -5.35714182e+01, -4.23717810e+00,
            2.77053232e+01,  2.91323555e+01,  3.98768737e+00,
            -1.29070966e+01],
        [-6.67565029e+00, -1.97950413e+01,  1.25372434e+01,
            4.15318701e+01, -1.67259835e+01, -4.30458000e+01,
            1.40563030e+01, -3.12544916e+01, -2.45676782e+01,
            2.28416313e+01],
        [-3.30534286e+00, -3.62347272e+00, -1.91330805e+01,
            1.41178828e+01,  8.79180884e+01,  5.25805245e+01,
            -2.47392269e+01, -8.36428487e+00, -3.78258178e+01,
            2.44831579e+01],
        [-3.64613118e+01,  1.54024602e+01, -1.98901073e+01,
            5.16901770e+01,  1.25423929e-01, -1.24026610e+01,
            -2.14066485e+01, -2.96141993e+01, -4.88017122e+00,
            1.13708432e+01],
        [ 1.02977567e+00,  2.41757706e+01,  8.03528266e+00,
            -2.41542307e+01, -6.42835602e+00, -2.49361903e+01,
            -2.11533008e+01, -2.10944297e+01, -2.06589816e+01,
            -1.21940039e+01],
        [-1.98582863e+01,  6.60603749e+01,  1.01037284e+01,
            -1.04764036e+01,  3.38034643e+01, -3.07347347e+01,
            3.77636236e+01, -9.58463482e+00, -4.41214354e+01,
            -2.34752460e+01]]
        )
        self.assertTrue(np.allclose(Z[8,10:20,10:20,42], expected_sample_8_channel_42))

