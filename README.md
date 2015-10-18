# face-func
Purpose: The project is to estimate the rotate angle of the face. 

Image: The images are from videos taken by my laptop camera. There are three videos in total. In all the videos, I turned my face from left to right. Two videos were used for training and the rest is used for test.

Methods: A siamese network is used for the angle estimation. Rather than a single contrastive loss used in the original caffe network, two regression loss are used. Three loss functions are tested for the estimation, i.e., contrastive loss only, contrastive loss combined with regression loss and regression loss only. Turning face from left to right was normalized from 0 to 6. 

Results: Preliminary experiments showed that regression model achieved higher estimation accuracy than contrastive loss involved model. Estimation accuracy histograms are shown below. The bins are bins=np.arange(0, 6, 0.1)

siamese-regression comb:
array([1386, 1077,  584,  332,  294,  207,   80,   69,   27,    7,    4,
          2,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0])

regression only
array([1378, 1086,  618,  422,  280,  162,   73,   33,   12,    2,    1,
          2,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0])

Siamese only
array([141, 103, 131,  77,  76,  68,  58,  75,  73,  65,  66,  68,  62,
        69,  83,  61,  65,  72,  68,  59,  77,  74,  59,  66,  62,  65,
        64,  65,  55,  69,  63,  58,  57,  64,  74,  52,  67,  64,  65,
        62,  55,  72,  66,  60,  65,  59,  76,  69,  68,  67,  64,  70,
        75,  62,  68,  63,  78,  51,  35])

This is only a simple demonstration of face rotate angle estimation. Further investigation is definetely needed. 

