# Learning Robust Representations by Projecting Superficial Statistics Out 

Example implementation of the paper: 
    
- H. Wang, Z. He, Z. C. Lipton, and E. P. Xing, [Learning Robust Representations by Projecting Superficial Statistics Out](https://openreview.net/pdf?id=rJEjjoR9K7), Proceedings of Seventh International Conference on Learning Representations (ICLR 2019). 

### Code structure: 

- scripts/
  - \_\_init\_\_.py
  - [datagenerator.py](https://github.com/HaohanWang/HEX/blob/master/scripts/datagenerator.py) Helper function for to load ImageNet data, not part of the contribution of this GitHub repo.  
  - [model.py](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py) implementation of NGLCM and HEX plugged into AlexNet
    - AlexNet (original implementation): Line 20 - Line 170
    - AlexNet (with NGLCM and HEX): Line 5-Line 17, Line 175 - Line 280
        - NGLCM: Line 5 - Line 17, Line 187 - Line 199
        - Equation 3: Line 238 - Line 244
        - Expanding the final layer: Line 246 - Line 253
        - Equation 4: Line 266 - Line 278
        - Normalization is recommended: Line 233 - Line 236
  - [run.py](https://github.com/HaohanWang/HEX/blob/master/scripts/run.py) training and testing the model in classification
    - Prepare the data for NGLCM Line 23 - Line 46
