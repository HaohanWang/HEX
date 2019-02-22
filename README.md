# Learning Robust Representations by Projecting Superficial Statistics Out 

Example implementation of the paper: 
    
- H. Wang, Z. He, Z. C. Lipton, and E. P. Xing, [Learning Robust Representations by Projecting Superficial Statistics Out](https://openreview.net/pdf?id=rJEjjoR9K7), Proceedings of Seventh International Conference on Learning Representations (ICLR 2019). 

### Code structure: 

- scripts/
  - \_\_init\_\_.py
  - [datagenerator.py](https://github.com/HaohanWang/HEX/blob/master/scripts/datagenerator.py) Helper function for to load ImageNet data, not part of the contribution of this GitHub repo.  
  - [model.py](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py) implementation of NGLCM and HEX plugged into AlexNet
    - AlexNet (original implementation): [Line 20-170](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L20)
    - AlexNet (with NGLCM and HEX): [Line 5-170](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L5), [Line 175-280](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L175)
        - NGLCM: Line 5-17, [Line 187-199](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L187)
        - Equation 3: [Line 238-244](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L238)
        - Expanding the final layer: [Line 246-253](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L246)
        - Equation 4: [Line 266-278](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L266)
        - Normalization is recommended: [Line 233-236](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L233)
  - [run.py](https://github.com/HaohanWang/HEX/blob/master/scripts/run.py) training and testing the model in classification
    - Prepare the data for NGLCM: [Line 23-46](https://github.com/HaohanWang/HEX/blob/master/scripts/run.py#L23)
    
### Replication

For the codes that are used to replicate the experiments in the paper, please visit [HaohanWang/HEX_experiments](https://github.com/HaohanWang/HEX_experiments)


### Contact
- [Haohan Wang](http://www.cs.cmu.edu/~haohanw/)
- [@HaohanWang](https://twitter.com/HaohanWang)
