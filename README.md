# Learning Robust Representations by Projecting Superficial Statistics Out 

Example implementation of the paper: 
    
- H. Wang, Z. He, Z. C. Lipton, and E. P. Xing, [Learning Robust Representations by Projecting Superficial Statistics Out](https://openreview.net/pdf?id=rJEjjoR9K7), Proceedings of Seventh International Conference on Learning Representations (ICLR 2019). 

### Code structure: 

- scripts/
  - \_\_init\_\_.py
  - [datagenerator.py](https://github.com/HaohanWang/HEX/blob/master/scripts/datagenerator.py) Helper function for to load ImageNet data, not part of the contribution of this work.  
  - [model.py](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py) implementation of NGLCM and HEX plugged into AlexNet
    - AlexNet (original implementation): [Line 20-170](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L20)
    - AlexNet (with NGLCM and HEX): [Line 5-114](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L5), [Line 175-280](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L175)
        - NGLCM: [Line 5-17](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L5), [Line 187-199](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L187)
        - Equation 3: [Line 238-244](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L238)
        - Expanding the final layer: [Line 246-253](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L246)
        - Equation 4: [Line 266-278](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L266)
        - Normalization is recommended: [Line 233-236](https://github.com/HaohanWang/HEX/blob/master/scripts/model.py#L233)
  - [run.py](https://github.com/HaohanWang/HEX/blob/master/scripts/run.py) training and testing the model in classification
    - Prepare the data for NGLCM: [Line 23-46](https://github.com/HaohanWang/HEX/blob/master/scripts/run.py#L23)
    
### Replication

For the codes that are used to replicate the experiments in the paper, please visit [HaohanWang/HEX_experiments](https://github.com/HaohanWang/HEX_experiments)

### FAQ

- **This method does not seem to converge.**  
   
   - As we mentioned in the paper, we also have a hard time to optimize AlexNet with our method from beginning, however, we noticed that there are many tricks that can help. 
   
        - Train a network following the standard manner and then finetune with our method is recommended.   
        - The initilization of NGLCM plays an important role; in fact, it plays a more important role than the optimization process of NGLCM. Therefore, if you notice that the initlization scales the representations too much and leads to NaN, we recommend to freeze the optimization of NGLCM (then at least it's a starndard GLCM) rather than to alter the initialization manner. Another useful stragety (thanks to [Songwei](https://github.com/SongweiGe)) is to normalize the representations to avoid scaling. 
        
- **This method does not seem to help improve the performance of my CNN.**  

   - As CNN is known to take advantage of superficial (non-semantic) information of the data, we do not guarantee our method to help improve the performance of the setting where testing data and training data are from the same distribution (where simply predicting through superficial information can also help). In other words, our method only excels in the setting where learning the semantic information plays an important role (such as domain adaptation/generalization settings). 

### Bibtex

    @inproceedings{
    wang2018learning,
    title={Learning Robust Representations by Projecting Superficial Statistics Out},
    author={Haohan Wang and Zexue He and Zachary L. Lipton and Eric P. Xing},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=rJEjjoR9K7},
    }

### Contact
- [Haohan Wang](http://www.cs.cmu.edu/~haohanw/)
- [@HaohanWang](https://twitter.com/HaohanWang)
