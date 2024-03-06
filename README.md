# RecJPQ

This is an official repository for the WSDM 2024 paper "RecJPQ: Training Large-Catalogue Sequential Recommenders", co-authored by [Aleksandr Petrov](https://asash.github.io) and [Craig Macdonald](https://www.dcs.gla.ac.uk/~craigm/)

RecJPQ replaces item embeddings layer in recommender systems. By decomposing atomic item id into limited number of sub-item ids, it allows to reduce memory consumption of the model by a large factor. For example, we were able to reduce memory consumption of the SASRec model by a factor of 47x withouth compromising effectiveness. 

More details in the paper: https://arxiv.org/abs/2312.06165


If you use any part of the code, please consider citing us: 

```
@inproceedings{petrov2024recjpq,
  author = {Petrov, Aleksandr V. and Macdonald, Craig},
  title = {RecJPQ: Training Large-Catalogue Sequential Recommenders},
  year = {2024},
  doi = {10.1145/3616855.3635821},
  booktitle = {Proceedings of the 17th ACM International Conference on Web Search and Data Mining},
  pages = {538–547},
  location = {Merida, Mexico},
series = {WSDM '24}
}
```
#Instruction

The code is based on  our aprec framework. Please clone this code and follow the original instructions https://github.com/asash/bert4rec_repro to setup the environment. 

The code for the RecJPQ versions of the model described in the paper is located in the folder recommenders/sequential/models/recjpq. 
Configuration files can be found in evaluation/configs/jpq

Please follow the instructions https://github.com/asash/bert4rec_repro to run the experiments. 

