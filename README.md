# DynamicGEM: Dynamic graph to vector embedding

I have made changes to remove the TIMERS and DynamicTRIAD dependency and to suit different graph formats. 

Learning graph representations is a fundamental task aimed at capturing various properties of graphs in vector space. Most recent methods learn such representations for static networks. However, real world networks evolve over time and have varying dynamics. Capturing such evolution is key to predicting the properties of unseen networks. To understand how the network dynamics affect the prediction performance, various embedding approaches have been proposed. In this dynamicGEM package, we present some of the recently proposed algorithms. These algorithms include [Incremental SVD](https://pdfs.semanticscholar.org/4e8f/82b0741c2151d36f2201fc11b0b148beab60.pdf), [Rerun SVD](https://arxiv.org/pdf/1711.09541.pdf), [Optimal SVD](https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf), [Dynamic TRIAD](http://yangy.org/works/dynamictriad/dynamic_triad.pdf), [Static AE](https://arxiv.org/pdf/1805.11273.pdf), [Dynamic AE](https://arxiv.org/pdf/1809.02657.pdf), [Dynamic RNN](https://arxiv.org/pdf/1809.02657.pdf), [Dynamic AERNN](https://arxiv.org/pdf/1809.02657.pdf). We have formatted the algorithms so that they can be easily compared with each other. This library is published as [DynamicGEM: A Library for Dynamic Graph Embedding Methods](https://arxiv.org/abs/1811.10734) [0]. 

## Implemented Methods
dynamicGEM implements the following graph embedding techniques:
* [Incremental SVD](https://pdfs.semanticscholar.org/4e8f/82b0741c2151d36f2201fc11b0b148beab60.pdf): This method utilizes a perturbation matrix capturing the dynamics of the graphs along with performing additive modification on the SVD. [1]
* [Rerun SVD](https://arxiv.org/pdf/1711.09541.pdf): This method uses incremental SVD to create the dynamic graph embedding.  In addition to that, it uses a tolerance threshold to restart the optimal SVD calculations and avoid deviation in incremental graph embedding. [2]
* [Optimal SVD](https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf): This method decomposes adjacency matrix of the graph at each timestep using Singular Value Decomposition (SVD) to represent each node using thedlargest singular values. [3]
* [Dynamic TRIAD](http://yangy.org/works/dynamictriad/dynamic_triad.pdf): This method utilizes the triadic closure process to generate a graphembedding that preserves structural and evolution patterns of the graph. [4]
* [Static AE](https://arxiv.org/pdf/1805.11273.pdf): This method uses deep autoencoder to learn the representation of each node in the graph. [5]
* [Dynamic AE](https://arxiv.org/pdf/1809.02657.pdf): This method models the interconnection of nodes within and acrosstime using multiple fully connected layers. It extends Static AE for dynamic graphs. [6]
* [Dynamic RNN](https://arxiv.org/pdf/1809.02657.pdf): This method uses sparsely connected Long Short Term Memory(LSTM) networks to learn the embedding. [6]
* [Dynamic AERNN](https://arxiv.org/pdf/1809.02657.pdf): This method uses a fully connected encoder to initially ac-quire low dimensional hidden representation and feeds this representation into LSTMs to capture network dynamics. [6]

## Graph Format
Due to variation in graph formats used by different embedding algorithms, we have written custom utils: dataprep_util which can convert various data types to the required dynamic graph format stored as list of [Digraph](https://networkx.github.io/documentation/networkx-1.10/reference/classes.digraph.html) (directed weighted graph) corresponding to the time-steps. [Networkx](https://networkx.github.io/documentation/networkx-1.10/index.html) package is used to handle these graph formats. The weight of the edges is stores as attibute "weight". The graphs are saved using [nx.write_gpickle](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.readwrite.gpickle.write_gpickle.html) and loaded using [nx.read_gpickle](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.readwrite.gpickle.read_gpickle.html). For datasets that do not have these structure, we have methods (for example "get_graph_academic" for academic dataset) which can convert it into the desired graph format.

## Repository Structure
* **DynamicGEM/embedding**: It consists of the most recent dynamic graph embedding approaches, with each files representing a single embedding method. We also have some static graph embedding approaches as baselines. 
* **DynamicGEM/evaluation**: Currently, we have graph reconstruction and link prediction implemented for the evaluation. 
* **DynamicGEM/utils**: It consists of various utility functions for graph data preparation, embedding formatting, plotting utilities, etc.
* **DynamicGEM/graph_generation**: It constis  of functions to generate dynamic stochastic block model with diminishing community.  
* **DynamicGEM/visualization**: It consists of functions for plotting the static and dynamic embeddings of the dataset.
* **DynamicGEM/experiments**: The functions for hyper-paramter tuning is present in this folder. 
* **DynamicGEM/TIMERS**: The matlab source code of the TIMERS along with added matlab modules for dataset preparation is present in this folder.
* **DynamicGEM/dynamicTriad**: It consists of the dynamicTriad source code.

## Dependencies
dynamicgem is tested to work on python 3.5. The module with working dependencies are listed as follows:

* gaph_tool>=2.27
* Cython==0.29
* decorator==4.3.0
* dill==0.2.8.2
* h5py==2.8.0
* joblib==0.12.5
* Keras==2.2.4
* Keras-Applications==1.0.6
* Keras-Preprocessing==1.0.5
* matlabruntimeforpython===R2017a
* matplotlib==3.0.1
* networkx==1.11
* numpy==1.15.3
* pandas==0.23.4
* scikit-learn==0.20.0
* scipy==1.1.0
* seaborn==0.9.0
* six==1.11.0
* sklearn==0.0
* tensorflow==1.11.0  or tensorflow-gpu==1.11.0 (whichever is compatible with python 3.5)
* Theano==1.0.3
    
## Viz:

The visualization of the the embedding is as follows:

<p align="center">
  <img width="420" height="300" src="images/dyntriad.png">
</p>


## Cite
   [0] Goyal, P., Chhetri, S. R., Mehrabi, N., Ferrara, E., & Canedo, A. (2018). DynamicGEM: A Library for Dynamic Graph Embedding Methods. arXiv preprint arXiv:1811.10734.
   ```
   @article{goyal2018dynamicgem,
   title={DynamicGEM: A Library for Dynamic Graph Embedding Methods},
   author={Goyal, Palash and Chhetri, Sujit Rokka and Mehrabi, Ninareh and Ferrara, Emilio and Canedo, Arquimedes},
   journal={arXiv preprint arXiv:1811.10734},
   year={2018}
   }
   ```
   [1] Brand, M. (2006). Fast low-rank modifications of the thin singular value decomposition. Linear algebra and its applications, 415(1), 20-30.
   ```
   @article{BRAND200620,
    title = "Fast low-rank modifications of the thin singular value decomposition",
    journal = "Linear Algebra and its Applications",
    volume = "415",
    number = "1",
    pages = "20 - 30",
    year = "2006",
    note = "Special Issue on Large Scale Linear and Nonlinear Eigenvalue Problems",
    issn = "0024-3795",
    doi = "https://doi.org/10.1016/j.laa.2005.07.021",
    url = "http://www.sciencedirect.com/science/article/pii/S0024379505003812",
    author = "Matthew Brand",
    keywords = "Singular value decomposition, Sequential updating, Subspace tracking"
    }
   ```
   [2] Zhang, Z., Cui, P., Pei, J., Wang, X., & Zhu, W. (2017). TIMERS: Error-Bounded SVD Restart on Dynamic Networks. arXiv                      preprint arXiv:1711.09541.
   ```
    @misc{zhang2017timers,
    title={TIMERS: Error-Bounded SVD Restart on Dynamic Networks},
    author={Ziwei Zhang and Peng Cui and Jian Pei and Xiao Wang and Wenwu Zhu},
    year={2017},
    eprint={1711.09541},
    archivePrefix={arXiv},
    primaryClass={cs.SI}
    }
   ```
    
   [3] Ou, M., Cui, P., Pei, J., Zhang, Z., & Zhu, W. (2016, August). Asymmetric transitivity preserving graph embedding. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 1105-1114). ACM.
   ```
    @inproceedings{ou2016asymmetric,
    title={Asymmetric transitivity preserving graph embedding},
    author={Ou, Mingdong and Cui, Peng and Pei, Jian and Zhang, Ziwei and Zhu, Wenwu},
    booktitle={Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining},
    pages={1105--1114},
    year={2016},
    organization={ACM}
    }
  ```
   [4] Zhou, L. K., Yang, Y., Ren, X., Wu, F., & Zhuang, Y. (2018, February). Dynamic Network Embedding by Modeling Triadic Closure Process. In AAAI.
   ```
  @inproceedings{zhou2018dynamic,
  title={Dynamic Network Embedding by Modeling Triadic Closure Process.},
  author={Zhou, Le-kui and Yang, Yang and Ren, Xiang and Wu, Fei and Zhuang, Yueting},
  booktitle={AAAI},
  year={2018}
  }
```
  [5] Goyal, P., Kamra, N., He, X., & Liu, Y. (2018). DynGEM: Deep Embedding Method for Dynamic Graphs. arXiv preprint arXiv:1805.11273.
```
@article{goyal2018dyngem,
  title={DynGEM: Deep Embedding Method for Dynamic Graphs},
  author={Goyal, Palash and Kamra, Nitin and He, Xinran and Liu, Yan},
  journal={arXiv preprint arXiv:1805.11273},
  year={2018}
}
```
   [6] Goyal, P., Chhetri, S. R., & Canedo, A. (2018). dyngraph2vec: Capturing Network Dynamics using Dynamic Graph Representation Learning. arXiv preprint arXiv:1809.02657.
   ```
     @misc{goyal2018dyngraph2vec,
    title={dyngraph2vec: Capturing Network Dynamics using Dynamic Graph Representation Learning},
    author={Palash Goyal and Sujit Rokka Chhetri and Arquimedes Canedo},
    year={2018},
    eprint={1809.02657},
    archivePrefix={arXiv},
    primaryClass={cs.SI}
}
```
