Medusa
======
 
Medusa is an approach to detect size-`k` modules of objects (`candidate objects`) that, taken together, appear most significant to another set of objects (`pivot objects`). 

Medusa operates on *large collections of heterogeneous data sets* and explicitly distinguishes between diverse data semantics. It builds on [collective matrix factorization](http://dx.doi.org/10.1109/TPAMI.2014.2343973) to derive different semantics, and it formulates the growing of the modules as a submodular optimization program. Medusa is flexible in choosing or combining the semantic meanings, and provides theoretical guarantees about the detection quality.

*Large heterogeneous data collections* contain interactions between variety of objects, such as genes, chemicals, molecular signatures, diseases, pathways and environmental exposures. Often, any pair of objects---like, a gene and a disease---can be related in different ways, for example, directly via gene-disease associations or indirectly via functional annotations, chemicals and pathways, yielding different semantic meanings.

This repository contains supplementary material for [*Jumping across biomedical contexts using compressive data fusion*](http://bioinformatics.oxfordjournals.org/cgi/content/long/32/12/i90) by Marinka Zitnik and Blaz Zupan.
 
 
Dependencies
------------
The required dependencies to build the software are `Numpy >= 1.8`, `SciPy >= 0.10`.


Usage
-----

[synthetic.py](synthetic.py) - Demonstrates Medusa on synthetic semantics.
    
See also [scikit-fusion](http://github.com/marinkaz/scikit-fusion), our module
for data fusion using collective latent factor models. 


Install
-------
To install in your home directory, use

    python setup.py install --user

To install for all users on Unix/Linux

    python setup.py build
    sudo python setup.py install

To install in development mode

    python setup.py develop


Citing
------

    @article{Zitnik2016,
      title     = {Jumping across biomedical contexts using compressive data fusion},
      author    = {Zitnik, Marinka and Zupan, Blaz},
      journal   = {Bioinformatics},
      volume    = {32},
      number    = {12},
      pages     = {90-100},
      year      = {2016},
      publisher = {Oxford Journals}
    }
    
    
License
-------
Medusa is licensed under the GPLv2.
