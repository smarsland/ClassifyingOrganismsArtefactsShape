Code and other details for the paper `Classifying Organisms and Artefacts by Their Shapes' by Arianna Salili-James et al.

This repository contains Python and Jupyter Notebook code for the following tasks:

(1) Curve extraction from images. In the contourExtraction folder. 

(2) Computation of distances by various methods. In the computeDistances folder broken down by method. Note that other packages are needed:

    (i) SRVF requires fdasrsf from https://github.com/jdtuck/fdasrsf_python Includes the code to compute the Karcher mean shapes.
    (ii) geometric currents requires https://github.com/olivierverdier/femshape
    (iii) LDDMM requires https://github.com/tonyshardlow/reg_sde
    (iv) Eigenshapes is stand-alone. Includes the code to compute the standard mean shapes.
    
(3) Data analysis. In KNN_Classification

The datasets we used in the paper are available at:
Vases: https://doi.org/10.6084/m9.figshare.14551002

Leaves: https://doi.org/10.6084/m9.figshare.14551005

Shells: https://doi.org/10.6084/m9.figshare.14551044
