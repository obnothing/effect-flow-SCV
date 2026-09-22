# Mean Pooling vs PDVQ t-SNE

Both panels use token states from the same frozen P11 BiGRU checkpoint. Panel (a) uses masked mean pooling of `H`; panel (b) uses the six-label evidence-difference representation `Z+ - Z-`. This is an aggregation comparison, not a separately trained baseline-classifier comparison. Visual compactness or overlap suggests differences in representation organization but does not prove a decision boundary. Test was not read.

PCA-50 descriptive combination statistics:

- Mean pooling: silhouette=-0.103894; mean within-group squared distance=35.427764.
- PDVQ evidence difference: silhouette=-0.077943; mean within-group squared distance=34.878245.
