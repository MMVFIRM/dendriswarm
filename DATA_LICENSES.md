# Dataset notices

## CIFAR-100

DendriSwarm v0.7 supports the CIFAR-100 Python dataset published by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The repository does not redistribute the archive. Operators download it from the official University of Toronto dataset page and the preparation command verifies the published archive MD5 before use.

The campaign stores a content-addressed local derivative containing the official images and labels, a deterministic stratified campaign split, native fine/coarse mapping metadata, and channel normalization statistics. Users remain responsible for reviewing and complying with the dataset's terms and citation requirements.

## Optical Recognition of Handwritten Digits

The historical v0.2 transport proof uses the 1,797-sample 8×8 digits dataset exposed by `sklearn.datasets.load_digits`, derived from:

E. Alpaydin and C. Kaynak, **Optical Recognition of Handwritten Digits**, UCI Machine Learning Repository, 1998. DOI: `10.24432/C50P49`.

The UCI repository lists that dataset under CC BY 4.0. DendriSwarm does not redistribute a separate raw copy.
