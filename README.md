# PFDNet

# Data preparation:
Download the ShanghaiTech dataset, then you also need to generate density map files via `generate_map.py`. And put them in `./data`, then download perspective map from paper `Revisiting Perspective Information for Efficient Crowd Counting`.
And put the corresonding files inside `./data`. You need to move coresponding files under the guidance of `config.py`. It is mentioned, that you need upgrade to `mat` files of perspective maps to v7.3 yourself.
Then `h5py` can read the mat files correctly. 

# Download model
```
bash download_models.sh
```
# Test
Pytorch version: 1.0 or 1.1
Install the cuda extension, and `sh SHA_test.sh`.
