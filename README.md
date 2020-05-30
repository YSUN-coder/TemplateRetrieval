# Image Retrieval Engine Based on Keras

## Introduction
Image Retrival based on VGGNet16 model application and cosine similarity.

## Environment
For Linux or MacOS

 1. create conda virtual environment

```
conda create -n ${your_env_name} python=3.6
```

 2. install libraries in requirements.txt

```
pip install -r requirements.txt
```

## Usage


### Example

### Speed and accuracy
*  **Speed**

*  **Accuracy on big dataset, challenge scenario and challenge card**


| Challenge type| Speed | Accuracy |
| :-----| ----: | :----: |
| 100000 templates | ss | first:% second:% |
| Crop half | ss | first:% second:%  |
| Crop 1/3 with angle | ss | first:% second:%  |
| -90 Orientation | ss | first:% second:%  |
| random Orientation | ss | first:% second:%  |


*  **Best dimension choice for retriavel task**

## Feature Storage
* CSV
* CSV.gzip
* Pickle
* HDF5
* HDF5 zips




**Conclusion**

1. Pickle is the fastest in read and write, but not usable for big data which causes SystemError.

2. HDF5 is second and easy for data  in structure, and also showing a good performance in compressibility.

3. Feather-format could be faster than HDF5 and Pickle.


## Update


## Todo


## Related Paper


## Useful linux command
1. file size: du -h --max-depth=1  ${filename}
2. file num: ls -l|grep "^-"| wc -l