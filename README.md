# LightFM

## Important

**This is a fork of the [original LightFM project](https://github.com/lyst/lightfm)**
1. Our main goal is to support emerging dependencies like newest versions of Python and to fix compatibility issues. 
2. We DO NOT guarantee support for all systems.
3. We DO NOT intend to expand or change any existing LightFM functionality.
4. We might fix essential bugs.

**About us:** we are maintainers of the [RecTools](https://github.com/MobileTeleSystems/RecTools) recommender systems framework. Our wrapper for LightFM provides 10-15 faster inference then the original model ([benchmark](https://github.com/MobileTeleSystems/RecTools/blob/main/examples/6_benchmark_lightfm_inference.ipynb)).  
In RecTools we support common RecSys models and validation tools in one easy-to-use framework.


![LightFM logo](lightfm.png)


[![PyPI](https://img.shields.io/pypi/v/rectools-lightfm.svg)](https://pypi.python.org/pypi/rectools-lightfm/)


LightFM is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback, including efficient implementation of BPR and WARP ranking losses. It's easy to use, fast (via multithreaded model estimation), and produces high quality results.

It also makes it possible to incorporate both item and user metadata into the traditional matrix factorization algorithms. It represents each user and item as the sum of the latent representations of their features, thus allowing recommendations to generalise to new items (via item features) and to new users (via user features).

For more details, see the [**Original** Documentation](http://lyst.github.io/lightfm/docs/home.html).

## Installation
Install from `pip`:
```
pip install rectools-lightfm
```

## Quickstart
Fitting an implicit feedback model on the MovieLens 100k dataset is very easy:
```python
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

# Load the MovieLens 100k dataset. Only five
# star ratings are treated as positive.
data = fetch_movielens(min_rating=5.0)

# Instantiate and train the model
model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

# Evaluate the trained model
test_precision = precision_at_k(model, data['test'], k=5).mean()
```

## Articles and tutorials on using LightFM
1. [Learning to Rank Sketchfab Models with LightFM](http://blog.ethanrosenthal.com/2016/11/07/implicit-mf-part-2/)
2. [Metadata Embeddings for User and Item Cold-start Recommendations](http://building-babylon.net/2016/01/26/metadata-embeddings-for-user-and-item-cold-start-recommendations/)
3. [Recommendation Systems - Learn Python for Data Science](https://www.youtube.com/watch?v=9gBC9R-msAk)
4. [Using LightFM to Recommend Projects to Consultants](https://medium.com/product-at-catalant-technologies/using-lightfm-to-recommend-projects-to-consultants-44084df7321c#.gu887ky51)
