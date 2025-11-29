import setuptools

setuptools.setup(
   name='RL4CRN',
   version='1.0',
   description='A package for generating CRNs using different reinforcement learning algorithms.',
   author=['Maurice Filo', 'Nicolo Rossi'],
   author_email=['maurice.filo@bsse.ethz.ch', 'nicolo.rossi@bsse.ethz.ch'],
   install_requires=['wheel', 'torch', 'gymnasium', 'numpy', 'matplotlib', 'tqdm', 'scipy', 'openpyxl', 'pytorch_lightning', 'comet-ml', 'torchrl', 'networkx', 'seaborn', 'joblib', 'scikit-learn', 'scikit-sundae', 'python-louvain', ],
   packages=setuptools.find_packages()
)