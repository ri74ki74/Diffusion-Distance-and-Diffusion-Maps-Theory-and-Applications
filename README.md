This repository contains code used for calculations and visualisations in my thesis "Diffusion Distance and Diffusion Maps: Theory and
Applications". Claude Opus 4.7 was used for writing this code. All decisions regarding the structure were made by me (meaning what to plot,
how to embed the graphs, etc). 

The data used for the FAUST example is downloaded from https://dfaust.is.tue.mpg.de/index.html:
  Navigate to "Download", choose FEMALE REGISTRATIONS.

Structure of the repository:
```
├── FAUST_data/                              # contains only registrations_f.hdf5
├── src/                                     # helper functions
│   ├── gen_functions/                       # functions used for generating artificial datasets
│   │      ├── gen_dumbbell.py               # generate dumbbell-shaped dataset
│   │      ├── gen_helix.py                  # generate toroidal_helix-shaped dataset
│   │      └── gen_torus.py                  # generate torus-shaped dataset
│   ├── graph/                               # functions for nx Graph
│   │      ├── create_graph.py               # construct weighted graph from the initial dataset 
│   │      └── highlight_diffusion_paths.py  # add diffusion attributes to nx Graph edges
│   ├── plot/                                # functions for creating illustrative plots
│   │      ├── visualise_dumbbell.py         # functions for visualising dumbbell-shaped dataset
│   │      ├── visualise_embedding.py        # functions for visualising embeddings
│   │      └── visualise_pointcloud.py       # functions for visualising datasets except the dumbbell-shaped dataset
│   ├── calculate_similarity.py              # calculate absolute error between datasets
│   ├── embed.py                             # embed pointcloud using diffusion maps
│   ├── kernels.py                           # kernels for calculating similarity
│   ├── transformations.py                   # functions for rigid transformations
├── artificial_examples.ipynb                # notebook for illustrating diffusion maps on artificially generated datasets
├── faust_example.ipynb                      # notebook for illustrating diffusion maps on FAUST dataset
└── Thesis_tex                               # all .tex for my thesis

```
