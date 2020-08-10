# pharmaco-image

In this project, we aim to use [Human U2OS cell images](http://www.cellimagelibrary.org/pages/project_20269) ([GigaScience dataset](http://gigadb.org/dataset/view/id/100351/File_page/2)) to predict a large number of [compound activities](https://solr.ideaconsult.net/search/excape/) against different protein targets.

Investigations and key findings:

1. Batch effects exist in the cell image dataset. There are several methods to detect batch effects.
    1. Visualizing cell image features vs. experiment ID
    2. [Interactive visualization tool](http://jayw-www.cs.wisc.edu/BAqZN82NTz1Zmbqay4HCfJBNM/umap_viewer/) to detect batch effects
    3. Plot feature correlation heatmap
2. It is challenging to remove such batch effects.
    1. ComBat normalization
    2. Z-score normalization
3. It is promising to use cell image data to predict compound assay activities.
    1. Use compound fingerprint feature as a baseline
    2. Experiment with random forest, logistic regression with features extracted from a pre-trained CNN
    3. End-to-end train a LeNet CNN

To learn more, please check out our Jupyter Notebooks below and Python scripts in [`./scripts`](./scripts).

|Notebook|Description|
|:---:|:---:|
|[`image_processing.ipynb`](./image_processing.ipynb)|Visualize the raw images and their features|
|[`meta_data.ipynb`](./meta_data.ipynb)|Explore the meta data come with the image dataset, such as compound chemical annotations|
|[`feature_visualization.ipynb`](./feature_visualization.ipynb)|Visualize the single cell images, CNN extracted features, and clusterings on the extracted features|
|[`normalization.ipynb`](./normalization.ipynb)|Experiment with batch normalization methods such as Combat and z-score normalization|
|[`explore_excape_db.ipynb`](./explore_excape_db.ipynb)|Align U2OS image data with ExCAPE-DB assay data using chemical annotations|
|[`positive_control.ipynb`](./positive_control.ipynb)|Find compounds that have been tested on U2OS cell-line from the CCLE database.|
|[`assay_selection.ipynb`](./assay_selection.ipynb)|Aggregate cell-level CellProfiler features to assay-level|
|[`assay_prediction.ipynb`](./assay_prediction.ipynb)|Predict assay activity using U2OS images with random forest and logistic regression models|
|[`simple_cnn.ipynb`](./simple_cnn.ipynb)|Predict assay activity using U2OS images by training a LeNet CNN model|



