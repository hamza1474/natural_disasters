# natural_disasters

### Find Optimial Learning Rate Range:

`python train.py --lr-find 1`

( A plot will be created in > /outputs/lrfind_plot.png, adjust min/max learning rates accordingly in utils/config.py )

_I've already run this for 20 epochs and the plot is saved in /outputs/lrfind_plot.png_

***


### Train the network:

`python train.py`

(Two plots will be saved in /outputs, training_plot.png and clr_plot.png)

***

### Predict

`python predict.py --input path/to/to/input/video --output path/to/output/video`

(optional parameter --display 1 to display output while processing)
