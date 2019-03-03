# hybrid attention for automatic segmentation of whole fetal head in ultrasound volumes 

## A customized deep learning algorithm used to segment fetus head volume
### *Abstract:*
  in this paper, we propose the first fully-automated solution to segment the whole fetal head in ultrasound volumes. the segmentation task is firstly formulated as an end-to-end volumetric mapping under an encoder-decoder deep architecture. we then combine the segmentor with our proposed hybrid attention scheme (has) to select discriminative features and suppress the non-informative volumetric features in a composite and hierarchical manner. has proves to be effective in helping segmentor combating boundary ambiguity and deficiency. to enhance the spatial consistency in the segmentation, we further organize multiple segmentors in a cascaded fashion to refine the result by revisiting the context encoded in the predictions from predecessors. validated on a large dataset, our method presents superior segmentation performance, high agreements with experts and decent reproducibilities, and hence is promising to be a feasible solution in advancing the volumetric ultrasound-based prenatal examinations.The source code for experimentation was written in Tensorflow and is available online in my GitHub repository.

***

### Prerequisites:

This code was tested in following environment setting:

* Python (version = 3.5.0)

* Tensorflow (version = 1.4.0)

#### The main package is installed:

[skimage](http://scikit-image.org/) and [nibabel](http://nipy.org/nibabel/) need be installed

### Usage

First clone this repository:

git clone https://github.com/wxde/USegNet-DS-HAS.git

the stage folder is the first stage of auto-context
'''
cd stage_1 
python main.py
'''


### Results:

### Contact:

For queries contact me at xuwang753@google.com.
