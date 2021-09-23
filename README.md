# Detection of traditional orchards meadows in Hesse, Germany via CNN & remote sensing data
Attempt to detect traditional orchard meadows (ger. Streuosbtwiesen). Traditional orchard meadows are standard fruit trees scattered on a meadow, pasture or more rarely arable land. Characteristic fruit varieties are apple, pear, plum and sweet cherry. Today traditional orchard meadows are managed extensive and without fertilizers. Due to the composition of vigorous, young, old trees and deadwood traditional orchard meadows habitat for a wide variation of animals and plants. In the state of Hesse (Germany) these landscape elements or more specific traditional orchard meadows with uniquely attributes are sanctuary. 

![Example images](https://github.com/jp-hecht/detect-streuobstwiesen/blob/main/img_out/IMG_20210918_121159.jpg?raw=true)



## To Do:
- [X] Comment all scripts
- [X] Delete old parts
- [ ] Add some more scripts 
-- especially data_preprocessing.R
- [ ] Add references
- [ ] Write more documentation
- [X] Find a better solution for the path settings according to the input (currently not very handy; if-statement)
- [ ] Update workflow figure
- [ ] Testing!!

## Workflow:
Currently the workflow is divided into four "subscripts" which could be controlled by another script (control_script.R). This other script is used to set the parameter values and also which subscript is necessary for a specific task.
<img src="/img_out/workflow.png" width="924" height="394" />
- data_preprocessing.R: necessary to produce a mask from the SOW data & mosaicing raster input to extent of Hesse
- data_split.R: Firstly subset the mask and raster input to smaller extent; afterwards copy theses tiles for latter prediction & remove all/or percentage of tiles which are equal for model training
- main_cnn_model.R: split the input data for modeling and visualization; save the model for prediction
- predict.R: load model for prediction & prepare saved data & predict

