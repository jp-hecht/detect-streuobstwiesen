# Detection of traditional orchards meadows in Hesse, Germany via CNN & remote sensing data
Attempt to detect traditional orchard meadows (ger. Streuosbtwiesen). Traditional orchard meadows are standard fruit trees scattered on a meadow, pasture or more rarely arable land. Characteristic fruit varieties are apple, pear, plum and sweet cherry. Today traditional orchard meadows are managed extensive and without fertilizers. Due to the composition of vigorous, young, old trees and deadwood traditional orchard meadows habitat for a wide variation of animals and plants. In the state of Hesse (Germany) these landscape elements or more specific traditional orchard meadows with uniquely attributes are sanctuary. 

__Example image from__ [MDR](https://www.mdr.de/nachrichten/sachsen/streuobst-wiese-gefaehrdet-jahrestag-umweltschutz-100.html)
[<img src="https://cdn.mdr.de/nachrichten/mdraktuell-4764-resimage_v-variantSmall24x9_w-832.jpg?version=23889">](https://cdn.mdr.de/)

## To Do:
- [ X ] Comment all scripts
- [ ] Sort out old parts
- [ ] Add some more scripts -> especially data_preprocessing.R
- [ ] Add references
- [ ] Write more documentation
- [ ] Find a better solution for the path settings according to the input (currently not very handy; if-statement)

## Workflow:
Currently the workflow is divided into four "subscripts" which could be controlled by another script (controll_scrip_rgb.R). This other script is used to set the parameter values and also which subscript is necessary for a specific task.
<img src="/img_out/workflow.png" width="924" height="394" />

