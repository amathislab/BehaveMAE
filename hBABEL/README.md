# Generation of hBABEL dataset

This work is based on the [BABEL dataset](https://babel.is.tue.mpg.de/index.html) from Punnakkal <i>et.al.</i>

## Data

Download the 3D pose data directly from the [AMASS website](https://amass.is.tue.mpg.de/). Then run this command to extract the AMASS sequences

```
python scripts/process_amass.py --input-path /path/to/data --output-path path/of/choice/default_is_/babel/babel-smplh-30fps-male --use-betas --gender male
```

Download the data from the [TEACH website](https://teach.is.tue.mpg.de/) and follow the instructions given in the [TEACH github page](https://github.com/athn-nik/teach?tab=readme-ov-file).


Finally, you can generate the hBABEL dataset by running the following command. </br>

<i>dependencies : numpy==1.24.3</i>

``` 
cd ..
python hBABEL/generate_hBABEL.py --babel_teach_path path/to/babel/teach --output_dir path/of/choice
```



