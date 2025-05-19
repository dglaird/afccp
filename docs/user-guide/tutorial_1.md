# Tutorial 1: Introduction

In this first tutorial, we'll go through the basics of how `afccp` is structured alongside the `CadetCareerProblem`
object. 

---

## 1. Setting up the Development/Testing Environment

By now, you should have a working version of afccp on your computer. You've tested that the module works and will 
perform as expected. If this is not the case, please follow the 
[Installation Guide](../getting-started/installation.md) to get the code working before following along in these 
tutorials.

In my afccp repo, I have an "executables" folder that contains all my various .py and .ipynb files that I use to
build my code and test it. If you're just getting started with this project, or if you're just getting started with
python in general, I'd recommend using jupyter notebooks. You can make a new jupyter notebook (.ipynb file) and follow
along with the code I use throughout these tutorials. For instructions on how to get jupyter notebook working
alongside the `afccp` virtual environment, you can follow the 
[Jupyter Installation Instructions](../getting-started/installation.md).

I HIGHLY recommend actually typing out the code and following along, rather than just reading this, as that will 
get you familiar with the many variables used. Assuming you're using a .py script or jupyter notebook inside an 
"executables" sub-folder in the root directory (which is already ignored by git), you'll first need to change the 
working directory back to the root directory. You can use the following code to do so:

```python
# Obtain initial working directory
import os
dir_path = os.getcwd() + '/'
print('initial working directory:', dir_path)

# Get main afccp folder path
index = dir_path.find('afccp') 
dir_path = dir_path[:index + 6]

# Update working directory
os.chdir(dir_path)
print('updated working directory:', dir_path)
```

You should expect to see the following output, relative to your paths of course!:

```
initial working directory: /Users/griffenlaird/Coding Projects/afccp/executables/
updated working directory: /Users/griffenlaird/Coding Projects/afccp/
```
___

## 2. The `CadetCareerProblem` class

---

## 📌 Summary

In this quickstart:
- We initialized a toy dataset of cadets and AFSCs.
- We ran the assignment model and viewed the results.
- We generated a plot to visualize the match.

To run more realistic examples...