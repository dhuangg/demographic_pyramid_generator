# CA100 Demographic Pyramid Generator
This is a project to generate visualizations specifically for [California 100](https://california100.org/)'s regional reports. We will create 9 regional demographic pyramid visualizations using the latest year from the US Census (ACS5) data.

## Demographic Pyramid
A [population pyramid](https://en.wikipedia.org/wiki/Population_pyramid) is a common socioeconomic data visualization to understand the age and gender distribution of a geographic region.

For this project, we will generate a more detailed **demographic pyramid** that displays the distribution of ethnicity for each age demographic. This was inspired by [this project](https://medium.com/@databayou/why-i-made-race-and-ethnicity-population-pyramids-e41b486e3806).

## How to run the generator
### Pre-Requisite
1. Install [Python](https://www.python.org/downloads/)

### Description
This project has 2 ways to generate the pyramid. If you just want to generate the pyramid, please use the **Hassle-Free Way**. If you wish to be more interactive, understand the code, and make customizations, use the **Interactive Way.** 

The **Interactive Way** is through a Jupyter notebook with descriptions of the code and its processes. If you wish to make any changes such as changing the colors or labels, please use this way. 

### Hassle Free Way
1. Open terminal on [Mac](https://support.apple.com/guide/terminal/open-or-quit-terminal-apd5265185d-f365-44cb-8b09-71a064a42125/mac) or [PC](https://www.wikihow.com/Open-Terminal-in-Windows)
2. Open the folder with the code in your finder
3. Follow [the instructions](https://osxdaily.com/2015/11/05/copy-file-path-name-text-mac-os-x-finder/) to copy the file path of your folder
4. In your terminal, type in `cd ` then what you copied from step 3. Example: `cd /Users/danielhuang/Documents/CA100/code`
5. Run `python demographic_pyramid_generator.py`

### Interactive Way
#### Pre-Requisites
1. Install [Python](https://www.python.org/downloads/)
2. Install [Jupyter](https://www.codecademy.com/article/how-to-use-jupyter-notebooks)
3. Open terminal on [Mac](https://support.apple.com/guide/terminal/open-or-quit-terminal-apd5265185d-f365-44cb-8b09-71a064a42125/mac) or [PC](https://www.wikihow.com/Open-Terminal-in-Windows)

### How to run the Notebook
The notebook is divided into "cells" of individual code.

1. Open terminal and run `jupyter notebook`
2. A tab should open in your browser with the url "localhost:8888". If the tab doesn't open, you can type in "localhost:8888" into the url manually.
3. Run the entire notebook by pressing the "Play" button.
4. You can run each cell by clicking the "Shift + Enter" keys.
5. **Run each cell**. Later cells depend on the earlier cells, so while iPython allows you to run later cells, this may result in errors.