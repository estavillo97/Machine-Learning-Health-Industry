# Data-Scientist-Challenge
Data Scientist Challenge for Mercado Libre
by Juan Pablo Estavillo


# Description
The goal of this repository is to present the technical challenge given by Mercado Libre, 
providing with evidence of my proficiency in python and my ability to extract
meaningful insights from complex data, and my creativity in approaching open-ended questions.


## Instalation

### Requirements
In order to run the task notebooks, we must install the libraries used to solve the challenges from the file 
'requirements.txt'using the following
command in your terminal.

```
pip install -r requirements.txt
```

# Setting up a Virtual Environment

A virtual environment is recommended to isolate project dependencies and prevent conflicts with your system-wide packages. You can create a virtual environment using different tools. Choose one of the following methods:

## Using pyenv (Recommended)

1. **Install pyenv**: If you haven't already, install `pyenv` by following the instructions in the [official documentation](https://github.com/pyenv/pyenv#installation).

2. **Create a Virtual Environment**: Open your project directory in the terminal and run the following commands to create a virtual environment.

   ```bash
   # Choose a Python version (e.g., 3.8.12)
   pyenv install 3.8.12

   # Create a virtual environment
   pyenv virtualenv 3.8.12 myproject-env

   # Activate the virtual environment
   pyenv local myproject-env

'''
pip install virtualenv
'''

# Create a virtual environment (replace 'myproject-env' with your preferred name)
virtualenv myproject-env

# Activate the virtual environment
source myproject-env/bin/activate


## Additional tools to get you started
about Sklearn
https://scikit-learn.org/stable/


## How To Use The Project

### Data Cleaning
The first thing to do is work with the data, to clean it and transform it to extract more meaningful features
for the predictive model, sci-kit learn library helps with that, combined with numpy, pandas and 
pyplot is possible to build the predictive model


### results
 n features | f1  | accuracy | ROC 
 10         .55    .92     .82  \
 20         .56    .93     .83  \
 30         .56    .93     .84  \
 40         .56    .93     .84  \
 50         .56    .93     .84  \
 60         .56    .93     .84  \
 73         .56    .93     .84  


## License
 
The MIT License (MIT)

Copyright (c) 2023 Juan Pablo Estavillo

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

