# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from  tensorflow import io as tfio

# Build data loading functions
# path to file
capuchin=os.path.join('data','Parsed_Capuchinbird_Clips','XC3776-0.wav')
capuchin