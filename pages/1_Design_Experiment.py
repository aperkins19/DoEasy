import streamlit as st
import subprocess
import pandas as pd 
import numpy as np 
import os
import json

import base64
from PIL import Image
import io
import pickle 

import sys



st.header('View DoE Data')

model_type = st.selectbox(
    "Select Design Type",
    ("Second Order", 'First Order', 'Two Way Interaction', 'Pure Quadratic')
    )

st.write('You selected:', model_type)

theta = st.slider(
    "Select Theta",
     0, 360, 180)

phi = st.slider(
    "Select Phi",
     0, 360, 35)

