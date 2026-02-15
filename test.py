import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import shap

def verify_setup():
    print("--- AI Triage System Environment Check ---")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print("Environment ready: SUCCESS")

if __name__ == "__main__":
    verify_setup()