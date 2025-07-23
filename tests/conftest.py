# tests/conftest.py
import sys, os

# Insert the project root (one level up) to the front of sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
