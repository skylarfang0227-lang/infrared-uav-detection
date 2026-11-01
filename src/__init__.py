"""Infrared UAV detector implementations."""
from pathlib import Path
import os
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).resolve().parents[1]/'logs/ultralytics'))
