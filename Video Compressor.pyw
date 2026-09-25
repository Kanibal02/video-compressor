"""Double-click to launch the Video Compressor GUI."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vidcomp.gui import main

main()
