import os
import sys
sys.path.append(os.getcwd())
import traceback


from typing import List, Dict


def cleanup(files: List):
    if isinstance(files, List):
        for file in files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                print(f"{e}, {repr(e)}, {traceback.format_exc()}")