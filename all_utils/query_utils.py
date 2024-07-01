from pathlib import Path
import shutil

import numpy as np

def get_y0(srcPath, destPath, run_ids, indices):
    
    srcPath= Path(srcPath) if not isinstance(srcPath, Path) else srcPath
    destPath= Path(destPath) if not isinstance(destPath, Path) else destPath

    for rid, index in zip(run_ids, indices):
        for ri in rid:
            for comp, c in zip(["incidence", "prevalence"], ["inc", "prev"]):
                src_filename= f"{ri}_y0_{comp}.npy"
                saved_filename= str(index)+ f"_y0_{c}.npy"
                if Path(src_filename).is_file():
                    shutil.copy(src_filename, destPath.joinpath(saved_filename))

