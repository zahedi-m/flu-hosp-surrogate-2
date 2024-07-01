
from pathlib import Path
import shutil

class FileRetriever:

    def __init__(self):
        pass

    def fetch(self, run_ids:list, srcPath, destPath):
        
        if not isinstance(srcPath, Path):
            srcPath= Path(srcPath)
        if not isinstance(destPath, Path):
            destPath= Path(destPath)
        
        destPath.mkdir(exist_ok=True, parents=True)

        for rid in list(run_ids):
            for file in srcPath.glob(str(rid)+"_*.npy"):  
                shutil.copy(file, destPath)