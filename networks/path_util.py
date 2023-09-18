from os.path import join
from os.path import normpath
import platform

def pjoin(path, *paths):
    p = join(path, *paths)
    if platform.system() == "Windows":
        return normpath(p).replace('\\','/')
    else:
        return p