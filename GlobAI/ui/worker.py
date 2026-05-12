"""
ui/worker.py
------------
Threaded workers for PyQt6 application to ensure UI does not block.
"""

from PyQt6.QtCore import QThread, pyqtSignal
from typing import Callable

class TaskWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
