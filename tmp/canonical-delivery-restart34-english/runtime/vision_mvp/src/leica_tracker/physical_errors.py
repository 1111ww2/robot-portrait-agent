"""共享的物理生命周期错误，与引导/运行时导入无关。"""


class SessionCleanupError(RuntimeError):
    """老工作线程可能会幸存下来：在这个过程中不要再打开另一个遭遇。"""
