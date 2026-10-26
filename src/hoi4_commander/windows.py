"""Только публичные Win32 API окна и имени процесса, без чтения памяти игры."""
import ctypes
import sys
from ctypes import wintypes
from pathlib import PureWindowsPath

from .models import Rect, WindowInfo
from .safety import SafetyError


class WindowsBackend:
    def __init__(self):
        if sys.platform != "win32":
            raise SafetyError("Захват реальной игры поддерживается только в Windows 11")
        self.u = ctypes.WinDLL("user32", use_last_error=True)
        self.k = ctypes.WinDLL("kernel32", use_last_error=True)
        self.u.GetForegroundWindow.restype = wintypes.HWND
        self.u.IsWindowVisible.argtypes = [wintypes.HWND]
        self.u.IsIconic.argtypes = [wintypes.HWND]
        self.u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.u.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.k.OpenProcess.restype = wintypes.HANDLE
        self.k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        self.k.CloseHandle.argtypes = [wintypes.HANDLE]
        self.u.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        self.u.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        # Физические пиксели MSS и клиентские координаты должны совпадать при масштабировании DPI.
        try:
            self.u.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
            self.u.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except AttributeError:
            self.u.SetProcessDPIAware()

    def current_window(self):
        # GUI может уже установить DPI процесса: задаём контекст потока захвата явно.
        if not self.u.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4)):
            raise SafetyError("Не удалось включить физические координаты DPI для потока")
        hwnd = self.u.GetForegroundWindow()
        if not hwnd or not self.u.IsWindowVisible(hwnd) or self.u.IsIconic(hwnd):
            raise SafetyError("Нет активного видимого окна HOI4")
        pid = wintypes.DWORD()
        self.u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = self.k.OpenProcess(0x1000, False, pid.value)
        if not handle:
            raise SafetyError("Не удалось проверить процесс активного окна")
        try:
            buf = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(buf))
            if not self.k.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                raise SafetyError("Не удалось определить имя процесса")
            name = PureWindowsPath(buf.value).name.lower()
        finally:
            self.k.CloseHandle(handle)
        if name != "hoi4.exe":
            raise SafetyError("На переднем плане не HOI4; агент не переключает окна")
        r, origin = wintypes.RECT(), wintypes.POINT()
        if not self.u.GetClientRect(hwnd, ctypes.byref(r)) or not self.u.ClientToScreen(hwnd, ctypes.byref(origin)):
            raise SafetyError("Не удалось получить границы окна")
        if r.right <= 0 or r.bottom <= 0:
            raise SafetyError("Клиентская область пуста")
        title = ctypes.create_unicode_buffer(1024)
        self.u.GetWindowTextW(hwnd, title, len(title))
        return WindowInfo(handle=int(hwnd), title=title.value, process_name=name,
                          client=Rect(x=origin.x, y=origin.y, width=r.right, height=r.bottom))
