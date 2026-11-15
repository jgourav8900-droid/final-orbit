import time
import threading
from collections import OrderedDict
from typing import Any, Optional

class BoundedTTLDict:
    """
    Thread-safe, bounded in-memory cache with LRU eviction and TTL expiration.
    Protects the server from unbounded memory growth and memory exhaustion attacks.
    """
    def __init__(self, maxsize: int = 2000, default_ttl: int = 1800):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key not in self._data:
                return default
            exp, val = self._data[key]
            if exp is not None and time.time() > exp:
                del self._data[key]
                return default
            self._data.move_to_end(key)
            return val

    def set(self, key: str, val: Any, ttl: Optional[int] = None) -> None:
        actual_ttl = self.default_ttl if ttl is None else ttl
        # actual_ttl=None → no expiry (permanent); 0 → expires immediately; positive → normal TTL
        if actual_ttl is None:
            exp = None
        elif actual_ttl <= 0:
            # Already expired — set exp to epoch zero so get() will immediately prune it
            exp = 0.0
        else:
            exp = time.time() + actual_ttl
        with self._lock:
            if key in self._data:
                del self._data[key]
            elif len(self._data) >= self.maxsize:
                # Evict oldest entry (LRU)
                self._data.popitem(last=False)
            self._data[key] = (exp, val)

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __setitem__(self, key: str, val: Any) -> None:
        self.set(key, val)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        with self._lock:
            now = time.time()
            expired = [k for k, (exp, _) in self._data.items() if exp is not None and now > exp]
            for k in expired:
                del self._data[k]
            return len(self._data)

    def pop(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key not in self._data:
                return default
            exp, val = self._data.pop(key)
            if exp is not None and time.time() > exp:
                return default
            return val

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
