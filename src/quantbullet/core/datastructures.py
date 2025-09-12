from collections import defaultdict

class FrozenKeysDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen = False

    def freeze(self):
        """Disallow creation of new keys"""
        self._frozen = True

    def __missing__(self, key):
        if self._frozen:
            raise KeyError(f"New key '{key}' not allowed (dict is frozen)")
        return super().__missing__(key)

    def __setitem__(self, key, value):
        if self._frozen and key not in self:
            raise KeyError(f"New key '{key}' not allowed (dict is frozen)")
        super().__setitem__(key, value)
