"""
Hall-of-Fame utilities for reinforcement-learning over CRN environments.

This module implements a small, efficient *Hall of Fame* (HoF) container that
keeps the best-performing environment snapshots seen so far during training.
Items are ranked by a scalar objective (typically the latest task reward/loss
stored in `crn_env.state.last_task_info['reward']`). The HoF supports:

- Bounded capacity via a heap-backed structure (fast add/replace of the worst item).
- Deduplication via a signature map (keeps only the best version per signature).
- Fast random sampling for replay-style training.
- Ranked iteration / indexing (best → worst) using a lazily rebuilt sorted cache.

Conventions:
    - The HoF is designed to keep *low-loss* (or equivalently high-quality) entries.
    - Internally, entries store `score = -loss` so that higher score means better.
    - Environment snapshots are cloned on insertion to avoid later mutation.
"""

import heapq
import random
import time
from copy import deepcopy

class HoFItem:
    """
    Container for a single Hall-of-Fame entry.

    Instances are ordered so they can be stored in a min-heap (`heapq`), where
    the heap root represents the *worst* entry currently kept (highest loss /
    lowest quality). Ties are broken by timestamp to ensure deterministic heap
    behavior when scores match.

    Args:
        loss (float): Objective value to minimize (lower is better).
        signature: Hashable identifier for the environment structure/state.
            This implementation stores `signature.tobytes()` to use as a dict key.
        timestamp (float): Time of insertion/update (e.g., `time.time()`).
        env: Snapshot of the environment to store (should be clone-safe).
    """
    def __init__(self, loss, signature, timestamp, env):
        # We invert loss because we want to keep LOW loss items.
        # heapq pops the smallest value.
        # We want to pop the WORST item (Highest Loss).
        # Smallest (-Loss) == Highest Loss.
        self.score = -loss  
        self.signature = signature.tobytes()
        self.timestamp = timestamp
        self.env = env

    def __lt__(self, other):
        """
        Heap ordering: worst entries compare as "smaller".

        We primarily compare by `score` (=-loss). Smaller score means worse.
        On ties, older timestamps are considered smaller.
        """
        # Standard min-heap comparison
        if self.score == other.score:
            return self.timestamp < other.timestamp
        return self.score < other.score
    
    def assign(self, other):
        """
        In-place update of this entry's contents from another HoFItem.

        This is used to refresh an existing signature with a better score
        without creating a new object (helps keep `signature_map` references valid).
        """
        self.score = other.score
        self.timestamp = other.timestamp
        self.env = other.env

class HallOfFame:
    """
    Fixed-capacity Hall-of-Fame for environment snapshots.

    Maintains up to `max_size` unique entries keyed by a state/environment
    signature. When adding:
    
      - If the signature already exists, the entry is updated only if it is better.
      - If the HoF is full, the new entry replaces the current worst entry only if
        it is better.

    The internal heap is optimized for fast worst-item access/replacement.
    Ranked access (best→worst) is provided via a lazily rebuilt sorted cache.

    Args:
        max_size (int): Maximum number of entries to keep.
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = [] 
        self.signature_map = {} 
        
        # Optimization for indexing/iteration
        self._sorted_cache = [] 
        self._cache_is_dirty = True 

    def _ensure_sorted(self):
        """Rebuild the sorted cache (best → worst) if it is marked dirty."""
        if self._cache_is_dirty:
            # Sort best (highest score/lowest loss) to worst
            self._sorted_cache = sorted(self.heap, key=lambda x: x.score, reverse=True)
            self._cache_is_dirty = False

    def add(self, crn_env):
        """
        Add a CRN environment snapshot to the Hall of Fame.

        The entry's loss is read from `crn_env.state.last_task_info['reward']`
        and the deduplication key is taken from `crn_env.state.get_bool_signature()`.

        Notes:
            - The environment is cloned before storage to prevent later mutation.
            - If a matching signature already exists, it is updated in-place only
              if the new candidate is better (lower loss / higher score).

        Args:
            crn_env: Environment instance expected to provide:

                - `state.last_task_info['reward']` (float-like loss)
                - `state.get_bool_signature()` (array-like signature with `.tobytes()`)
                - `clone()` (deep-ish copy used for snapshotting)

        Raises:
            ValueError: If the environment does not expose the expected reward field.
        """
        try:
            loss = crn_env.state.last_task_info['reward']
            unhashable_signature = crn_env.state.get_bool_signature() 
        except KeyError:
            # Depending on strictness, you might want to just return here instead of crashing
            raise ValueError("Environment state must have 'reward' in last_task_info.")

        # NOTE: this is actually critical to avoid issues when resetting older environments
        env_snapshot = crn_env.clone() 
        
        # New entry wrapper
        entry = HoFItem(loss, unhashable_signature, time.time(), env_snapshot)

        # 1. Handle Duplicates
        if entry.signature in self.signature_map:
            existing_entry = self.signature_map[entry.signature]
            
            # Compare scores explicitly for clarity
            if entry.score > existing_entry.score:
                # Update existing entry in-place
                existing_entry.assign(entry)
                # Re-establish heap invariant (O(N))
                heapq.heapify(self.heap)
                self._cache_is_dirty = True
            return 

        # 2. Add New Item
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, entry)
            self.signature_map[entry.signature] = entry
            self._cache_is_dirty = True
        else:
            # Check against the worst item (Root of min-heap)
            worst_entry = self.heap[0]
            
            if entry.score > worst_entry.score:
                # Remove worst from map
                del self.signature_map[worst_entry.signature]
                
                # Pop worst, push new
                # Note: heappushpop is more efficient than pop then push
                heapq.heappushpop(self.heap, entry)
                self.signature_map[entry.signature] = entry
                self._cache_is_dirty = True

    def add_all(self, crn_envs):
        """
        Add a collection of environments to the Hall of Fame.

        Args:
            crn_envs (iterable): Iterable of environments compatible with `add()`.
        """
        for env in crn_envs:
            self.add(env)

    def sample(self, batch_size):
        """
        Uniformly sample stored environments (unordered).

        Sampling does not require sorting and is therefore fast.

        Args:
            batch_size (int): Number of samples to draw.

        Returns:
            list: A list of sampled environment snapshots (length ≤ batch_size).
        """
        if not self.heap:
            return []
        
        k = min(len(self.heap), batch_size)
        samples = random.sample(self.heap, k)
        return [s.env for s in samples]

    def __iter__(self):
        """
        Iterate over stored environments ranked from best to worst.

        Yields:
            env: Environment snapshots ordered by increasing loss (best first).
        """
        self._ensure_sorted()
        for item in self._sorted_cache:
            yield item.env
    
    def __getitem__(self, index):
        """
        Get the environment snapshot by rank.

        Args:
            index (int): Rank index where 0 is best, 1 is second-best, etc.

        Returns:
            env: The environment snapshot at the requested rank.
        """
        self._ensure_sorted()
        return self._sorted_cache[index].env

    def __len__(self):
        """Return the number of entries currently stored in the Hall of Fame."""
        return len(self.heap)