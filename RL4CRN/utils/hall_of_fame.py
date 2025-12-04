import heapq
import random
import time
from copy import deepcopy

class HoFItem:
    """
    Helper class to store items in the Hall of Fame with ordering.
    """
    def __init__(self, loss, signature, timestamp, environement):
        # We invert loss because we want to keep LOW loss items.
        # heapq pops the smallest value.
        # We want to pop the WORST item (Highest Loss).
        # Smallest (-Loss) == Highest Loss.
        self.score = -loss  
        self.signature = signature.tobytes()
        self.timestamp = timestamp
        self.environment = environement

    def __lt__(self, other):
        # Standard min-heap comparison
        if self.score == other.score:
            return self.timestamp < other.timestamp
        return self.score < other.score
    
    def assign(self, other):
        self.score = other.score
        self.timestamp = other.timestamp
        self.environement = other.environement

class HallOfFame:
    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = [] 
        self.signature_map = {} 
        
        # Optimization for indexing/iteration
        self._sorted_cache = [] 
        self._cache_is_dirty = True 

    def _ensure_sorted(self):
        """Internal helper to rebuild the sorted cache only when needed."""
        if self._cache_is_dirty:
            # Sort best (highest score/lowest loss) to worst
            self._sorted_cache = sorted(self.heap, key=lambda x: x.score, reverse=True)
            self._cache_is_dirty = False

    def add(self, crn_env):
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
        for env in crn_envs:
            self.add(env)

    def sample(self, batch_size):
        """
        Samples a batch of environments. No sorting needed, so it stays fast.
        """
        if not self.heap:
            return []
        
        k = min(len(self.heap), batch_size)
        samples = random.sample(self.heap, k)
        return [s.environment for s in samples]

    def __iter__(self):
        """
        Iterates from Best (-Loss is high) to Worst (-Loss is low).
        """
        self._ensure_sorted()
        for item in self._sorted_cache:
            yield item.environment
    
    def __getitem__(self, index):
        """
        Get item by rank (0 is Best).
        """
        self._ensure_sorted()
        return self._sorted_cache[index].environment

    def __len__(self):
        return len(self.heap)