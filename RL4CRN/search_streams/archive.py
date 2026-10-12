"""Correct utility ordering without modifying the frozen synchronous archive."""

from RL4CRN.search_exploration.archive import DiscoveryArchive


class StreamDiscoveryArchive(DiscoveryArchive):
    def _eviction_priority(self, record):
        representative = self.region_best.get(record["region"], {}).get("topology_hash") == record["topology_hash"]
        # A min-heap evicts the smallest key: low utility must precede high utility.
        return (int(representative), self.utility(record), -record["loss"])
