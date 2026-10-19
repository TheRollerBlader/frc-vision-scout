"""Balanced (stratified) sampling engine for FRC match videos across years and locations."""

import collections
import random
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

from progress import MicroProgressTracker
from tba_client import TBAClient


@dataclass
class MatchVideoRecord:
    year: int
    event_key: str
    event_name: str
    location: str
    match_key: str
    comp_level: str
    match_number: int
    youtube_id: str
    youtube_url: str

    def to_dict(self) -> dict:
        return asdict(self)


def round_robin_sample(
    items_by_group: Dict[str, List[MatchVideoRecord]],
    target_count: int,
    seed: Optional[int] = 42,
) -> List[MatchVideoRecord]:
    """Evenly samples up to target_count records across groups using fair round-robin.
    
    If some groups have fewer items than others, remaining quota is distributed
    evenly among the groups with remaining items.
    """
    if target_count <= 0 or not items_by_group:
        return []

    rng = random.Random(seed)

    # Make randomized copies of pools
    pools: Dict[str, List[MatchVideoRecord]] = {}
    for group_name, items in items_by_group.items():
        if items:
            copied = list(items)
            rng.shuffle(copied)
            pools[group_name] = copied

    selected: List[MatchVideoRecord] = []
    # Shuffle groups so there is no alphabetical bias towards 'A' or numbers
    active_groups = list(pools.keys())
    rng.shuffle(active_groups)

    while len(selected) < target_count and active_groups:
        next_active = []
        for g in active_groups:
            if len(selected) >= target_count:
                break
            if pools[g]:
                selected.append(pools[g].pop(0))
                if pools[g]:
                    next_active.append(g)
        active_groups = next_active

    return selected


class FRCSampler:
    """Manages discovery and balanced sampling of FRC match videos."""

    def __init__(
        self,
        client: TBAClient,
        progress_callback: Optional[Callable[[str], None]] = None,
        tracker: Optional[MicroProgressTracker] = None,
    ):
        self.client = client
        self.tracker = tracker
        if progress_callback is not None:
            self.log = progress_callback
        elif tracker is not None:
            self.log = tracker.log
        else:
            self.log = print

    def discover_matches_for_year(
        self,
        year: int,
        max_events_per_location: int = 4,
        max_matches_per_event: int = 6,
    ) -> Dict[str, List[MatchVideoRecord]]:
        """Discovers matches with YouTube videos for a given year, grouped by location."""
        if self.tracker:
            self.tracker.start_item(
                phase_label="TBA Discovery",
                item_label=f"Season {year}",
                weight=4.0,
                sub_detail="Fetching events list...",
            )

        self.log(f"[TBA] Fetching events for season {year}...")
        events = self.client.get_events(year, official_only=True)
        if not events:
            self.log(f"[Warning] No official events found for year {year}.")
            if self.tracker:
                self.tracker.finish_item(sub_detail="No events")
            return {}

        # Group events by geographical location
        events_by_loc: Dict[str, List[dict]] = collections.defaultdict(list)
        for ev in events:
            loc = self.client.format_location(ev)
            events_by_loc[loc].append(ev)

        self.log(f"[TBA] Found {len(events)} events across {len(events_by_loc)} distinct locations for {year}.")

        matches_by_loc: Dict[str, List[MatchVideoRecord]] = collections.defaultdict(list)

        # For each location, check events until candidate matches with videos are gathered
        sorted_locations = sorted(events_by_loc.keys())
        total_locs = max(1, len(sorted_locations))

        for loc_idx, loc in enumerate(sorted_locations, start=1):
            if self.tracker:
                self.tracker.update_item(
                    fraction=(loc_idx - 1) / total_locs,
                    sub_detail=f"Scanning {loc} ({loc_idx}/{total_locs} locations)",
                )

            loc_events = events_by_loc[loc]
            # Shuffle events within location to avoid bias towards alphabetical event keys
            random.Random(year).shuffle(loc_events)

            checked_events = 0
            for ev in loc_events:
                if checked_events >= max_events_per_location:
                    break

                ev_key = ev.get("key", "")
                ev_name = ev.get("name", ev_key)
                matches = self.client.get_event_matches(ev_key)
                if not matches:
                    continue

                checked_events += 1
                ev_matches_with_video: List[MatchVideoRecord] = []

                for m in matches:
                    videos = self.client.extract_youtube_videos(m)
                    if not videos:
                        continue

                    # Primary YouTube video
                    yt = videos[0]
                    rec = MatchVideoRecord(
                        year=year,
                        event_key=ev_key,
                        event_name=ev_name,
                        location=loc,
                        match_key=m.get("key", ""),
                        comp_level=m.get("comp_level", ""),
                        match_number=m.get("match_number", 0),
                        youtube_id=yt["id"],
                        youtube_url=yt["url"],
                    )
                    ev_matches_with_video.append(rec)

                if ev_matches_with_video:
                    random.Random(ev_key).shuffle(ev_matches_with_video)
                    matches_by_loc[loc].extend(ev_matches_with_video[:max_matches_per_event])

        total_found = sum(len(v) for v in matches_by_loc.values())
        self.log(f"[TBA] Gathered {total_found} candidate videos across {len(matches_by_loc)} locations for {year}.")
        if self.tracker:
            self.tracker.finish_item(sub_detail=f"{total_found} candidates found")
        return matches_by_loc

    def sample_this_year(
        self,
        year: int,
        target_count: int = 100,
        seed: int = 42,
    ) -> List[MatchVideoRecord]:
        """Samples match videos for this year evenly across all available locations."""
        self.log(f"\n=== Sampling This Year ({year}) — Target: {target_count} videos ===")
        matches_by_loc = self.discover_matches_for_year(year)
        if not matches_by_loc:
            return []

        sampled = round_robin_sample(matches_by_loc, target_count, seed=seed)
        self.log(f"[Sampled] Selected {len(sampled)} videos across {len(set(r.location for r in sampled))} locations.")
        return sampled

    def sample_multi_year(
        self,
        years: List[int],
        target_count: int = 100,
        seed: int = 42,
    ) -> List[MatchVideoRecord]:
        """Samples match videos evenly across multiple years, and within each year evenly across locations."""
        self.log(f"\n=== Sampling Multi-Year ({years}) — Target: {target_count} videos ===")
        if not years:
            return []

        # Discover match video pools for each year
        year_pools: Dict[int, Dict[str, List[MatchVideoRecord]]] = {}
        valid_years = []
        for y in years:
            self.log(f"\n--- Checking Season {y} ---")
            pool = self.discover_matches_for_year(y)
            year_pools[y] = pool
            if pool:
                valid_years.append(y)
            else:
                self.log(f"[Notice] Year {y} has no match videos on TBA (e.g. COVID season). Quota reallocated to other years.")

        if not valid_years:
            self.log("[Error] No match videos found in any of the requested years.")
            return []

        # Divide target count evenly across valid years
        base_per_year = target_count // len(valid_years)
        remainder = target_count % len(valid_years)

        all_sampled: List[MatchVideoRecord] = []
        for i, y in enumerate(valid_years):
            quota = base_per_year + (1 if i < remainder else 0)
            year_sample = round_robin_sample(year_pools[y], quota, seed=seed + y)
            self.log(f"[Year {y}] Selected {len(year_sample)} videos across {len(set(r.location for r in year_sample))} locations.")
            all_sampled.extend(year_sample)

        # If any year fell short of its quota, redistribute deficit among years that have remaining pool
        deficit = target_count - len(all_sampled)
        if deficit > 0:
            already_selected_keys = {r.match_key for r in all_sampled}
            remaining_pools: Dict[str, List[MatchVideoRecord]] = collections.defaultdict(list)
            for y in valid_years:
                for loc, records in year_pools[y].items():
                    unpicked = [r for r in records if r.match_key not in already_selected_keys]
                    if unpicked:
                        remaining_pools[f"{y}_{loc}"].extend(unpicked)
            if remaining_pools:
                extra = round_robin_sample(remaining_pools, deficit, seed=seed + 999)
                all_sampled.extend(extra)

        self.log(f"[Multi-Year Total] Selected {len(all_sampled)}/{target_count} videos across {len(valid_years)} seasons.")
        return all_sampled
