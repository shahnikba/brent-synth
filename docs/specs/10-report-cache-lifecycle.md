# Report — cache lifecycle fixes

## Status
All three fixed. 77 tests passing (15 new, all offline).

## 1. Stale cache — fixed
The cache is stamped with its UTC fetch date in parquet schema metadata
and trusted only for that calendar day. At most one network call per day.
A cache written before the key existed reads back as unknown age, so the
existing cache healed itself on first use.

Repro, before and after: 1,818 rows ending 2014-12-31 -> 4754 rows ending
2026-09-03.

A failed refresh no longer raises: with a cache present it returns stale
data with a RuntimeWarning naming the last date it holds.

## 2. Corrupt cache — fixed
Every read failure warns and falls through to a download. Writes are
atomic (`.tmp` + `os.replace`), so an interrupted write leaves the
previous cache intact.

## 3. Provisional bar — fixed, with a correction to the diagnosis
Confirmed provisional: Close 96.279999 equals `fast_info.lastPrice`, and
Volume 48231 is a copy of the prior day.

**But the run was on a Saturday and the bar is dated the preceding
Friday.** A "drop bars dated today" rule misses it entirely. Detection
has to be the duplicated volume. Base rate: 69 of 4755 rows (1.45%), and
only the final row is ever tested — a false positive costs nothing
lasting, since the row is kept once it has a successor.

Series now ends 2026-09-03 at 95.52 (the real settle) instead of
2026-09-04 at 96.28 (the live quote).

## Downstream
4754 prices / 4753 returns. Model refit unchanged in substance.
