# Review — three defects in the cache lifecycle

## 1. Stale cache is served forever, silently
Truncating the cache to 2014-12-31 made `load_prices()` return 1,818 rows
ending 2014 without a murmur. There is no age check and no top-up —
`data.py` returns any non-empty cache. Whoever runs this next month gets
today's data and will not know. Only `refresh=True` escapes.

## 2. A corrupt cache is unrecoverable
`_read_cache` guards exists, empty and missing-close, but
`pd.read_parquet` on a truncated or garbage file raises `ArrowInvalid`
straight out of `data.py` with no fall-through to download. An
interrupted write bricks the module until someone deletes the file by
hand.

## 3. A provisional last bar can be frozen in permanently
Yahoo's final daily row is the live quote — `Close` on the last row
equals `fast_info.lastPrice`, and its volume is a verbatim copy of the
prior day's. Run this mid-session and you cache an intraday price as a
settlement; per issue 1 it then never gets corrected.
