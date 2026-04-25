import json

avail = json.load(open("/data/freqtrade_data/_meta/pair_availability.json"))
cutoff_start = "2024-04-25"

mainstream = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "SOL/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "DOT/USDT:USDT", "LINK/USDT:USDT", "UNI/USDT:USDT", "LTC/USDT:USDT",
    "NEAR/USDT:USDT", "FIL/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT",
    "ATOM/USDT:USDT", "APT/USDT:USDT", "SUI/USDT:USDT", "TRX/USDT:USDT",
    "ETC/USDT:USDT", "AAVE/USDT:USDT", "MKR/USDT:USDT", "INJ/USDT:USDT",
    "SEI/USDT:USDT", "FTM/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT",
    "ONDO/USDT:USDT", "ORDI/USDT:USDT", "WLD/USDT:USDT", "JUP/USDT:USDT",
    "BONK/USDT:USDT", "RENDER/USDT:USDT", "1000PEPE/USDT:USDT",
]

selected = []
for pair in mainstream:
    pinfo = avail["pairs"].get(pair, {})
    tf5m = pinfo.get("timeframes", {}).get("5m", {})
    if tf5m:
        start = tf5m["start"][:10]
        end = tf5m["end"][:10]
        if start <= cutoff_start:
            selected.append((pair, start, end))
            if len(selected) >= 30:
                break

print(f"Selected {len(selected)} pairs:")
for p, s, e in selected:
    print(f"  {p}: {s} -> {e}")

print("\nJSON pair list:")
print(json.dumps([p for p, _, _ in selected], indent=2))
