"""Collect a resumable public-Ethereum verified-contract address pool from Blockscout."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "https://eth.blockscout.com/api/v2/smart-contracts"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--target-addresses", type=int, default=100000)
    parser.add_argument("--items-count", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=0.38)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--output",
        default="data/raw/ethereum_public_pretrain_100k/blockscout_verified_addresses.jsonl",
    )
    parser.add_argument(
        "--state",
        default="data/raw/ethereum_public_pretrain_100k/blockscout_verified_addresses_state.json",
    )
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument(
        "--initial-smart-contract-id",
        type=int,
        default=None,
        help="Start below this Blockscout cursor when creating a fresh state file.",
    )
    return parser.parse_args()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_existing(path):
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {
            json.loads(line).get("address", "").lower()
            for line in handle
            if line.strip()
        }


def load_state(path):
    if not path.exists():
        return {"next_page_params": None, "pages_completed": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def request_page(session, api_url, params, retries):
    error = None
    for attempt in range(retries):
        try:
            response = session.get(api_url, params=params, timeout=45)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload.get("items"), list):
                raise ValueError("Blockscout payload has no items list")
            return payload
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
            time.sleep(min(30.0, 2**attempt))
    raise RuntimeError(error or "Blockscout request failed")


def normalize_item(item):
    address = ((item.get("address") or {}).get("hash") or "").lower()
    if not address:
        return None
    address_data = item.get("address") or {}
    return {
        "id": f"ethereum_blockscout:{address}",
        "address": address,
        "source": "Ethereum mainnet Blockscout verified-contract index",
        "blockscout_retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "compiler_version": item.get("compiler_version"),
        "language": item.get("language"),
        "contract_name": address_data.get("name"),
        "proxy_type": address_data.get("proxy_type"),
        "is_verified": address_data.get("is_verified"),
        "verified_at": item.get("verified_at"),
    }


def main():
    args = parse_args()
    output_path, state_path = resolve(args.output), resolve(args.state)
    seen = load_existing(output_path)
    state = load_state(state_path)
    if args.initial_smart_contract_id is not None and not state_path.exists():
        state["next_page_params"] = {
            "items_count": args.items_count,
            "smart_contract_id": args.initial_smart_contract_id,
        }
    if len(seen) >= args.target_addresses:
        print(f"[OK] already collected {len(seen)} addresses")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    pages_this_run = 0
    while len(seen) < args.target_addresses:
        if args.max_pages is not None and pages_this_run >= args.max_pages:
            break
        params = {"items_count": args.items_count}
        if state.get("next_page_params"):
            params.update(state["next_page_params"])
        payload = request_page(session, args.api_url, params, args.retries)
        rows = []
        for item in payload["items"]:
            row = normalize_item(item)
            if row and row["address"] not in seen and len(seen) + len(rows) < args.target_addresses:
                seen.add(row["address"])
                rows.append(row)
        with output_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        state = {
            "api_url": args.api_url,
            "target_addresses": args.target_addresses,
            "pages_completed": state.get("pages_completed", 0) + 1,
            "next_page_params": payload.get("next_page_params"),
            "addresses_collected": len(seen),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_state(state_path, state)
        pages_this_run += 1
        print(f"[INFO] pages={state['pages_completed']} addresses={len(seen)}")
        if not payload.get("next_page_params"):
            break
        time.sleep(args.sleep_seconds)
    print(f"[OK] collected {len(seen)} addresses to {output_path}")


if __name__ == "__main__":
    main()
