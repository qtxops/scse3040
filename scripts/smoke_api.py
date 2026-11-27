"""Check a running API, including its real model and client input rules."""

import argparse
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(base_url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    call = Request(base_url.rstrip("/") + path, data=data)
    if data is not None:
        call.add_header("Content-Type", "application/json")
    try:
        with urlopen(call, timeout=3) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    deadline = time.monotonic() + 60
    last_error = "API did not become ready"
    while time.monotonic() < deadline:
        try:
            status, body = request(args.base_url, "/health")
            if status == 200 and body.get("status") == "ok":
                break
            last_error = f"Health returned {status}: {body}"
        except (URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(0.5)
    else:
        raise SystemExit(f"Health check failed: {last_error}")

    payload = {
        "hour": 9, "nearby_offices": 25, "temp_c": 30.0, "rain_mm": 5.0,
        "feeder_buses": 12, "ticket_price_rs": 30.0,
        "day_type": "weekday", "line": "Red Line",
    }
    status, body = request(args.base_url, "/predict", payload)
    prediction = body.get("prediction")
    if (
        status != 200 or isinstance(prediction, bool)
        or not isinstance(prediction, (int, float))
        or not math.isfinite(prediction) or prediction < 0
    ):
        raise SystemExit(f"Prediction check failed: {status}: {body}")
    for field, value in [("rain_mm", 100.01), ("ticket_price_rs", 4.99)]:
        status, body = request(args.base_url, "/predict", {**payload, field: value})
        if status != 422:
            raise SystemExit(f"Expected 422 for {field}, received {status}: {body}")
    print(json.dumps({"health": "passed", "prediction": "passed", "client_rules": "passed"}))


if __name__ == "__main__":
    main()
