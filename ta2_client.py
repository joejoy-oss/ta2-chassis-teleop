#!/usr/bin/env python3
"""TA2 HTTP JSON client (Python 3.8+, standard library only).

    from ta2_client import TA2Client, TA2ActionFailed
    c = TA2Client("http://192.168.31.228:8088", timeout=5)
    aid = c.start("route.run", route="g2r")
    record = c.wait(aid, timeout=300, poll=0.5, raise_on_fail=True)
    record = c.arm_pose("start", wait=True)

Waiting timeouts never stop the robot. A route accepts only estop(), including
while waiting between paths. Software E-STOP does not replace hardware E-STOP.
CLI: python3 ta2_client.py [--url http://host:8088] state|estop
"""
import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class TA2Error(Exception):
    def __init__(self, code, message, http_status=None):
        self.code, self.message, self.http_status = code, message, http_status
        super().__init__("{}: {}".format(code, message))


class TA2Timeout(TA2Error):
    def __init__(self, message):
        super().__init__("TIMEOUT", message)


class TA2ActionFailed(TA2Error):
    def __init__(self, record):
        self.record = record
        error = record.get("error") or {}
        super().__init__(error.get("code", "ACTION_FAILED"),
                         error.get("message", "action ended: " + record["status"]))


class TA2Client:
    def __init__(self, base_url, timeout=5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method, path, body=None, timeout=None):
        data = json.dumps(body).encode("utf-8") if body is not None else (b"" if method == "POST" else None)
        request = Request(self.base_url + "/api/v1" + path, data=data, method=method,
                          headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout if timeout is None else timeout) as response:
                return json.load(response)
        except HTTPError as e:
            try:
                error = json.load(e).get("error", {})
                code, message = error["code"], error["message"]
            except (ValueError, KeyError, TypeError):
                code, message = "HTTP_ERROR", str(e)
            raise TA2Error(code, message, e.code) from e
        except (URLError, OSError) as e:
            raise TA2Error("CONNECTION_ERROR", str(e)) from e

    def health(self):
        return self._request("GET", "/health")

    def state(self):
        return self._request("GET", "/state")

    def capabilities(self):
        return self._request("GET", "/capabilities")

    def grippers(self):
        return self._request("GET", "/grippers")

    def preflight(self, model=None):
        query = "?" + urlencode({"model": model}) if model is not None else ""
        return self._request("GET", "/preflight" + query)

    def models(self):
        return self._request("GET", "/models")

    def actions(self, limit=20):
        return self._request("GET", "/actions?" + urlencode({"limit": limit}))["actions"]

    def action(self, aid):
        return self._request("GET", "/actions/" + quote(aid, safe=""))

    def start(self, action_type, **params):
        return self._request("POST", "/actions", {"type": action_type, "params": params})["action_id"]

    def wait(self, aid, timeout=300, poll=0.5, raise_on_fail=False):
        if timeout <= 0 or poll <= 0:
            raise ValueError("timeout and poll must be positive")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TA2Timeout("waiting for action {} timed out; robot was not stopped".format(aid))
            try:
                record = self._request("GET", "/actions/" + quote(aid, safe=""),
                                       timeout=min(self.timeout, remaining))
            except TA2Error as e:
                if e.code == "CONNECTION_ERROR" and time.monotonic() >= deadline:
                    raise TA2Timeout("waiting for action {} timed out; robot was not stopped".format(aid)) from e
                raise
            if record["status"] != "running":
                if raise_on_fail and record["status"] != "succeeded":
                    raise TA2ActionFailed(record)
                return record
            time.sleep(min(poll, max(0, deadline - time.monotonic())))

    def _run(self, kind, params, wait, timeout, poll, raise_on_fail):
        aid = self.start(kind, **params)
        return self.wait(aid, timeout, poll, raise_on_fail) if wait else aid

    def run_route(self, route, wait=False, timeout=300, poll=0.5, raise_on_fail=False):
        return self._run("route.run", {"route": route}, wait, timeout, poll, raise_on_fail)

    def run_path(self, name, wait=False, timeout=300, poll=0.5, raise_on_fail=False):
        return self._run("path.run", {"name": name}, wait, timeout, poll, raise_on_fail)

    def run_all(self, wait=False, timeout=300, poll=0.5, raise_on_fail=False):
        return self._run("path.run_all", {}, wait, timeout, poll, raise_on_fail)

    def arm_pose(self, pose, wait=False, timeout=300, poll=0.5, raise_on_fail=False):
        return self._run("arm.pose", {"pose": pose}, wait, timeout, poll, raise_on_fail)

    def arm_down(self, hand, cm, wait=False, timeout=300, poll=0.5, raise_on_fail=False):
        return self._run("arm.down", {"hand": hand, "cm": cm}, wait, timeout, poll, raise_on_fail)

    def vla_grasp(self, model=None, seconds=None, stop_on_grasp=False, wait=False,
                  timeout=300, poll=0.5, raise_on_fail=False):
        params = {k: v for k, v in (("model", model), ("seconds", seconds)) if v is not None}
        params["stop_on_grasp"] = stop_on_grasp
        return self._run("vla.grasp", params, wait, timeout, poll, raise_on_fail)

    def run_mission(self, attempts=None, prompt=None, port=None, wait=False, timeout=1800,
                    poll=0.5, raise_on_fail=False, flow=None):
        """flow: None/"v1" = steps 1-6, "v2" = steps 1-7 (adds the voice interaction module), "voice" = step 7 only."""
        params = {k: v for k, v in (("attempts", attempts), ("prompt", prompt), ("port", port), ("flow", flow))
                  if v is not None}
        return self._run("mission.run", params, wait, timeout, poll, raise_on_fail)

    def step(self, direction, wait=False, timeout=300, poll=0.5, raise_on_fail=False):
        return self._run("chassis.step", {"direction": direction}, wait, timeout, poll, raise_on_fail)

    def reset_odom(self, wait=False, timeout=300, poll=0.5, raise_on_fail=False):
        return self._run("odom.reset", {}, wait, timeout, poll, raise_on_fail)

    def stop(self):
        return self._request("POST", "/stop")

    def estop(self):
        return self._request("POST", "/estop")

    def cancel(self, aid):
        return self._request("POST", "/actions/" + quote(aid, safe="") + "/cancel")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://192.168.31.228:8088")
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("command", choices=("health", "state", "capabilities", "actions", "stop", "estop"))
    args = parser.parse_args()
    try:
        print(json.dumps(getattr(TA2Client(args.url, args.timeout), args.command)(),
                         ensure_ascii=False, indent=2))
    except TA2Error as e:
        parser.exit(1, str(e) + "\n")


if __name__ == "__main__":
    main()
