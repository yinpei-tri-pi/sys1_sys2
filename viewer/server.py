"""Browse a results dir written by sim/official_eval.py: per-task success, every episode's rollout
(played turn by turn) and every System2 turn (what it saw, its thought, judgement, subgoal, plan
update, language-steering interventions) next to the System1 segment it produced.

    sim/.venv/bin/python viewer/server.py --log-dir results/<run> [--port 8090] [--host 127.0.0.1]

Standard library only. Reads the results live, so it can watch a run in progress.
"""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import urllib.parse

HERE = Path(__file__).resolve().parent
SIM_FPS = 20

try:  # task -> leaderboard split; optional so the viewer also runs outside the sim env
    from robocasa.utils.dataset_registry import TARGET_TASKS

    GROUP_OF = {t: g for g, ts in TARGET_TASKS.items() for t in ts}
    GROUP_SIZE = {g: len(ts) for g, ts in TARGET_TASKS.items()}
except ImportError:
    GROUP_OF, GROUP_SIZE = {}, {}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class Results:
    def __init__(self, log_dir: Path):
        self.root = log_dir.resolve()
        self._cache: dict[Path, tuple[float, dict]] = {}

    def _episode_doc(self, path: Path) -> dict | None:
        """episode.json, cached by mtime (a full run has thousands)."""
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        hit = self._cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        doc = _read_json(path)
        if doc is not None:
            self._cache[path] = (mtime, doc)
        return doc

    def _media(self, ep_dir: Path, recorded: str | None, prefer_full: bool = True) -> str | None:
        """Episode-relative media path. Paths in turn.json are absolute at write time, so only the
        <turnNN>/<file> tail is trusted -- the results dir may have been copied from another host."""
        if not recorded:
            return None
        p = Path(recorded)
        cands = []
        if prefer_full and "_full" not in p.stem:
            cands.append(p.with_name(f"{p.stem}_full{p.suffix}"))
        cands.append(p)
        for c in cands:
            local = ep_dir / c.parent.name / c.name
            if local.exists():
                return str(local.relative_to(self.root))
        return None

    def summary(self) -> dict:
        tasks: dict[str, dict] = {}
        for method_dir in sorted(p for p in (self.root / "episodes").glob("*") if p.is_dir()):
            for ep_json in sorted(method_dir.glob("*/episode.json")):
                d = self._episode_doc(ep_json)
                if not d or not d.get("task_name"):
                    continue
                task = d["task_name"]
                t = tasks.setdefault(task, {"task": task, "group": GROUP_OF.get(task, "other"),
                                            "episodes": [], "official": None})
                t["episodes"].append({
                    "key": str(ep_json.parent.relative_to(self.root / "episodes")),
                    "method": method_dir.name,
                    "trial": d.get("episode_index"),
                    "success": bool(d.get("episode_success")),
                    "termination": d.get("termination"),
                    "error": d.get("error"),
                    "steps": d.get("n_env_steps"),
                    "horizon": d.get("max_official_steps"),
                    "turns": d.get("n_turns"),
                    "rules": d.get("n_rule_interventions"),
                    "seconds": d.get("seconds"),
                })
        for split_dir in sorted((self.root / "evals").glob("*")):
            for task_dir in sorted(split_dir.glob("*")):
                stamps = sorted(p for p in task_dir.glob("*") if (p / "stats.json").exists())
                if stamps and task_dir.name in tasks:
                    tasks[task_dir.name]["official"] = _read_json(stamps[-1] / "stats.json")
        for t in tasks.values():
            t["episodes"].sort(key=lambda e: (e["method"], e["trial"] if e["trial"] is not None else -1))
            n = len(t["episodes"])
            t["n"] = n
            t["n_success"] = sum(e["success"] for e in t["episodes"])
        return {"log_dir": str(self.root), "group_size": GROUP_SIZE,
                "tasks": sorted(tasks.values(), key=lambda t: (t["group"], t["task"]))}

    def episode(self, key: str) -> dict | None:
        ep_dir = (self.root / "episodes" / key).resolve()
        if self.root not in ep_dir.parents or not (ep_dir / "episode.json").exists():
            return None
        d = self._episode_doc(ep_dir / "episode.json") or {}
        plan = d.get("plan") or {}
        scene = next((str((ep_dir / "plan" / n).relative_to(self.root))
                      for n in ("scene_full.png", "scene.png") if (ep_dir / "plan" / n).exists()), None)
        turns, step = [], 0
        for td in sorted(p for p in ep_dir.glob("turn*") if (p / "turn.json").exists()):
            t = _read_json(td / "turn.json") or {}
            s2, s1, rules = t.get("s2") or {}, t.get("s1") or {}, t.get("rules") or {}
            eff = rules.get("effective") or {}
            media = s2.get("media") or {}
            n = int(s1.get("n_steps") or 0)
            video = td / "s1_rollout_raw.mp4"
            turns.append({
                "turn": t.get("turn"),
                "judge": s2.get("judge"),
                "thought": s2.get("thought"),
                "s2_subgoal": s2.get("subgoal"),
                "subgoal": eff.get("subgoal") or s2.get("subgoal"),
                "subgoal_detail": eff.get("subgoal_detail") or s2.get("subgoal_detail"),
                "estimated_step": eff.get("estimated_step", s2.get("estimated_step")),
                "plan_update": s2.get("plan_update"),
                "plan_after": t.get("plan_after"),
                "held_by_rule": s2.get("held_by_rule"),
                "privileged": s2.get("privileged"),
                "latency_s": s2.get("latency_s"),
                "response_raw": s2.get("response_raw"),
                "s2_input_video": self._media(ep_dir, (media.get("video") or {}).get("path")),
                "s2_input_image": self._media(ep_dir, (media.get("image") or {}).get("path")),
                "interventions": rules.get("interventions") or [],
                "s1_prompt": (s1.get("s1_prompts") or [None])[0],
                "s1_steps": n,
                "s1_stop": s1.get("stop_reason"),
                "s1_budget": s1.get("budget"),
                "success_step": s1.get("success_step"),
                "step_start": step,
                "s1_video": str(video.relative_to(self.root)) if video.exists() and n else None,
                "anchor": str((td / "s1_anchor.png").relative_to(self.root)) if (td / "s1_anchor.png").exists() else None,
            })
            step += n
        return {
            "key": key, "episode_id": d.get("episode_id"), "task": d.get("task_name"),
            "trial": d.get("episode_index"),
            "instruction": d.get("instruction"), "success": bool(d.get("episode_success")),
            "termination": d.get("termination"), "error": d.get("error"),
            "steps": d.get("n_env_steps"), "horizon": d.get("max_official_steps"),
            "n_turns": d.get("n_turns"), "seconds": d.get("seconds"), "rule_tier": d.get("rule_tier"),
            "n_rule_interventions": d.get("n_rule_interventions"),
            "reset_mode": (d.get("config") or {}).get("reset_mode"),
            "plan": {"thought": plan.get("thought"), "plan": plan.get("plan"), "scene": scene},
            "final_plan": d.get("final_plan"), "sim_fps": SIM_FPS, "turns": turns,
        }


class Handler(BaseHTTPRequestHandler):
    results: Results

    def log_message(self, fmt, *args):  # quiet: media range requests are chatty
        pass

    def _json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:  # suffix range: the last N bytes
                start = max(0, size - int(m.group(2)))
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser cancels range reads while seeking

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._file(HERE / "index.html")
        if url.path == "/api/summary":
            return self._json(self.results.summary())
        if url.path == "/api/episode":
            key = urllib.parse.parse_qs(url.query).get("key", [""])[0]
            ep = self.results.episode(key)
            return self._json(ep) if ep else self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        if url.path.startswith("/files/"):
            rel = urllib.parse.unquote(url.path[len("/files/"):])
            path = (self.results.root / rel).resolve()
            if self.results.root in path.parents and path.is_file():
                return self._file(path)
        self.send_error(HTTPStatus.NOT_FOUND)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log-dir", type=Path, required=True, help="the --log-dir of an evaluation run")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve on the network")
    args = ap.parse_args()
    if not (args.log_dir / "episodes").is_dir():
        raise SystemExit(f"{args.log_dir} has no episodes/ (pass the --log-dir of an evaluation run)")
    Handler.results = Results(args.log_dir)
    print(f"viewer: http://{args.host}:{args.port}/   ({Handler.results.root})", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
