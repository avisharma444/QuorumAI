"""deploy/runpod/runpod_deploy.py against a fake runpodctl and a fake REST API.
TEST-ONLY: nothing reaches RunPod."""
import importlib.util
import io
import json
import os
import stat
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "deploy/runpod/runpod_deploy.py").exists():  # evaluation/ mounted alone
    pytest.skip("deploy/ not available in this checkout layout", allow_module_level=True)
spec = importlib.util.spec_from_file_location("runpod_deploy", ROOT / "deploy/runpod/runpod_deploy.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

MGMT = "rpa_MGMTKEYforTESTS0123456789"
INFERENCE = "rpa_INFERENCEkeyMUSTnotLEAK99"
DIGEST = "sha256:6d8e4ff417e09aeffe6a55e547ed81203b02228533c5305e726caaa7dab34571"

FAKE_CLI = r'''#!/usr/bin/env python3
import json, os, sys
home = os.environ["HOME"]; st_path = os.path.join(home, "state.json")
st = json.load(open(st_path))
st["calls"].append({"argv": sys.argv[1:], "key": os.environ.get("RUNPOD_API_KEY"), "env": sorted(os.environ)})
a = [x for x in sys.argv[1:] if x not in ("-o", "json")]
def out(x): print(json.dumps(x))
def arg(name): return a[a.index(name) + 1]
if a[:1] == ["version"]:
    print("runpodctl " + st.get("version", "3.0.1") + "-abc")
elif a[:1] == ["user"]:
    out({"id": "u", "email": "x@example.com", "clientBalance": st["balance"], "currentSpendPerHr": 0, "spendLimit": 80})
elif a[:2] == ["gpu", "list"]:
    out(st["gpus"])
elif a[:2] == ["template", "list"]:
    out(st["templates"] or None)
elif a[:2] == ["template", "get"]:
    out(next(t for t in st["templates"] if t["id"] == a[2]))
elif a[:2] == ["template", "create"]:
    t = {"id": "tpl%d" % (len(st["templates"]) + 1), "name": arg("--name"), "imageName": arg("--image"),
         "isServerless": "--serverless" in a, "containerDiskInGb": int(arg("--container-disk-in-gb")), "env": json.loads(arg("--env"))}
    st["templates"].append(t); out(t)
elif a[:2] == ["template", "delete"]:
    st["templates"] = [t for t in st["templates"] if t["id"] != a[2]]; out({"deleted": True})
elif a[:2] == ["serverless", "list"]:
    out([{"id": e["id"], "name": e["name"], "workersMin": e["workersMin"], "workersMax": e["workersMax"]} for e in st["endpoints"]])
elif a[:2] == ["serverless", "health"]:
    e = next(e for e in st["endpoints"] if e["id"] == a[2])
    w = 0 if (e["workersMax"] == 0 and not st.get("stuck_workers")) else st.get("workers_now", 0)
    out({"jobs": {"inQueue": 0, "inProgress": 0}, "workers": {"idle": w, "running": 0}})
elif a[:2] == ["serverless", "update"]:
    e = next(e for e in st["endpoints"] if e["id"] == a[2])
    e["workersMin"], e["workersMax"] = int(arg("--workers-min")), int(arg("--workers-max")); out(e)
elif a[:2] == ["serverless", "delete"]:
    st["endpoints"] = [e for e in st["endpoints"] if e["id"] != a[2]]; out({"deleted": True})
else:
    print(json.dumps({"error": "unknown command " + " ".join(a), "code": "usage"})); sys.exit(1)
json.dump(st, open(st_path, "w"))
'''


class World:
    """Shared fake provider state for the CLI script and the REST opener."""

    def __init__(self, tmp: Path):
        self.home = tmp / "home"
        self.home.mkdir()
        self.cli = tmp / "runpodctl"
        self.cli.write_text(FAKE_CLI)
        self.cli.chmod(self.cli.stat().st_mode | stat.S_IEXEC)
        self.save({"calls": [], "balance": 10, "templates": [], "endpoints": [], "rest": [],
                   "gpus": [{"gpuId": g, "memoryInGb": 24, "available": True, "stockStatus": "Low"}
                            for g in ("NVIDIA L4", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090")]})

    def load(self):
        return json.loads((self.home / "state.json").read_text())

    def save(self, st):
        (self.home / "state.json").write_text(json.dumps(st))

    def update(self, **kw):
        st = self.load()
        st.update(kw)
        self.save(st)

    def opener(self, req, timeout=0):
        st = self.load()
        assert req.get_header("Authorization") == f"Bearer {MGMT}"
        assert req.get_header("User-agent") == D.USER_AGENT  # the edge blocks Python's default agent
        st["rest"].append([req.get_method(), req.full_url])
        url = req.full_url.split("?")[0]
        if req.get_method() == "POST" and url.endswith("/endpoints"):
            body = json.loads(req.data)
            ep = {"id": f"ep{len(st['endpoints']) + 1}", **body, "networkVolumeId": None}
            ep.update(st.get("server_overrides", {}))
            st["endpoints"].append(ep)
            self.save(st)
            return _Resp(200, ep)
        if req.get_method() == "POST" and url.endswith("/graphql"):
            q = json.loads(req.data)
            if "saveEndpoint" in q["query"]:
                inp = q["variables"]["input"]
                ep = next(e for e in st["endpoints"] if e["id"] == inp["id"])
                ep["allowedCudaVersions"] = inp["allowedCudaVersions"].split(",")
                st.setdefault("gql_saves", []).append(inp)
                self.save(st)
                return _Resp(200, {"data": {"saveEndpoint": self._gql_view(ep, st)}})
            ep = next(e for e in st["endpoints"] if e["id"] == q["variables"]["id"])
            self.save(st)
            return _Resp(200, {"data": {"myself": {"endpoint": self._gql_view(ep, st)}}})
        if req.get_method() == "PATCH" and "/endpoints/" in url:
            if "allowedCudaVersions" in json.loads(req.data):
                raise AssertionError("REST PATCH must not carry allowedCudaVersions (the live API rejects 13.1+)")
            ep = next(e for e in st["endpoints"] if e["id"] == url.rsplit("/", 1)[1])
            ep.update(json.loads(req.data))
            st.setdefault("patches", []).append(json.loads(req.data))
            self.save(st)
            return _Resp(200, ep)
        if req.get_method() == "GET" and "/endpoints/" in url:
            ep = next(e for e in st["endpoints"] if e["id"] == url.rsplit("/", 1)[1])
            tpl = next(t for t in st["templates"] if t["id"] == ep["templateId"])
            self.save(st)
            # Like the live API: the embedded template has no imageName.
            embedded = {k: v for k, v in tpl.items() if k != "imageName"}
            return _Resp(200, {**ep, "template": embedded, "workers": []})
        if req.get_method() == "GET" and "/templates/" in url:
            tpl = next(t for t in st["templates"] if t["id"] == url.rsplit("/", 1)[1])
            self.save(st)
            return _Resp(200, tpl)
        raise AssertionError(f"unexpected REST call {req.get_method()} {url}")


def _gql_view(ep, st):
    return {"id": ep["id"], "name": ep["name"], "templateId": ep["templateId"], "gpuIds": "ADA_24", "gpuCount": ep["gpuCount"],
            "instanceIds": [], "workersMin": ep["workersMin"], "workersMax": ep["workersMax"], "locations": "",
            "networkVolumeId": "", "networkVolumeIds": [], "idleTimeout": ep["idleTimeout"], "scalerType": ep["scalerType"],
            "scalerValue": ep["scalerValue"], "executionTimeoutMs": ep["executionTimeoutMs"], "minCudaVersion": None,
            "flashBootType": "FLASHBOOT", "allowedCudaVersions": ",".join(ep["allowedCudaVersions"]), "modelReferences": []}


World._gql_view = staticmethod(_gql_view)


class _Resp(io.BytesIO):
    def __init__(self, status, body):
        super().__init__(json.dumps(body).encode())
        self.status = status


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path)
    monkeypatch.setenv("HOME", str(w.home))
    monkeypatch.setenv("RUNPOD_API_KEY", INFERENCE)  # the benchmark's key must never reach the CLI
    return w


def cfg():
    return D.load(D.DEFAULT_CONFIG)


def campaign():
    return D.load_campaign(D.DEFAULT_CAMPAIGN)


def provider(w):
    return D.Provider(MGMT, cli=str(w.cli), opener=w.opener)


GOOD = dict(digest_fn=lambda image: DIGEST,
            hf_fn=lambda m, r: {"exists": True, "gated": False, "architectures": ["Phi3ForCausalLM"]},
            placement_fn=lambda cfg, key: {"ok": True, "match": [{"gpu": "NVIDIA GeForce RTX 4090", "cuda": ["13.2"]}]})


def test_plan_is_offline_and_explicit():
    p = D.plan(cfg(), campaign())
    body = p["endpoint_request"]["body"]
    assert body == {"name": "quorumai-phi4-bench", "templateId": "<template id from the command above>", "computeType": "GPU",
                    "gpuTypeIds": ["NVIDIA GeForce RTX 4090"], "gpuCount": 1,
                    "allowedCudaVersions": ["13.0", "13.1", "13.2", "13.3", "13.4"], "workersMin": 0, "workersMax": 1, "idleTimeout": 30,
                    "executionTimeoutMs": 120000, "flashboot": True, "scalerType": "QUEUE_DELAY", "scalerValue": 4}
    assert "networkVolumeId" not in body and "dataCenterIds" not in body
    cmd = p["template_command"]
    assert "--image runpod/worker-v1-vllm:v2.28.0" in cmd and "--container-disk-in-gb 30" in cmd and "--serverless" in cmd
    assert "cfbefacb99257ffa30c83adab238a50856ac3083" in cmd and p["deployment_cost_usd"] == 0.0
    assert p["benchmark_costs"]["total"]["worst_case_usd"] <= p["benchmark_costs"]["caps"]["max_gpu_spend_usd"]


@pytest.mark.parametrize("change,needle", [
    (lambda e: e.pop("container_disk_gb"), "missing"),
    (lambda e: e.update(image="runpod/worker-v1-vllm:latest"), "pinned"),
    (lambda e: e.update(min_workers=1), "min 0"),
    (lambda e: e.update(network_volume="nv-1"), "network volumes"),
    (lambda e: e["env"].update(MODEL_REVISION="main"), "pinned commit"),
])
def test_config_must_be_explicit_and_approved(tmp_path, change, needle):
    import yaml
    c = yaml.safe_load(D.DEFAULT_CONFIG.read_text())
    change(c["endpoint"])
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(c))
    with pytest.raises(D.Stop, match=needle):
        D.load(p)


def test_preflight_reports_problems_without_writing(world):
    world.update(balance=1)
    rep = D.preflight(cfg(), provider(world), campaign(), digest_fn=lambda i: "sha256:other",
                      hf_fn=lambda m, r: {"exists": True, "gated": True}, placement_fn=lambda c, k: {"ok": False})
    joined = " | ".join(rep["problems"])
    assert "balance" in joined and "now points to sha256:other" in joined and "gated" in joined
    assert "no Serverless host matches" in joined
    st = world.load()
    assert st["rest"] == [] and st["templates"] == [] and st["endpoints"] == []
    assert all(c["argv"][0] in ("version", "user", "gpu", "template", "serverless") and "create" not in c["argv"] for c in st["calls"])
    rep = D.preflight(cfg(), provider(world), campaign(), digest_fn=lambda i: None, hf_fn=GOOD["hf_fn"], placement_fn=GOOD["placement_fn"])
    assert any("does not exist in the registry" in p for p in rep["problems"])


def test_deploy_needs_authorisation_and_a_clean_preflight(world):
    with pytest.raises(D.Stop, match="authorize"):
        D.deploy(cfg(), provider(world), campaign(), authorize=False, **GOOD)
    world.update(gpus=[])
    with pytest.raises(D.Stop, match="preflight failed"):
        D.deploy(cfg(), provider(world), campaign(), authorize=True, **GOOD)
    assert world.load()["templates"] == [] and world.load()["endpoints"] == []


def test_deploy_creates_once_and_rerun_reuses(world, tmp_path):
    sd = tmp_path / "state"
    s1 = D.deploy(cfg(), provider(world), campaign(), authorize=True, state_dir=sd, **GOOD)
    assert s1["template_action"] == "created" and s1["endpoint_action"] == "created" and s1["settings_problems"] == []
    s2 = D.deploy(cfg(), provider(world), campaign(), authorize=True, state_dir=sd, **GOOD)
    assert s2["template_action"] == "reused" and s2["endpoint_action"] == "reused" and s2["endpoint_id"] == s1["endpoint_id"]
    st = world.load()
    assert len(st["templates"]) == 1 and len(st["endpoints"]) == 1, "re-running must not duplicate"
    assert [m for m, _ in st["rest"]].count("POST") == 1
    t = st["templates"][0]
    assert t["imageName"] == "runpod/worker-v1-vllm:v2.28.0" and t["containerDiskInGb"] == 30 and t["isServerless"]
    # Credentials: the CLI sees only the management key, in a minimal environment.
    assert {c["key"] for c in st["calls"] if c["argv"][0] != "version"} == {MGMT}
    # LC_CTYPE may be added by the interpreter itself (PEP 538 locale coercion)
    assert all(set(c["env"]) - {"LC_CTYPE"} <= {"PATH", "HOME", "RUNPOD_API_KEY"} for c in st["calls"]), {tuple(c["env"]) for c in st["calls"]}
    saved = (sd / "quorumai-phi4-bench.json").read_text()
    assert MGMT not in saved and INFERENCE not in saved and "rpa_" not in saved
    # No inference: only management routes were called.
    assert all("/v2/" not in u and "openai" not in u for _, u in st["rest"])


def test_deploy_refuses_a_same_named_template_with_other_settings(world, tmp_path):
    world.update(templates=[{"id": "old", "name": "quorumai-phi4-vllm-v2.28.0", "imageName": "runpod/worker-v1-vllm:latest",
                             "isServerless": True, "containerDiskInGb": 30, "env": {}}])
    with pytest.raises(D.Stop, match="differs"):
        D.deploy(cfg(), provider(world), campaign(), authorize=True, state_dir=tmp_path, **GOOD)
    assert world.load()["endpoints"] == []


def test_deploy_stops_when_live_settings_do_not_match(world, tmp_path):
    world.update(server_overrides={"workersMax": 3, "allowedCudaVersions": ["12.4", "13.0"]})
    with pytest.raises(D.Stop, match="workersMax.*allowedCudaVersions|settings do not match"):
        D.deploy(cfg(), provider(world), campaign(), authorize=True, state_dir=tmp_path, **GOOD)


def test_cleanup_is_idempotent_and_scales_down_first(world, tmp_path):
    sd = tmp_path / "state"
    with pytest.raises(D.Stop, match="--yes"):
        D.cleanup(cfg(), provider(world), yes=False, state_dir=sd)
    D.deploy(cfg(), provider(world), campaign(), authorize=True, state_dir=sd, **GOOD)
    world.update(workers_now=1)
    done = D.cleanup(cfg(), provider(world), yes=True, sleep=lambda s: None, state_dir=sd)
    assert done["endpoints_deleted"] == ["ep1"] and done["templates_deleted"] == ["tpl1"] and not done["nothing_to_do"]
    argv = [c["argv"][:2] for c in world.load()["calls"]]
    upd, dele = argv.index(["serverless", "update"]), argv.index(["serverless", "delete"])
    assert upd < dele < argv.index(["template", "delete"])
    again = D.cleanup(cfg(), provider(world), yes=True, sleep=lambda s: None, state_dir=sd)
    assert again["nothing_to_do"] and again["remaining"] == {"endpoints": [], "templates": []}


def test_cleanup_does_not_delete_while_workers_remain(world, tmp_path):
    D.deploy(cfg(), provider(world), campaign(), authorize=True, state_dir=tmp_path, **GOOD)
    world.update(stuck_workers=True, workers_now=1)
    with pytest.raises(D.Stop, match="still reports workers"):
        D.cleanup(cfg(), provider(world), yes=True, wait_s=0.01, sleep=lambda s: None, state_dir=tmp_path)
    assert len(world.load()["endpoints"]) == 1


def test_cli_errors_and_output_are_redacted(world):
    pv = provider(world)
    with pytest.raises(D.Stop) as exc:
        pv.ctl("bogus", "command")
    assert MGMT not in str(exc.value)
    assert D.redact(f"token {MGMT} and {INFERENCE}") == "token <redacted> and <redacted>"
    with pytest.raises(D.Stop, match="RUNPOD_MANAGEMENT_API_KEY"):
        D.Provider("")


def test_update_reconciles_gpu_types_in_place_and_verifies(world, tmp_path):
    D.deploy(cfg(), provider(world), campaign(), authorize=True, state_dir=tmp_path, **GOOD)
    st = world.load()
    st["endpoints"][0]["gpuTypeIds"] = ["NVIDIA L4", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"]  # the old approval
    world.save(st)
    cfg_old = cfg()  # first reconcile GPU types only, with CUDA as deployed
    st["endpoints"][0]["allowedCudaVersions"] = list(cfg_old["endpoint"]["allowed_cuda_versions"])
    world.save(st)
    with pytest.raises(D.Stop, match="--authorize-update"):
        D.update(cfg(), provider(world), authorize=False, state_dir=tmp_path)
    out = D.update(cfg(), provider(world), authorize=True, state_dir=tmp_path)
    assert out["changed"] == {"gpuTypeIds": {"from": ["NVIDIA L4", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"],
                                             "to": ["NVIDIA GeForce RTX 4090"]}} and out["settings_problems"] == []
    st = world.load()
    patch = st["patches"][0]
    assert set(patch) == set(D.PATCHABLE), "every approved setting is sent, not just the changed one"
    assert patch["workersMin"] == 0 and patch["workersMax"] == 1 and "allowedCudaVersions" not in patch
    assert len(st["endpoints"]) == 1 and len(st["templates"]) == 1, "update never creates or deletes"
    again = D.update(cfg(), provider(world), authorize=True, state_dir=tmp_path)
    assert again["changed"] == {} and len(world.load()["patches"]) == 1, "no PATCH when nothing differs"


def test_update_refuses_while_workers_are_up(world, tmp_path):
    D.deploy(cfg(), provider(world), campaign(), authorize=True, state_dir=tmp_path, **GOOD)
    world.update(workers_now=1)
    with pytest.raises(D.Stop, match="idle endpoint"):
        D.update(cfg(), provider(world), authorize=True, state_dir=tmp_path)


def test_update_sets_cuda_versions_through_graphql_full_replace(world, tmp_path):
    D.deploy(cfg(), provider(world), campaign(), authorize=True, state_dir=tmp_path, **GOOD)
    st = world.load()
    st["endpoints"][0]["allowedCudaVersions"] = ["13.0"]  # what was live before the fix
    world.save(st)
    out = D.update(cfg(), provider(world), authorize=True, state_dir=tmp_path)
    ch = out["changed"]["allowedCudaVersions"]
    assert ch["from"] == ["13.0"] and ch["to"] == ["13.0", "13.1", "13.2", "13.3", "13.4"]
    assert ch["graphql"] == {"before": "13.0", "after": "13.0,13.1,13.2,13.3,13.4"} and out["settings_problems"] == []
    save = world.load()["gql_saves"][0]
    # Full replace: every field was sent back unchanged apart from the CUDA list.
    assert save["gpuIds"] == "ADA_24" and save["workersMin"] == 0 and save["workersMax"] == 1 and save["templateId"] == "tpl1"
    assert save["flashBootType"] == "FLASHBOOT" and save["executionTimeoutMs"] == 120000 and save["idleTimeout"] == 30
