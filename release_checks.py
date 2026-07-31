"""Pre-publish checks shared by the miniav / miniav_tools / minigpu release scripts.

Every check here exists because the failure it catches actually shipped. None of
them are hypothetical, and all of them share one property: **they are invisible
in-repo**. Local development resolves through `pubspec_overrides.yaml`, which
replaces every sibling constraint with a path -- so a constraint can be years
stale, or name a version that never existed, and every local build, test and
`pub get` still passes. The first report is a consumer's version solve failing.

That is why these run against pub.dev rather than against the working tree.

Drop this file next to release.py and call `preflight(root_dir, PACKAGES)`.
"""
import json
import os
import re
import subprocess
import urllib.request

# Constraints on these are checked against what is actually published. A stale
# constraint on a package in the family is an error rather than a warning,
# because the family releases together and a lagging pin blocks every consumer
# of both halves -- cross-REPO is the case that bites, since no single release.py
# sees both sides of it.
FAMILY_PREFIXES = ("miniav", "minigpu", "gpu_tensor", "gpu_pipeline", "gpu_ml")

_cache = {}


def _pub(package):
    """Latest published version + its pubspec, or None if unpublished."""
    if package in _cache:
        return _cache[package]
    try:
        with urllib.request.urlopen(
                f"https://pub.dev/api/packages/{package}", timeout=15) as r:
            d = json.loads(r.read().decode())
        _cache[package] = (d["latest"]["version"], d["latest"]["pubspec"])
    except Exception:
        _cache[package] = None
    return _cache[package]


def _versions(package):
    try:
        with urllib.request.urlopen(
                f"https://pub.dev/api/packages/{package}", timeout=15) as r:
            d = json.loads(r.read().decode())
        return {v["version"]: v["pubspec"] for v in d["versions"]}
    except Exception:
        return {}


def _local_version(root, pkg):
    p = os.path.join(root, pkg, "pubspec.yaml")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        m = re.search(r'^version:\s*(\S+)', f.read(), re.M)
    return m.group(1) if m else None


def _deps(root, pkg):
    """{name: constraint} for hosted deps only -- path/sdk/git entries are skipped."""
    p = os.path.join(root, pkg, "pubspec.yaml")
    out, section = {}, None
    with open(p, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for i, line in enumerate(lines):
        if re.match(r'^[a-zA-Z_]+:', line):
            section = line.split(":")[0]
            continue
        if section not in ("dependencies", "dev_dependencies"):
            continue
        m = re.match(r'^  ([a-z0-9_]+):\s*(\S.*)?$', line)
        if not m:
            continue
        name, constraint = m.group(1), (m.group(2) or "").strip()
        if not constraint:  # a block form: path:/sdk:/git: on the next line
            continue
        out[name] = constraint
    return out


def _satisfies(constraint, version):
    """Does `constraint` admit `version`? Handles exact and caret only.

    Deliberately narrow: anything else returns None meaning "not evaluated",
    so an unusual constraint is reported as unknown rather than silently
    passed. Guessing here would be worse than declining to answer.
    """
    c, v = constraint.strip(), version.strip()
    if c == v:
        return True
    if re.fullmatch(r'\d+\.\d+\.\d+\S*', c):
        return False  # a bare version is an exact pin, and it did not match
    if not c.startswith("^"):
        return None
    def parts(s):
        return [int(x) for x in re.split(r'[.+-]', s)[:3] if x.isdigit()]
    lo, got = parts(c[1:]), parts(v)
    if len(lo) < 3 or len(got) < 3:
        return None
    if got < lo:
        return False
    # pub's caret: <1.0.0 pins the leading non-zero, so ^0.5.3 means <0.6.0.
    if lo[0] == 0:
        return got[0] == 0 and got[1] == lo[1]
    return got[0] == lo[0]


def check_internal_pins(root, packages):
    """Sibling pins must be caret and must admit the sibling's on-disk version.

    Exact pins are the cascade bug: publishing platform_interface 0.5.3 made the
    already-published miniav_tools 0.5.3 unsatisfiable beside it, because it
    pinned 0.5.2 exactly and nothing in the set could move independently.
    """
    errs = []
    local = {p: _local_version(root, p) for p in packages}
    for pkg in packages:
        if not local.get(pkg):
            continue
        for dep, constraint in _deps(root, pkg).items():
            if dep not in local or dep == pkg or not local[dep]:
                continue
            if not constraint.startswith("^"):
                errs.append(f"{pkg}: pins {dep} as '{constraint}' -- use a caret "
                            f"range (^{local[dep]}); exact pins cascade")
            elif _satisfies(constraint, local[dep]) is False:
                errs.append(f"{pkg}: pins {dep} '{constraint}' but {dep} is at "
                            f"{local[dep]} on disk -- run `release.py sync`")
    return errs


def check_external_constraints(root, packages):
    """Family constraints must admit what is currently PUBLISHED.

    This is the cross-repo check no single release.py could do before. It is how
    minigpu_view sat on `miniav: ^0.5.2` while miniav shipped 0.7.0, silently
    blocking every consumer of both -- for as long as it took someone to try.
    """
    errs = []
    local = set(packages)
    for pkg in packages:
        if not _local_version(root, pkg):
            continue
        for dep, constraint in _deps(root, pkg).items():
            if dep in local or not dep.startswith(FAMILY_PREFIXES):
                continue
            info = _pub(dep)
            if not info:
                continue
            latest, _ = info
            if _satisfies(constraint, latest) is False:
                errs.append(f"{pkg}: constrains {dep} to '{constraint}' but "
                            f"{dep} {latest} is published -- a consumer of both "
                            f"cannot resolve")
    return errs


def check_version_bumped(root, packages):
    """A published version number must never describe two different pubspecs.

    minigpu_view's repo copy of 1.5.9 had newer pins than the 1.5.9 on pub.dev:
    someone fixed the constraints and never bumped, so the fix could not be
    published and nobody could tell from the version alone.
    """
    errs = []
    for pkg in packages:
        v = _local_version(root, pkg)
        if not v:
            continue
        published = _versions(pkg)
        if v not in published:
            continue  # unpublished version -- exactly right for a release
        want = published[v].get("dependencies", {}) or {}
        have = {k: c for k, c in _deps(root, pkg).items()}
        diff = [k for k in set(want) | set(have)
                if isinstance(want.get(k), str) and want.get(k) != have.get(k)]
        if diff:
            errs.append(f"{pkg}: {v} is already published with DIFFERENT "
                        f"dependencies ({', '.join(sorted(diff)[:4])}) -- bump "
                        f"the version, the fix cannot ship under {v}")
    return errs


def check_publish_validation(root, packages):
    """`pub publish --dry-run`, failing on ERRORS only.

    Warnings are tolerated on purpose: split plugins legitimately produce them
    (an endorsed implementation, a platform-interface-only package), and
    refusing on warnings is why publishing runs with validation skipped in the
    first place. Errors are a different animal -- miniav_tools_codecs 0.6.1
    shipped past four of them and was unusable as a dependency, because its
    build hook imported packages it never declared, and a consumer does not get
    dev_dependencies. Same defect is live in minigpu_ffi.
    """
    errs = []
    for pkg in packages:
        d = os.path.join(root, pkg)
        if not os.path.exists(os.path.join(d, "pubspec.yaml")):
            continue
        try:
            # shell=True on Windows: `dart` is dart.bat and CreateProcess
            # will not find it otherwise.
            r = subprocess.run("dart pub publish --dry-run", shell=True,
                               cwd=d, capture_output=True, text=True, timeout=300)
        except Exception as e:
            errs.append(f"{pkg}: could not run publish --dry-run ({e})")
            continue
        out = (r.stdout or "") + (r.stderr or "")
        m = re.search(r'found the following (\d+) error', out)
        if not m:
            continue
        detail = [l.strip() for l in out.split("\n")
                  if l.startswith("* ")][:int(m.group(1))]
        errs.append(f"{pkg}: {m.group(1)} publish validation ERROR(s):\n      " +
                    "\n      ".join(detail))
    return errs


def preflight(root, packages, skip_validation=False):
    """Run every check. Returns True when it is safe to publish."""
    checks = [
        ("internal pins", check_internal_pins),
        ("published family constraints", check_external_constraints),
        ("version already published", check_version_bumped),
    ]
    if not skip_validation:
        checks.append(("publish validation", check_publish_validation))

    failed = False
    for label, fn in checks:
        print(f"\n[preflight] {label} ...")
        try:
            errs = fn(root, packages)
        except Exception as e:
            print(f"  SKIPPED -- check itself failed: {e}")
            continue
        if errs:
            failed = True
            for e in errs:
                print(f"  ERROR  {e}")
        else:
            print("  ok")

    print()
    if failed:
        print("PREFLIGHT FAILED -- fix the above before publishing.")
        print("Each of these has shipped broken at least once; none of them are")
        print("visible from inside the repo, because pubspec_overrides.yaml")
        print("hides every sibling constraint behind a local path.")
    else:
        print("PREFLIGHT PASSED.")
    return not failed


def discover_family_deps(root, packages):
    """Every family package these pubspecs actually depend on, from disk.

    `deps` used to iterate a hand-maintained EXTERNAL_DEPS list, which silently
    skipped anything missing from it -- exactly how miniav_player sat on a stale
    `minigpu_view` constraint while every other family dep was kept current. A
    list that must be updated by hand to stay correct will eventually be wrong,
    and the failure is invisible: the command reports success for the packages
    it did look at.

    Reading the pubspecs makes the list impossible to drift. Managed packages
    are excluded -- those are internal pins, and `sync` owns them.
    """
    found = set()
    for pkg in packages:
        path = os.path.join(root, pkg, "pubspec.yaml")
        if not os.path.exists(path):
            continue
        for dep in _deps(root, pkg):
            if dep.startswith(FAMILY_PREFIXES) and dep not in packages:
                found.add(dep)
    return sorted(found)
