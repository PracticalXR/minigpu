"""Pre-publish checks shared by the miniav / miniav_tools / minigpu release scripts.

Every check here exists because the failure it catches actually shipped. None of
them are hypothetical, and all of them share one property: **they are invisible
in-repo**. Local development resolves through `pubspec_overrides.yaml`, which
replaces every sibling constraint with a path -- so a constraint can be years
stale, or name a version that never existed, and every local build, test and
`pub get` still passes. The first report is a consumer's version solve failing.

That is why these run against pub.dev rather than against the working tree.

A second property is newer and was learned the hard way: **a check that cannot
answer must fail, not pass**. Every silent path in this file -- an unparsed
constraint, an unreachable pub.dev, a `--dry-run` whose wording changed, a check
that raised -- used to print `ok`. A green preflight that means "nothing was
actually verified" is worse than no preflight, because it is trusted.

Drop this file next to release.py and call `preflight(root_dir, PACKAGES)`.
"""
import json
import os
import re
import subprocess
import urllib.error
import urllib.request

# Constraints on these are checked against what is actually published. A stale
# constraint on a package in the family is an error rather than a warning,
# because the family releases together and a lagging pin blocks every consumer
# of both halves -- cross-REPO is the case that bites, since no single release.py
# sees both sides of it.
FAMILY_PREFIXES = ("miniav", "minigpu", "gpu_tensor", "gpu_pipeline", "gpu_ml")

_cache = {}


class FetchError(Exception):
    """pub.dev could not be asked -- which is not the same as "not published".

    Both used to collapse into the same value: `_pub` cached None on any
    exception and its callers read that as "nothing published, nothing to
    check", and `_versions` returned an empty dict. Run preflight with no
    network and every check that consults pub.dev printed `ok`, which is the
    exact opposite of what happened.
    """


def _fetch(package):
    """pub.dev metadata for `package`, or None when it is genuinely unpublished.

    A 404 is an answer and is cached as one. Anything else -- DNS, timeout, 5xx,
    a body that will not parse -- raises, and the failure itself is cached so a
    single outage is not re-tried once per dependency.
    """
    if package in _cache:
        cached = _cache[package]
        if isinstance(cached, FetchError):
            raise cached
        return cached
    try:
        with urllib.request.urlopen(
                f"https://pub.dev/api/packages/{package}", timeout=15) as r:
            _cache[package] = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            _cache[package] = None
        else:
            _cache[package] = FetchError(
                f"pub.dev answered HTTP {e.code} for {package}")
    except Exception as e:
        _cache[package] = FetchError(f"pub.dev unreachable for {package}: {e}")
    cached = _cache[package]
    if isinstance(cached, FetchError):
        raise cached
    return cached


def _pub(package):
    """Latest published version + its pubspec, or None if unpublished."""
    data = _fetch(package)
    if data is None:
        return None
    return (data["latest"]["version"], data["latest"]["pubspec"])


def _versions(package):
    """{version: pubspec} for every published release, {} if unpublished."""
    data = _fetch(package)
    if data is None:
        return {}
    return {v["version"]: v["pubspec"] for v in data["versions"]}


def _local_version(root, pkg):
    p = os.path.join(root, pkg, "pubspec.yaml")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        m = re.search(r'^version:\s*(\S+)', f.read(), re.M)
    return m.group(1) if m else None


def _deps_by_section(root, pkg):
    """{section: {name: constraint}} for hosted deps -- path/sdk/git are skipped.

    Sections stay separate because folding them together is not lossless: the
    merged dict let dev_dependencies overwrite the real dependency of the same
    name, so a package depending on X and dev-depending on a different range of
    X reported the dev range as its constraint. That fabricates a difference
    against what was published, and hides a real one just as easily.
    """
    p = os.path.join(root, pkg, "pubspec.yaml")
    out = {"dependencies": {}, "dev_dependencies": {}}
    section = None
    with open(p, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines:
        if re.match(r'^[a-zA-Z_]+:', line):
            section = line.split(":")[0]
            continue
        if section not in out:
            continue
        m = re.match(r'^  ([a-z0-9_]+):\s*(\S.*)?$', line)
        if not m:
            continue
        name = m.group(1)
        # Strip a trailing YAML comment. `foo: ^1.2.3  # keep in sync` used to
        # be stored with the comment attached, which no constraint parser can
        # read, and `foo:  # block below` stored the comment AS the constraint.
        constraint = re.sub(r'(?:^|\s)#.*$', '', (m.group(2) or "")).strip()
        # Unquote. YAML needs quotes around a constraint starting with `>`, so
        # `'>=0.3.9 <0.5.0'` is the same constraint as `>=0.3.9 <0.5.0` -- but
        # this reads the file as text, and pub.dev serves it parsed, so the
        # quotes alone read as a changed dependency.
        if len(constraint) > 1 and constraint[0] == constraint[-1] and \
                constraint[0] in "\"'":
            constraint = constraint[1:-1].strip()
        if not constraint:  # a block form: path:/sdk:/git: on the next line
            continue
        out[section][name] = constraint
    return out


def _deps(root, pkg, section="all"):
    """{name: constraint} for hosted deps in `section`.

    `section` is "dependencies", "dev_dependencies", or "all" -- and "all"
    resolves a name present in both to the real dependency, never the dev one.
    """
    by_section = _deps_by_section(root, pkg)
    if section == "all":
        merged = dict(by_section["dev_dependencies"])
        merged.update(by_section["dependencies"])
        return merged
    return dict(by_section[section])


def _parse_version(text):
    """(major, minor, patch, prerelease) or None. Build metadata is ignored."""
    m = re.fullmatch(
        r'(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:-([0-9A-Za-z.-]+))?'
        r'(?:\+[0-9A-Za-z.-]+)?', text.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0),
            m.group(4) or "")


def _key(v):
    """Comparable key for a parsed version, with semver prerelease ordering."""
    if not v[3]:
        return (v[0], v[1], v[2], (1,), ())
    pre = tuple((0, int(p), "") if p.isdigit() else (1, 0, p)
                for p in v[3].split("."))
    return (v[0], v[1], v[2], (0,), pre)


def _caret_upper(lo):
    """Pub's caret bound: the next bump of the leading non-zero component."""
    if lo[0] > 0:
        return (lo[0] + 1, 0, 0, "")
    if lo[1] > 0:
        return (0, lo[1] + 1, 0, "")
    return (0, 0, lo[2] + 1, "")


_TERM = re.compile(r'(>=|<=|>|<|=|\^)?\s*([0-9][0-9A-Za-z.+-]*)')


def _satisfies(constraint, version):
    """Does `constraint` admit `version`? True / False / None ("not evaluated").

    Handles `any`, an exact version, a caret range, and a whitespace-separated
    conjunction of `>= > <= < =` terms -- between them, every form these
    pubspecs actually use. Anything else still returns None.

    None used to be the quiet answer: both callers tested `is False`, so
    `>=1.0.0 <2.0.0`, `any`, `^1.0` and any caret with a trailing comment were
    reported as fine while never having been evaluated at all. A None is now an
    ERROR at the call site -- unevaluated is not the same as correct.
    """
    c = constraint.strip().strip('"').strip("'").strip()
    got = _parse_version(version)
    if got is None:
        return None
    if not c or c == "any":
        return True

    terms, pos = [], 0
    for m in _TERM.finditer(c):
        if c[pos:m.start()].strip():
            return None  # something between terms that is not an operator
        lo = _parse_version(m.group(2))
        if lo is None:
            return None
        terms.append((m.group(1) or "=", lo))
        pos = m.end()
    if c[pos:].strip() or not terms:
        return None

    ok = True
    for op, lo in terms:
        if op == "^":
            if len(terms) > 1:
                return None  # a caret does not compose with other terms
            ok = ok and _key(lo) <= _key(got) < _key(_caret_upper(lo))
        elif op == "=":
            ok = ok and _key(got) == _key(lo)
        elif op == ">=":
            ok = ok and _key(got) >= _key(lo)
        elif op == ">":
            ok = ok and _key(got) > _key(lo)
        elif op == "<=":
            ok = ok and _key(got) <= _key(lo)
        elif op == "<":
            ok = ok and _key(got) < _key(lo)
    return ok


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
            verdict = _satisfies(constraint, local[dep])
            if not constraint.startswith("^"):
                errs.append(f"{pkg}: pins {dep} as '{constraint}' -- use a caret "
                            f"range (^{local[dep]}); exact pins cascade")
            elif verdict is False:
                errs.append(f"{pkg}: pins {dep} '{constraint}' but {dep} is at "
                            f"{local[dep]} on disk -- run `release.py sync`")
            elif verdict is None:
                errs.append(f"{pkg}: pins {dep} '{constraint}', a constraint "
                            f"form this script cannot evaluate -- rewrite it as "
                            f"a caret range (^{local[dep]}); unevaluated is not "
                            f"the same as correct")
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
            try:
                info = _pub(dep)
            except FetchError as e:
                errs.append(f"{pkg}: its {dep} constraint '{constraint}' could "
                            f"not be checked -- {e}")
                continue
            if not info:
                continue
            latest, _ = info
            verdict = _satisfies(constraint, latest)
            if verdict is False:
                errs.append(f"{pkg}: constrains {dep} to '{constraint}' but "
                            f"{dep} {latest} is published -- a consumer of both "
                            f"cannot resolve")
            elif verdict is None:
                errs.append(f"{pkg}: constrains {dep} to '{constraint}', a "
                            f"constraint form this script cannot evaluate, so it "
                            f"was NOT checked against the published {dep} "
                            f"{latest} -- rewrite it as a caret range")
    return errs


def check_version_bumped(root, packages):
    """A published version number must never describe two different pubspecs.

    minigpu_view's repo copy of 1.5.9 had newer pins than the 1.5.9 on pub.dev:
    someone fixed the constraints and never bumped, so the fix could not be
    published and nobody could tell from the version alone.

    Added and removed dependencies count as differences. The comparison used to
    require the PUBLISHED side to be a string, which made a locally added
    dependency invisible -- miniav_ffi adding its three build-hook deps under
    the already published 0.7.0 passed on that alone. Only dependencies are
    compared: dev_dependencies do not reach a consumer.
    """
    errs = []
    for pkg in packages:
        v = _local_version(root, pkg)
        if not v:
            continue
        try:
            published = _versions(pkg)
        except FetchError as e:
            errs.append(f"{pkg}: could not check whether {v} is already "
                        f"published -- {e}")
            continue
        if v not in published:
            continue  # unpublished version -- exactly right for a release
        want = published[v].get("dependencies", {}) or {}
        have = _deps(root, pkg, "dependencies")
        diff = []
        for name in sorted(set(want) | set(have)):
            was, now = want.get(name), have.get(name)
            if not isinstance(was, str) and name not in have:
                # sdk:/path:/git: descriptors are block form in both places and
                # _deps cannot see them; absence here is not a difference.
                continue
            if was == now:
                continue
            if name not in want:
                diff.append(f"added {name} {now}")
            elif name not in have:
                diff.append(f"removed {name}")
            else:
                diff.append(f"{name} {was} -> {now}")
        if diff:
            errs.append(f"{pkg}: {v} is already published with DIFFERENT "
                        f"dependencies ({'; '.join(diff[:4])}) -- bump "
                        f"the version, the fix cannot ship under {v}")
    return errs


# pub says "found the following error:" for one and "found the following 3
# errors:" for several -- reading only the counted form let a single error look
# like a clean run. The refusal line is a second, independent signal.
_PUBLISH_ERROR = re.compile(r'found the following (?:(\d+) )?error')
_PUBLISH_REFUSED = "missing a requirement and can't be published"
# The line that proves validation ran to completion with nothing worse than
# warnings. Its presence is what separates "tolerated" from "no idea".
_PUBLISH_SUMMARY = re.compile(r'^Package has \d+ warning', re.M)


def check_publish_validation(root, packages):
    """`pub publish --dry-run`, failing on ERRORS only.

    Warnings are tolerated on purpose: split plugins legitimately produce them
    (an endorsed implementation, a platform-interface-only package), and
    refusing on warnings is why publishing runs with validation skipped in the
    first place. Errors are a different animal -- miniav_tools_codecs 0.6.1
    shipped past four of them and was unusable as a dependency, because its
    build hook imported packages it never declared, and a consumer does not get
    dev_dependencies. Same defect is live in minigpu_ffi.

    The exit code cannot be the rule on its own: pub exits 65 for warnings just
    as it does for errors, so failing on nonzero would refuse every split plugin
    in the family. It cannot be ignored either -- a dry-run that dies in `pub
    get` prints no summary at all and used to be read as a pass. So: an error
    summary fails, a warning summary passes, and a nonzero exit with NEITHER
    fails, because nothing was actually validated.
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
        m = _PUBLISH_ERROR.search(out)
        if m or _PUBLISH_REFUSED in out:
            count = int(m.group(1)) if (m and m.group(1)) else 1
            detail = [l.strip() for l in out.split("\n")
                      if l.startswith("* ")][:count]
            errs.append(f"{pkg}: {count} publish validation ERROR(s):\n      " +
                        "\n      ".join(detail))
            continue
        if r.returncode != 0 and not _PUBLISH_SUMMARY.search(out):
            tail = [l.rstrip() for l in out.strip().split("\n") if l.strip()][-10:]
            errs.append(f"{pkg}: `dart pub publish --dry-run` exited "
                        f"{r.returncode} without validating anything this "
                        f"script can read -- last output:\n      " +
                        "\n      ".join(tail))
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
            # A check that crashed has not passed. This printed SKIPPED and left
            # `failed` alone, so a broken check was indistinguishable from a
            # clean one in the exit code.
            print(f"  ERROR  the check itself failed: {e}")
            failed = True
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
