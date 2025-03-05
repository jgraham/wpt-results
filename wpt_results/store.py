import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import urllib
from datetime import datetime, timedelta, date as Date
from typing import (
    Any,
    BinaryIO,
    Callable,
    ContextManager,
    Generator,
    IO,
    Iterable,
    Iterator,
    Mapping,
    NotRequired,
    Optional,
    Sequence,
    TypedDict,
    cast,
)

import httpx
import pygit2
import tcfetch
import zstandard as zstd
from pygit2.enums import FileMode


GIT_AUTHOR = pygit2.Signature("Results Ingestion", "wptresults@mozilla.bugs")

STATUS_ORDER = [
    "PASS",
    "OK",
    "FAIL",
    "PRECONDITION_FAILED",
    "ERROR",
    "TIMEOUT",
    "CRASH",
    "ASSERT",
    "SKIP",
]


class SubtestResultDict(TypedDict):
    name: str
    status: str
    statuses: NotRequired[list[str]]


class ResultDict(TypedDict):
    status: str
    statuses: NotRequired[list[str]]
    subtests: list[SubtestResultDict]


class TestResult:
    def __init__(self) -> None:
        self.statuses: set[str] = set()
        self.subtests: dict[str, set[str]] = {}

    def add_subtest(self, name: str, status: str) -> None:
        if name not in self.subtests:
            self.subtests[name] = set()
        self.subtests[name].add(status)

    def get_single_status(self, statuses: set[str]) -> str:
        if len(statuses) == 1:
            return list(statuses)[0]

        # If we have more than one status, get the most positive one.
        # This isn't ideal, but updating all the interop code to handle
        # more than one status is challenging
        for candidate in STATUS_ORDER:
            if candidate in statuses:
                return candidate

        raise ValueError(f"Didn't find valid status, got {', '.join(self.statuses)}")

    def to_dict(self) -> ResultDict:
        subtests: list[SubtestResultDict] = []
        rv: ResultDict = {
            "status": self.get_single_status(self.statuses),
            "subtests": subtests,
        }
        if len(self.statuses) > 1:
            rv["statuses"] = sorted(self.statuses)
        for name, statuses in self.subtests.items():
            subtest: SubtestResultDict = {
                "name": name,
                "status": self.get_single_status(statuses),
            }
            assert subtest["status"] != "OK"
            if len(statuses) > 1:
                subtest["statuses"] = sorted(statuses)
            rv["subtests"].append(subtest)
        return rv


class RunResults:
    def __init__(self) -> None:
        self._data: dict[str, TestResult] = {}

    def __len__(self) -> int:
        return len(self._data)

    def add_test(
        self, test_name: str, status: str, subtests: Iterable[tuple[str, str]]
    ) -> None:
        if test_name not in self._data:
            self._data[test_name] = TestResult()
        self._data[test_name].statuses.add(status)
        for name, result in subtests:
            self._data[test_name].add_subtest(name, result)

    def __iter__(self) -> Iterator[tuple[str, TestResult]]:
        yield from self._data.items()


NameMap = Mapping[str, Mapping[str, pygit2.Oid] | pygit2.Oid]


def create_tree_map(path_oid_map: Mapping[str, pygit2.Oid]) -> NameMap:
    tree_map: dict[str, dict[str, Any] | pygit2.Oid] = {}
    for path, oid in path_oid_map.items():
        if path.startswith("/"):
            path = path[1:]
        parts = path.split("/", 1)
        name = parts[0]

        if len(parts) > 1:
            if name not in tree_map:
                tree_map[name] = {}
            target = tree_map[name]
            assert isinstance(target, dict)
            target[parts[1]] = oid
        else:
            assert name not in tree_map
            tree_map[name] = oid
    return tree_map


def insert_blobs(
    repo: pygit2.Repository,
    src_tree: Optional[pygit2.Tree],
    path_oid_map: Mapping[str, pygit2.Oid],
) -> tuple[pygit2.Oid, int]:
    insert_count = 0
    tree_builder = repo.TreeBuilder()

    tree_map = create_tree_map(path_oid_map)

    # Copy over unchanged entries in the existing tree
    if src_tree is not None:
        for obj in src_tree:
            src_name = obj.name
            if src_name is not None and src_name not in tree_map:
                tree_builder.insert(src_name, obj.id, obj.filemode)

    # Add new entries
    for name, entry in tree_map.items():
        current_obj = (
            src_tree[name] if src_tree is not None and name in src_tree else None
        )

        if current_obj is None or current_obj.type_str == "blob":
            new_src_tree = None
        else:
            assert isinstance(current_obj, pygit2.Tree)
            new_src_tree = current_obj

        if isinstance(entry, dict):
            tree, count = insert_blobs(repo, new_src_tree, entry)
            tree_builder.insert(name, tree, FileMode.TREE)
            insert_count += count
        else:
            assert isinstance(entry, pygit2.Oid)
            tree_builder.insert(name, entry, FileMode.BLOB)
            insert_count += 1

    return (tree_builder.write(), insert_count)


def get_path(src_tree: pygit2.Tree, path: str) -> Optional[pygit2.Object]:
    path_parts = path.split("/")
    target = src_tree
    for part in path_parts:
        if not isinstance(target, pygit2.Tree) or part not in target:
            return None
        target = cast(pygit2.Tree, target[part])
    return target


class ResultsRepo:
    index_ref = "refs/runs/index"

    def __init__(self, path: str):
        self.path = path
        self.repo = pygit2.Repository(self.path)

    @staticmethod
    def test_path(test_id: str) -> str:
        output = []
        parts = test_id.split("/")
        for i, part in enumerate(parts):
            last = False
            if "?" in part or "?" in part:
                # we've reched a part of the path with a query string, so the
                # entire rest of the path is the test name
                part = "/".join(parts[i:])
                last = True

            output.append(urllib.parse.quote(part, safe="!~*'()"))
            if last:
                break
        return "/".join(output) + ".json"

    @staticmethod
    def index_path(branch: str, date: Date | datetime, commit: str) -> str:
        return f"runs/{branch}/{date.strftime('%Y-%m-%d')}/{commit}.json"

    def get_stored_commits(self, branch: str) -> Mapping[str, list[str]]:
        refs = self.repo.references
        data: dict[str, list[str]] = {}
        if self.index_ref in refs:
            index_commit = refs[self.index_ref].peel()
            dates = get_path(index_commit.tree, f"runs/{branch}")
            if dates is None:
                return data
            if not isinstance(dates, pygit2.Tree):
                raise ValueError(
                    f"Expected {self.index_ref}:runs/{branch} to be a tree"
                )
            for date_obj in dates:
                assert date_obj.name is not None
                if not isinstance(date_obj, pygit2.Tree):
                    raise ValueError(
                        f"Expected {self.index_ref}:runs/{branch}/{date_obj.name} to be a tree"
                    )
                data[date_obj.name] = []
                commits = get_path(date_obj, "revision")
                if not isinstance(commits, pygit2.Tree):
                    raise ValueError(
                        f"Expected {self.index_ref}:runs/{branch}/{date_obj.name}/revision to be a tree"
                    )
                for commit in commits:
                    assert commit.name is not None
                    if not isinstance(commit, pygit2.Blob):
                        raise ValueError(
                            f"Expected {self.index_ref}:runs/{branch}/{date_obj.name}/revision/{commit.name} to be a blob"
                        )
                    if commit.name.endswith(".json"):
                        commit_sha = commit.name[:-5]
                        data[date_obj.name].append(commit_sha)
        return data

    def get_index(
        self, branch: str, date: Date | datetime, commit: str
    ) -> Optional[dict[str, Any]]:
        refs = self.repo.references
        if self.index_ref in refs:
            index_commit = refs[self.index_ref].peel()
            path = self.index_path(branch, date, commit)
            index_blob = get_path(index_commit.tree, path)
            if index_blob is None:
                return None
            if not isinstance(index_blob, pygit2.Blob):
                raise ValueError(f"Index path {path} expected a blob")
            return json.loads(index_blob.data)
        return None

    def has_commit(self, branch: str, date: Date | datetime, commit: str) -> bool:
        return self.get_index(branch, date, commit) is not None

    def get_run_metadata(
        self,
        branch: str,
        date: datetime,
        commit: str,
        run_info_filter: Optional[Mapping[str, Any]],
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        index = self.get_index(branch, date, commit)
        if index is None:
            return None

        filter_set = (
            set(run_info_filter.items()) if run_info_filter is not None else None
        )

        for name, data in index.items():
            include = filter_set is None or filter_set.issubset(
                set(data["run_info"].items())
            )
            if include:
                yield name, data

    def add_run(
        self,
        branch: str,
        revision: str,
        date: datetime,
        run_name: str,
        wpt_report_paths: list[str],
    ) -> None:
        logging.info(f"Adding {revision} {run_name}")
        results_data = self._read_results(wpt_report_paths)
        if results_data is None:
            logging.warning(f"No results data found for {revision} {run_name}")
            return
        run_info, results = results_data
        run_id = hashlib.sha1(f"{revision} {run_name}".encode("utf8")).hexdigest()

        self._write_results(revision, run_name, run_id, results)
        self._write_index(branch, revision, date, run_name, run_id, run_info)

    def _read_results(
        self, wpt_report_paths: list[str], run_info: Optional[dict[str, Any]] = None
    ) -> Optional[tuple[dict[str, Any], RunResults]]:
        results = RunResults()
        for path in wpt_report_paths:
            if os.path.splitext(path) == ".zstd":
                open_fn = cast(Callable[[str, str], ContextManager[BinaryIO]], zstd.open)
            else:
                open_fn = cast(Callable[[str, str], ContextManager[BinaryIO]], open)
            with open_fn(path, "rb") as f:
                try:
                    data = json.load(f)
                except Exception:
                    logging.warning(f"Error reading results data from {path}")
                    continue
            report_run_info = data["run_info"]
            if run_info is None:
                run_info = report_run_info
            elif report_run_info != run_info:
                difference = set(report_run_info.items()) ^ set(run_info.items())
                logging.warning(f"run_info differed, {difference}")
                # raise ValueError("Reports had incompatible run_info")
            for test_result in data["results"]:
                subtests = [
                    (item["name"], item["status"]) for item in test_result["subtests"]
                ]
                results.add_test(test_result["test"], test_result["status"], subtests)
        if run_info is None:
            return None
        return run_info, results

    def _write_results(
        self, revision: str, run_name: str, run_id: str, results: RunResults
    ) -> None:
        # ref: runs/{run_id}/results /path/to/test
        results_ref = f"refs/runs/{run_id}/results"
        refs = self.repo.references
        if results_ref in refs:
            current_commit = refs[results_ref].peel()
            parents = [current_commit.id]
        else:
            parents = []

        blobs = {}
        for test_id, result in results:
            blob_data = json.dumps(result.to_dict(), indent=1).encode()
            blobs[self.test_path(test_id)] = self.repo.create_blob(blob_data)
        tree_oid, count = insert_blobs(self.repo, None, blobs)
        commit = self.repo.create_commit(
            results_ref,
            GIT_AUTHOR,
            GIT_AUTHOR,
            f"Add results for {revision} {run_name}",
            tree_oid,
            parents,
        )

    def _write_index(
        self,
        branch: str,
        revision: str,
        date: datetime,
        run_name: str,
        run_id: str,
        run_info: dict[str, Any],
    ) -> None:
        # ref: runs/index /runs/{branch}/{date}/revision/{sha1}.json
        refs = self.repo.references
        push_date = f"{date.year}-{date.month:02}-{date.day:02}"
        path = f"runs/{branch}/{push_date}/revision/{revision}.json"
        parents = []
        index_tree = None
        data = None
        if self.index_ref in refs:
            index_commit = refs[self.index_ref].peel()
            index_tree = index_commit.tree
            parents = [index_commit.id]
            current_index = get_path(index_commit.tree, path)
            if current_index is not None:
                if not isinstance(current_index, pygit2.Blob):
                    raise ValueError(f"Path {path} is not a blob")
                data = json.loads(current_index.data)
                if "push_date" not in data or "runs" not in data:
                    data = None

        if data is None:
            data = {"push_date": date.isoformat(), "runs": {}}

        data["runs"][run_name] = {"id": run_id, "run_info": run_info}

        blobs = {path: self.repo.create_blob(json.dumps(data, indent=1).encode())}
        tree_oid, count = insert_blobs(self.repo, index_tree, blobs)

        self.repo.create_commit(
            self.index_ref,
            GIT_AUTHOR,
            GIT_AUTHOR,
            f"Add index for {revision} {run_name}",
            tree_oid,
            parents,
        )


def get_pushlog(
    branch: str, commit: Optional[str] = None, date: Optional[Date] = None
) -> dict[str, Any]:
    path = BRANCH_TO_PATH[branch]
    url = f"https://hg.mozilla.org/{path}/json-pushes?version=2&tipsonly=1"
    if commit is not None:
        url += f"&changeset={commit}"
    if date is not None:
        start_date = date.strftime("%Y-%m-%d")
        end_date = (date + timedelta(days=1)).strftime("%Y-%m-%d")
        url += f"&startdate={start_date}&enddate={end_date}"
    resp = httpx.get(url)
    resp.raise_for_status()

    data = resp.json()
    if "error" in data:
        raise ValueError(data["error"])

    return data


test_types = ["reftest", "wdspec", "crashtest", "print-reftest"]
separate_jobs = ["backlog", "canvas", "eme", "webcodecs", "webgpu-long", "webgpu"]


def get_task_runs(
    download_data: Sequence[tcfetch.TaskDownloadData],
) -> Mapping[str, list[str]]:
    name_re = re.compile(r"(.*)-\d+$")
    by_run_id: dict[str, tuple[list[str], list[str]]] = {}
    for artifact in download_data:
        name = artifact.name
        run_id = artifact.run_id
        if run_id is None:
            raise ValueError(
                "Got run with no id (from extra.test-settings._hash in task definition"
            )

        m = name_re.match(name)
        if m is not None:
            group_name = m.group(1)
        else:
            group_name = name
        if run_id not in by_run_id:
            by_run_id[run_id] = [], []
        by_run_id[run_id][0].append(group_name)
        by_run_id[run_id][1].append(artifact.path)
    rv: dict[str, list[str]] = {}
    for names, paths in by_run_id.values():
        name = min(names, key=lambda x: len(x))
        rv[name] = paths
    logging.info(
        f"Found {len(download_data)} tasks and {len(rv)} different runs for commit"
    )
    return rv


BRANCH_TO_PATH = {
    "try": "try",
    "mozilla-release": "releases/mozilla-release",
    "mozilla-beta": "releases/mozilla-beta",
    "mozilla-central": "mozilla-central",
    "mozilla-inbound": "integration/mozilla-inbound",
    "autoland": "integration/autoland",
}


def get_latest_commit(branch: str) -> Optional[tuple[str, dict[str, Any]]]:
    push_data = get_pushlog(branch, None)["pushes"]
    pushes = sorted([item for item in push_data.keys()], key=lambda x: -int(x))
    for push in pushes:
        commit = push_data[push]["changesets"][-1]
        if tcfetch.check_complete(branch, commit):
            return commit, push_data[push]
        else:
            logging.info(f"{commit} is not complete")
    return None


@contextlib.contextmanager
def PathWrapper(path: str) -> Generator[str, None, None]:
    if not os.path.exists(path):
        os.makedirs(path)
    yield path


def add_commit(
    results_repo: ResultsRepo,
    branch: str,
    commit: str,
    push_data: Optional[Mapping[str, Any]],
    log_path: Optional[str],
) -> None:
    if push_data is None:
        data = get_pushlog(branch, commit)
        push_data = data["pushes"].values()[0]

    logging.info(f"Updating for commit {branch} {commit}")
    if commit and push_data["changesets"][0] != commit:
        raise ValueError(
            f"Expected {commit} as push head, got {push_data['changesets'][0]}"
        )

    date = datetime.fromtimestamp(push_data["date"])

    if log_path is None:
        download_context = cast(ContextManager[str], tempfile.TemporaryDirectory())
    else:
        download_context = PathWrapper(log_path)

    with download_context as path:
        fetch_results = tcfetch.download_artifacts(
            branch, commit, "wptreport.json", out_dir=path
        )
        task_runs = get_task_runs(fetch_results)
        for run_name, download_paths in task_runs.items():
            results_repo.add_run(branch, commit, date, run_name, download_paths)


def get_last_stored_date(results_repo: ResultsRepo, branch: str) -> datetime:
    return max(
        datetime.strptime(date, "%Y-%m-%d")
        for date in results_repo.get_stored_commits(branch).keys()
    )


def list_stored(results_repo: ResultsRepo, branch: str) -> None:
    for date, commits in results_repo.get_stored_commits(branch).items():
        print(date)
        for commit in commits:
            print(f"  {commit}")
    return


def get_backfill_commits(
    results_repo: ResultsRepo, backfill: str, branch: str
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if backfill == "new":
        start_date = get_last_stored_date(results_repo, branch)
    else:
        try:
            start_date = datetime.strptime(backfill, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                "--backfill argument must be in format YYYY-MM-DD or 'new'"
            )

    date = start_date.date()
    today = datetime.now().date()
    if date > today:
        raise ValueError("--backfill argument must be a date in the past")
    if today - date > timedelta(days=100):
        raise ValueError("can't backfill more than 100 days of data")

    commits = []
    while date <= today:
        logging.info(f"Getting commits for {date}")
        pushlog = get_pushlog(branch, date=date)
        for push, push_data in pushlog["pushes"].items():
            commit = push_data["changesets"][-1]
            if date < today - timedelta(days=1) or tcfetch.check_complete(
                branch, commit
            ):
                if not results_repo.has_commit(branch, date, commit):
                    logging.info(f"Got commit {commit}")
                    commits.append((commit, push_data))
                else:
                    logging.debug(f"Commit {commit} already has data stored")
            else:
                logging.info(f"{commit} is not yet complete")
        date += timedelta(days=1)
    return commits


def update(results_repo: ResultsRepo, args: argparse.Namespace) -> None:
    if args.backfill is not None:
        commits = get_backfill_commits(results_repo, args.backfill, args.branch)
    elif args.commit is None:
        latest_commit = get_latest_commit(args.branch)
        if latest_commit is None:
            raise ValueError("Failed to get latest commit and none supplied")
        commits = [latest_commit]
    else:
        commit = args.commit
        push_data = None

    for commit, push_data in commits:
        add_commit(results_repo, args.branch, commit, push_data, args.log_path)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "warn", "info", "debug"],
        help="Logging level",
    )
    parser.add_argument(
        "--git-repo", default="./data", help="Path to the output git repository"
    )
    parser.add_argument(
        "--branch",
        default="mozilla-central",
        choices=list(BRANCH_TO_PATH.keys()),
        help="Gecko branch to read from",
    )
    parser.add_argument("--commit", help="Commit to import")
    parser.add_argument("--log-path", default="None", help="Path to store logs")
    parser.add_argument("--list", action="store_true", help="List stored commits")
    parser.add_argument(
        "--backfill",
        action="store",
        help="Backfill from a given date or 'new' to get runs since the last recorded one",
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.getLevelNamesMapping()[args.log_level.upper()])
    logging.getLogger("wpt_interop").setLevel(logging.INFO)

    results_repo = ResultsRepo(args.git_repo)

    if args.branch not in BRANCH_TO_PATH:
        raise ValueError(f"Unknown branch {args.branch}")

    if args.list:
        list_stored(results_repo, args.branch)
    else:
        try:
            update(results_repo, args)
        except Exception:
            import traceback
            import pdb

            traceback.print_exc()
            pdb.post_mortem()
