//! The memory a process may use (#532), the base of the default memory limit: on Linux the
//! smaller of `MemTotal` and every cgroup limit on the path from the process's cgroup to the
//! root (v2 `memory.max`, v1 `memory.limit_in_bytes`; the smallest wins, so a limit set on a
//! parent slice or pod counts). `max`, v1's "unlimited" value and a missing file are no limit.
//! On a hybrid system v2 is read only when the root's `cgroup.controllers` lists `memory`.
//! When nothing can be read the fallback is 1 GiB, never unlimited, and the result says so, so
//! the open can warn. The default budget is `DEFAULT_MEMORY_FRACTION` (0.75) of it.
//!
//! The detection reads through pure functions over text and a cgroup filesystem root, so the
//! fixtures run on every platform.

use std::path::Path;

use super::*;

const MIB: usize = 1024 * 1024;

/// A fresh directory standing for a cgroup filesystem's root, removed with the guard.
struct Root(std::path::PathBuf);

impl Root {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "grafeo-cgroups-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Root {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
/// What v1 writes for "no limit" (the largest page-aligned i64).
const V1_UNLIMITED: &str = "9223372036854771712";

/// A cgroup filesystem under `root`, with `file` holding `value` in each `(directory, value)`.
fn cgroups(root: &Path, file: &str, levels: &[(&str, &str)]) {
    for (directory, value) in levels {
        let dir = root.join(directory.trim_start_matches('/'));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(file), format!("{value}\n")).unwrap();
    }
}

const V2_SELF: &str = "0::/user.slice/app.scope\n";
const MEMINFO: &str = "MemTotal:       16777216 kB\nMemFree:         1048576 kB\n";

#[test]
fn memtotal_is_read_in_kibibytes() {
    assert_eq!(meminfo_total(MEMINFO), Some(16_777_216 * 1024));
    assert_eq!(meminfo_total("MemFree: 12 kB\n"), None, "no MemTotal line");
    assert_eq!(meminfo_total("MemTotal: lots kB\n"), None);
}

#[test]
fn a_v2_limit_on_the_own_cgroup_of_the_process() {
    let root = Root::new();
    cgroups(
        root.path(),
        "memory.max",
        &[
            ("/", "max"),
            ("/user.slice", "max"),
            ("/user.slice/app.scope", "536870912"),
        ],
    );
    assert_eq!(cgroup_limit(root.path(), V2_SELF), Some(512 * MIB));
}

/// A limit set on a parent only (a pod or a slice) binds the process below it.
#[test]
fn a_v2_limit_on_a_parent_only_counts() {
    let root = Root::new();
    cgroups(
        root.path(),
        "memory.max",
        &[
            ("/user.slice", "1073741824"),
            ("/user.slice/app.scope", "max"),
        ],
    );
    assert_eq!(cgroup_limit(root.path(), V2_SELF), Some(1024 * MIB));
}

#[test]
fn the_smallest_limit_on_the_path_wins() {
    let root = Root::new();
    cgroups(
        root.path(),
        "memory.max",
        &[
            ("/user.slice", "268435456"),
            ("/user.slice/app.scope", "1073741824"),
        ],
    );
    assert_eq!(cgroup_limit(root.path(), V2_SELF), Some(256 * MIB));
    let root = Root::new();
    cgroups(
        root.path(),
        "memory.max",
        &[
            ("/user.slice", "1073741824"),
            ("/user.slice/app.scope", "268435456"),
        ],
    );
    assert_eq!(cgroup_limit(root.path(), V2_SELF), Some(256 * MIB));
}

#[test]
fn v2_max_everywhere_and_missing_files_are_no_limit() {
    let root = Root::new();
    cgroups(
        root.path(),
        "memory.max",
        &[("/user.slice", "max"), ("/user.slice/app.scope", "max")],
    );
    assert_eq!(cgroup_limit(root.path(), V2_SELF), None);
    let empty = Root::new();
    assert_eq!(
        cgroup_limit(empty.path(), V2_SELF),
        None,
        "no cgroup files at all"
    );
    assert_eq!(
        cgroup_limit(empty.path(), ""),
        None,
        "no /proc/self/cgroup text"
    );
}

/// cgroup v1: the memory controller's line names the cgroup, under its own mount.
#[test]
fn a_v1_limit_and_v1_unlimited() {
    let v1_self = "12:pids:/docker/abc\n4:memory:/docker/abc\n1:name=systemd:/docker/abc\n";
    let root = Root::new();
    cgroups(
        root.path(),
        "memory.limit_in_bytes",
        &[
            ("/memory", V1_UNLIMITED),
            ("/memory/docker/abc", "268435456"),
        ],
    );
    assert_eq!(cgroup_limit(root.path(), v1_self), Some(256 * MIB));

    let root = Root::new();
    cgroups(
        root.path(),
        "memory.limit_in_bytes",
        &[
            ("/memory", V1_UNLIMITED),
            ("/memory/docker", "536870912"),
            ("/memory/docker/abc", V1_UNLIMITED),
        ],
    );
    assert_eq!(
        cgroup_limit(root.path(), v1_self),
        Some(512 * MIB),
        "unlimited is no limit; the parent's limit binds"
    );
}

/// The memory the process may use is the smaller of the machine's and the cgroups'.
#[test]
fn the_process_may_use_the_smaller_of_memtotal_and_the_cgroups() {
    let root = Root::new();
    cgroups(
        root.path(),
        "memory.max",
        &[("/user.slice/app.scope", "536870912")],
    );
    let found = linux_memory(Some(MEMINFO), root.path(), Some(V2_SELF));
    assert_eq!(
        (found.bytes, found.source),
        (512 * MIB, MemorySource::Limit)
    );

    let root = Root::new();
    cgroups(
        root.path(),
        "memory.max",
        &[("/user.slice/app.scope", "max")],
    );
    let found = linux_memory(Some(MEMINFO), root.path(), Some(V2_SELF));
    assert_eq!(
        (found.bytes, found.source),
        (16_777_216 * 1024, MemorySource::Machine)
    );

    // A cgroup limit above the machine's memory: the machine binds.
    let root = Root::new();
    cgroups(
        root.path(),
        "memory.max",
        &[("/user.slice/app.scope", &(64 * 1024 * MIB).to_string())],
    );
    let found = linux_memory(Some(MEMINFO), root.path(), Some(V2_SELF));
    assert_eq!(
        (found.bytes, found.source),
        (16_777_216 * 1024, MemorySource::Machine)
    );
}

/// Nothing readable: 1 GiB, never unlimited, and the source says so for the open's warning.
#[test]
fn the_fallback_is_one_gib_and_says_so() {
    let empty = Root::new();
    let found = linux_memory(None, empty.path(), None);
    assert_eq!(
        (found.bytes, found.source),
        (FALLBACK_SYSTEM_MEMORY, MemorySource::Fallback)
    );
    assert_eq!(FALLBACK_SYSTEM_MEMORY, 1024 * MIB);
    // A cgroup limit alone, with no meminfo, still binds.
    let root = Root::new();
    cgroups(
        root.path(),
        "memory.max",
        &[("/user.slice/app.scope", "268435456")],
    );
    let found = linux_memory(None, root.path(), Some(V2_SELF));
    assert_eq!(
        (found.bytes, found.source),
        (256 * MIB, MemorySource::Limit)
    );
}

#[test]
fn the_default_budget_is_three_quarters() {
    assert!((DEFAULT_MEMORY_FRACTION - 0.75).abs() < f64::EPSILON);
    let fallback = DetectedMemory {
        bytes: FALLBACK_SYSTEM_MEMORY,
        source: MemorySource::Fallback,
    };
    assert_eq!(
        fallback.default_budget(),
        768 * MIB,
        "the fallback's budget"
    );
    let machine = DetectedMemory {
        bytes: 16 * 1024 * MIB,
        source: MemorySource::Machine,
    };
    assert_eq!(machine.default_budget(), 12 * 1024 * MIB);
}

/// A hybrid system lists a v2 line and a v1 memory line. v2 holds the memory controller only
/// when the root's `cgroup.controllers` names `memory`; otherwise the v1 hierarchy is read.
/// The two limits differ, so each row shows which was read.
#[test]
fn a_hybrid_system_reads_the_hierarchy_that_holds_memory() {
    let hybrid = "0::/user.slice/app.scope
4:memory:/user.slice/app.scope
";
    let tree = |controllers: Option<&str>| {
        let root = Root::new();
        cgroups(
            root.path(),
            "memory.max",
            &[("/user.slice/app.scope", "536870912")],
        );
        cgroups(
            root.path(),
            "memory.limit_in_bytes",
            &[("/memory/user.slice/app.scope", "268435456")],
        );
        if let Some(controllers) = controllers {
            std::fs::write(root.path().join("cgroup.controllers"), controllers).unwrap();
        }
        root
    };
    let root = tree(Some(
        "cpuset cpu io memory pids
",
    ));
    assert_eq!(
        cgroup_limit(root.path(), hybrid),
        Some(512 * MIB),
        "v2 holds memory"
    );
    let root = tree(Some(
        "cpuset cpu io pids
",
    ));
    assert_eq!(
        cgroup_limit(root.path(), hybrid),
        Some(256 * MIB),
        "v2 without memory"
    );
    let root = tree(None);
    assert_eq!(
        cgroup_limit(root.path(), hybrid),
        Some(256 * MIB),
        "no cgroup.controllers"
    );
}

/// Inside a container the process's cgroup path need not exist under the container's mount;
/// the walk up still reaches the root and finds the container's own limit, v2 and v1 alike.
#[test]
fn a_container_whose_path_is_not_mounted_finds_the_roots_limit() {
    let root = Root::new();
    std::fs::write(
        root.path().join("memory.max"),
        "268435456
",
    )
    .unwrap();
    assert_eq!(
        cgroup_limit(
            root.path(),
            "0::/docker/abc
"
        ),
        Some(256 * MIB)
    );

    let root = Root::new();
    cgroups(
        root.path(),
        "memory.limit_in_bytes",
        &[("/memory", "268435456")],
    );
    assert_eq!(
        cgroup_limit(
            root.path(),
            "4:memory:/docker/abc
"
        ),
        Some(256 * MIB)
    );
}
