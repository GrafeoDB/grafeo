//! The memory this process may use, the base of the default memory limit (#532).
//!
//! [`detect_memory`] reads the smaller of the machine's physical memory and every limit set on
//! the process from outside:
//!
//! - **Linux**: `MemTotal` from `/proc/meminfo`, and the cgroup limits on the path from the
//!   process's cgroup up to the root (v2 `memory.max`, v1 `memory.limit_in_bytes`), so a limit
//!   set on a parent slice or a pod counts. `max`, v1's "unlimited" value and a missing file are
//!   no limit at that level.
//! - **Windows**: the physical memory, and the memory limits of the job object the process
//!   runs in.
//! - **macOS**: `hw.memsize`.
//! - **wasm32**: a fixed platform figure, [`WASM_PLATFORM_MEMORY`] (a 768 MiB budget). A
//!   browser tab's real ceiling depends on the browser and the device, and nothing the engine
//!   can read reports it; an application that knows better sets the limit itself.
//!
//! When nothing can be read the result is [`FALLBACK_SYSTEM_MEMORY`] (1 GiB, never unlimited),
//! with [`MemorySource::Fallback`] so the open can warn. The default budget is
//! [`DEFAULT_MEMORY_FRACTION`] of the result.

#![allow(
    unsafe_code,
    reason = "the Windows and macOS detection calls are FFI; each unsafe block says why it is sound"
)]

#[cfg(all(not(miri), any(target_os = "linux", test)))]
use std::path::Path;

/// The share of the detected memory the default budget takes.
pub const DEFAULT_MEMORY_FRACTION: f64 = 0.75;

/// The memory used when detection fails: 1 GiB, a 768 MiB budget.
pub const FALLBACK_SYSTEM_MEMORY: usize = 1 << 30;

/// The memory wasm32 counts on: 1 GiB, a 768 MiB budget, the budget WASM has always had. Three
/// quarters of wasm32's 4 GiB address space would sit past where a browser tab dies.
pub const WASM_PLATFORM_MEMORY: usize = 1 << 30;

/// Where a [`DetectedMemory`] figure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemorySource {
    /// The machine's physical memory.
    Machine,
    /// A container or job limit below it (a cgroup level, a Windows job).
    Limit,
    /// Detection failed: [`FALLBACK_SYSTEM_MEMORY`].
    Fallback,
    /// A fixed figure for a platform that has nothing to read, as intended rather than failed
    /// (wasm32: [`WASM_PLATFORM_MEMORY`]).
    Platform,
}

/// The memory this process may use, and where the figure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetectedMemory {
    /// The memory in bytes.
    pub bytes: usize,
    /// Where `bytes` came from.
    pub source: MemorySource,
}

impl DetectedMemory {
    /// The fallback: [`FALLBACK_SYSTEM_MEMORY`], marked as such.
    #[must_use]
    pub const fn fallback() -> Self {
        Self {
            bytes: FALLBACK_SYSTEM_MEMORY,
            source: MemorySource::Fallback,
        }
    }

    /// [`DEFAULT_MEMORY_FRACTION`] of [`bytes`](Self::bytes): the default memory budget.
    #[must_use]
    pub fn default_budget(self) -> usize {
        fraction_of(self.bytes, DEFAULT_MEMORY_FRACTION)
    }
}

/// `fraction` of `bytes`, rounded down: 0 for a negative or NaN fraction, `usize::MAX` at most.
#[must_use]
pub fn fraction_of(bytes: usize, fraction: f64) -> usize {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "`as` saturates (NaN and negatives to 0, overflow to usize::MAX); f64 rounds \
                  sizes above 2^53 bytes by less than one part in 10^15"
    )]
    let budget = (bytes as f64 * fraction) as usize;
    budget
}

/// The memory this process may use: the smaller of the machine's memory and every limit set
/// on the process, or the fallback (see the module docs).
#[must_use]
pub fn detect_memory() -> DetectedMemory {
    detect()
}

// Under Miri, file I/O and FFI are blocked by isolation.
#[cfg(miri)]
fn detect() -> DetectedMemory {
    DetectedMemory::fallback()
}

#[cfg(all(not(miri), target_os = "linux"))]
fn detect() -> DetectedMemory {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok();
    let self_cgroup = std::fs::read_to_string("/proc/self/cgroup").ok();
    linux_memory(
        meminfo.as_deref(),
        Path::new("/sys/fs/cgroup"),
        self_cgroup.as_deref(),
    )
}

#[cfg(all(not(miri), windows))]
fn detect() -> DetectedMemory {
    use windows_sys::Win32::System::JobObjects::{
        JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        QueryInformationJobObject,
    };
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status = MEMORYSTATUSEX {
        dwLength: u32::try_from(size_of::<MEMORYSTATUSEX>()).unwrap_or(u32::MAX),
        ..Default::default()
    };
    // SAFETY: `status` is a valid, writable MEMORYSTATUSEX whose `dwLength` is its size, as
    // the call requires; it writes only into it.
    let machine = if unsafe { GlobalMemoryStatusEx(&raw mut status) } != 0 {
        Some(usize::try_from(status.ullTotalPhys).unwrap_or(usize::MAX))
    } else {
        None
    };

    let mut job = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    let length = u32::try_from(size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>()).unwrap_or(0);
    // SAFETY: a null job handle asks for the job of the calling process; `job` is a valid,
    // writable JOBOBJECT_EXTENDED_LIMIT_INFORMATION of `length` bytes, the size this
    // information class writes, and the returned-length pointer may be null.
    let queried = unsafe {
        QueryInformationJobObject(
            std::ptr::null_mut(),
            JobObjectExtendedLimitInformation,
            (&raw mut job).cast(),
            length,
            std::ptr::null_mut(),
        )
    } != 0;
    let flags = job.BasicLimitInformation.LimitFlags;
    let limit = [
        (JOB_OBJECT_LIMIT_PROCESS_MEMORY, job.ProcessMemoryLimit),
        (JOB_OBJECT_LIMIT_JOB_MEMORY, job.JobMemoryLimit),
    ]
    .into_iter()
    .filter(|&(flag, bytes)| queried && flags & flag != 0 && bytes > 0)
    .map(|(_, bytes)| bytes)
    .min();
    combine(machine, limit)
}

#[cfg(all(not(miri), target_os = "macos"))]
fn detect() -> DetectedMemory {
    let mut bytes: u64 = 0;
    let mut size = size_of::<u64>();
    // SAFETY: the name is a NUL-terminated C string; `hw.memsize` is a 64-bit integer, and
    // `bytes` is a writable u64 whose size is passed in `size`; nothing new is set.
    let read = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut bytes).cast(),
            &raw mut size,
            std::ptr::null_mut(),
            0,
        )
    } == 0;
    let machine = (read && bytes > 0).then(|| usize::try_from(bytes).unwrap_or(usize::MAX));
    combine(machine, None)
}

// wasm32: a fixed figure, not a failure, so no warning.
#[cfg(all(not(miri), target_arch = "wasm32"))]
fn detect() -> DetectedMemory {
    DetectedMemory {
        bytes: WASM_PLATFORM_MEMORY,
        source: MemorySource::Platform,
    }
}

#[cfg(all(
    not(miri),
    not(any(
        target_os = "linux",
        windows,
        target_os = "macos",
        target_arch = "wasm32"
    ))
))]
fn detect() -> DetectedMemory {
    DetectedMemory::fallback()
}

/// The machine's memory and the smallest limit on the process, as one figure: a limit binds
/// only below the machine's memory; with neither, the fallback.
#[cfg(all(
    not(miri),
    any(target_os = "linux", windows, target_os = "macos", test)
))]
fn combine(machine: Option<usize>, limit: Option<usize>) -> DetectedMemory {
    match (machine, limit) {
        (Some(machine), Some(limit)) if limit < machine => DetectedMemory {
            bytes: limit,
            source: MemorySource::Limit,
        },
        (Some(machine), _) => DetectedMemory {
            bytes: machine,
            source: MemorySource::Machine,
        },
        (None, Some(limit)) => DetectedMemory {
            bytes: limit,
            source: MemorySource::Limit,
        },
        (None, None) => DetectedMemory::fallback(),
    }
}

/// A cgroup value at or above this is no limit: v1 writes the largest page-aligned `i64`
/// (`0x7FFF_FFFF_FFFF_F000`) for "unlimited", and no machine has 4 EiB.
#[cfg(all(not(miri), any(target_os = "linux", test)))]
const NO_CGROUP_LIMIT: u64 = 1 << 62;

/// The `MemTotal` of a `/proc/meminfo` text, in bytes.
#[cfg(all(not(miri), any(target_os = "linux", test)))]
pub(crate) fn meminfo_total(meminfo: &str) -> Option<usize> {
    let line = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))?;
    let mut fields = line.split_whitespace();
    let kib: u64 = fields.next()?.parse().ok()?;
    if fields.next().is_some_and(|unit| unit != "kB") {
        return None;
    }
    Some(usize::try_from(kib.checked_mul(1024)?).unwrap_or(usize::MAX))
}

/// The smallest cgroup memory limit on the path from the process's cgroup to the root, given
/// the cgroup filesystem's root (`/sys/fs/cgroup`) and the text of `/proc/self/cgroup`.
///
/// v2 (a `0::<path>` line) reads `memory.max` at `<root>/<path>` and each parent up to the
/// root; v1 (a line naming the `memory` controller) reads `memory.limit_in_bytes` at
/// `<root>/memory/<path>` up to `<root>/memory`. On a hybrid system, with both lines, v2 counts
/// only when `<root>/cgroup.controllers` lists `memory`. A cgroup path that does not exist
/// inside a container's mount is walked up to the root, which is the container's own cgroup.
#[cfg(all(not(miri), any(target_os = "linux", test)))]
pub(crate) fn cgroup_limit(cgroup_root: &Path, proc_self_cgroup: &str) -> Option<usize> {
    let mut v2 = None;
    let mut v1 = None;
    for line in proc_self_cgroup.lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(id), Some(controllers), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if id == "0" && controllers.is_empty() {
            v2 = Some(path);
        } else if controllers.split(',').any(|name| name == "memory") {
            v1 = Some(path);
        }
    }
    let v2_has_memory = || {
        std::fs::read_to_string(cgroup_root.join("cgroup.controllers"))
            .is_ok_and(|controllers| controllers.split_whitespace().any(|name| name == "memory"))
    };
    match (v2, v1) {
        (Some(path), None) => smallest_on_path(cgroup_root, path, "memory.max"),
        (Some(path), Some(_)) if v2_has_memory() => {
            smallest_on_path(cgroup_root, path, "memory.max")
        }
        (_, Some(path)) => {
            smallest_on_path(&cgroup_root.join("memory"), path, "memory.limit_in_bytes")
        }
        (None, None) => None,
    }
}

/// The smallest limit in `file` at `<base>/<path>` and each parent up to `base`.
#[cfg(all(not(miri), any(target_os = "linux", test)))]
fn smallest_on_path(base: &Path, path: &str, file: &str) -> Option<usize> {
    let relative = Path::new(path.trim_start_matches('/'));
    relative
        .ancestors()
        .filter_map(|level| std::fs::read_to_string(base.join(level).join(file)).ok())
        .filter_map(|value| {
            let value = value.trim();
            // `max` (v2) and anything unparsable are no limit at this level.
            let bytes: u64 = value.parse().ok()?;
            (bytes < NO_CGROUP_LIMIT).then(|| usize::try_from(bytes).unwrap_or(usize::MAX))
        })
        .min()
}

/// Linux detection over texts and a root, for the fixtures: `meminfo` and `proc_self_cgroup`
/// stand for the texts of `/proc/meminfo` and `/proc/self/cgroup` (`None` when unreadable),
/// `cgroup_root` for `/sys/fs/cgroup`.
#[cfg(all(not(miri), any(target_os = "linux", test)))]
pub(crate) fn linux_memory(
    meminfo: Option<&str>,
    cgroup_root: &Path,
    proc_self_cgroup: Option<&str>,
) -> DetectedMemory {
    combine(
        meminfo.and_then(meminfo_total),
        proc_self_cgroup.and_then(|text| cgroup_limit(cgroup_root, text)),
    )
}

// The fixtures write and read cgroup trees on disk, which Miri's isolation refuses; the code
// they test does no unsafe, so Miri has nothing to check there.
#[cfg(all(test, not(miri)))]
mod tests;
