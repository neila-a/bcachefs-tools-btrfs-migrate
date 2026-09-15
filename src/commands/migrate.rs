// SPDX-License-Identifier: GPL-2.0
//
// bcachefs migrate — convert an existing filesystem to bcachefs in place.
//
// Reimplements c_src/cmd_migrate.c in Rust.
//
// The reservation file is what makes this safe: everything bcachefs writes
// lands in blocks the old filesystem has allocated to /bcachefs, so the old
// filesystem stays consistent and mountable throughout. That also fixes where
// cleanup is allowed — before format() the reservation is disposable, after it
// the file is what keeps the old filesystem's allocator away from blocks
// bcachefs is using, and removing it would corrupt both filesystems.

use std::ffi::{CString, c_char, c_ulong};
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use bch_bindgen::fs::FsExt;
use bch_bindgen::c;
use bcachefs_kernel::fs::Fs;
use bcachefs_kernel::opt_set;
use clap::Parser;

use crate::commands::format::take_opt_value;
use crate::commands::opts::{bch_opt_lookup_negated, parse_opt_val};
use crate::key::Passphrase;
use crate::commands::format_util::{format_opts_default, DevOpts};
use crate::util::fmt_bytes_human;
use crate::btrfs_migrate::{preflight_source, ExtentMapper, BTRFS_BLOCK_GROUP_DATA_MASK};
use crate::wrappers::super_io;

// ---- C shim declarations ----

extern "C" {
    fn rust_set_bit(nr: c_ulong, addr: *mut c_ulong);
}

use fiemap::FiemapExtentFlags;

/// Iterate over physical extents of a file via FIEMAP ioctl.
fn fiemap_iter(fd: BorrowedFd<'_>, mapper: &ExtentMapper) -> Result<Vec<CRange>> {
    let known_flags = FiemapExtentFlags::LAST
        | FiemapExtentFlags::UNKNOWN
        | FiemapExtentFlags::DELALLOC
        | FiemapExtentFlags::ENCODED
        | FiemapExtentFlags::DATA_ENCRYPTED
        | FiemapExtentFlags::NOT_ALIGNED
        | FiemapExtentFlags::DATA_INLINE
        | FiemapExtentFlags::DATA_TAIL
        | FiemapExtentFlags::UNWRITTEN
        | FiemapExtentFlags::MERGED
        | FiemapExtentFlags::SHARED;

    let mut ranges = Vec::new();
    for extent in fiemap::Fiemap::new(fd) {
        let extent = extent?;
        if extent.fe_length == 0 {
            bail!("Unable to continue: FIEMAP returned a zero-length metadata extent");
        }
        if extent.fe_flags.bits() & !known_flags.bits() != 0
            || extent.fe_flags.intersects(
                FiemapExtentFlags::UNKNOWN
                    | FiemapExtentFlags::DELALLOC
                    | FiemapExtentFlags::ENCODED
                    | FiemapExtentFlags::DATA_ENCRYPTED
                    | FiemapExtentFlags::NOT_ALIGNED
                    | FiemapExtentFlags::DATA_INLINE
                    | FiemapExtentFlags::DATA_TAIL
                    | FiemapExtentFlags::MERGED
                    | FiemapExtentFlags::SHARED,
            )
        {
            bail!(
                "Unable to continue: metadata file extent flags {:?} are not stable raw extents",
                extent.fe_flags
            );
        }

        for segment in mapper
            .map_physical(extent.fe_physical, extent.fe_length)
            .map_err(|e| anyhow!("Unable to map metadata file extent: {}", e))?
        {
            let end = segment
                .physical
                .checked_add(segment.len)
                .ok_or_else(|| anyhow!("metadata extent raw range overflows"))?;
            ranges.push(CRange {
                start: segment.physical,
                end,
            });
        }
    }

    ranges_sort_merge(&mut ranges);
    Ok(ranges)
}

// ---- Range utilities ----

/// Matches the C `struct range { u64 start; u64 end; }`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CRange {
    start: u64,
    end:   u64,
}

fn ranges_sort_merge(ranges: &mut Vec<CRange>) {
    ranges.sort_by_key(|r| r.start);
    let mut merged = Vec::with_capacity(ranges.len());
    for r in ranges.drain(..) {
        if let Some(last) = merged.last_mut() {
            let last: &mut CRange = last;
            if last.end >= r.start {
                last.end = last.end.max(r.end);
                continue;
            }
        }
        merged.push(r);
    }
    *ranges = merged;
}

/// Iterate over holes in the range [0, max) not covered by `ranges`.
/// `ranges` must be sorted and non-overlapping.
fn for_each_hole(ranges: &[CRange], max: u64) -> Vec<CRange> {
    let mut holes = Vec::new();
    let mut pos = 0u64;
    for r in ranges {
        if r.start > pos {
            holes.push(CRange { start: pos, end: r.start });
        }
        pos = r.end;
    }
    if pos < max {
        holes.push(CRange { start: pos, end: max });
    }
    holes
}

// ---- Utility functions ----

/// Resolve a dev_t to a device path via /sys/dev/block/.
fn dev_t_to_path(dev: libc::dev_t) -> Result<String> {
    let major = libc::major(dev);
    let minor = libc::minor(dev);
    let sysfs = format!("/sys/dev/block/{}:{}", major, minor);

    let link = fs::read_link(&sysfs)
        .map_err(|e| anyhow!("readlink error looking up block device {}: {}", sysfs, e))?;

    let name = link.file_name()
        .ok_or_else(|| anyhow!("error looking up device name from {}", link.display()))?;

    Ok(format!("/dev/{}", name.to_string_lossy()))
}

/// Check if a path is a filesystem mount point.
fn path_is_fs_root(path: &str) -> Result<bool> {
    let mountinfo = fs::read_to_string("/proc/self/mountinfo")
        .map_err(|e| anyhow!("Error reading mount information: {}", e))?;

    for line in mountinfo.lines() {
        // Fields: mount_id parent_id dev root mount_point ...
        let mut fields = line.split(' ');
        let _mount_id = fields.next();
        let _parent_id = fields.next();
        let _dev = fields.next();
        let _root = fields.next();
        if let Some(mount_point) = fields.next() {
            if mount_point == path {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// sector_to_bucket: sector / bucket_size
fn sector_to_bucket(bucket_size: u32, sector: u64) -> u64 {
    sector / bucket_size as u64
}

/// bucket_to_sector: bucket * bucket_size
fn bucket_to_sector(bucket_size: u32, bucket: u64) -> u64 {
    bucket * bucket_size as u64
}

/// Mark a range of buckets as nouse.
unsafe fn mark_nouse_range(ca: *const c::bch_dev, sector_from: u64, sector_to: u64) {
    let bucket_size = (*ca).mi.bucket_size as u32;
    let nouse = (*ca).buckets_nouse;
    let mut b = sector_to_bucket(bucket_size, sector_from);
    loop {
        rust_set_bit(b as c_ulong, nouse);
        b += 1;
        if bucket_to_sector(bucket_size, b) >= sector_to {
            break;
        }
    }
}

/// Mark all space NOT covered by the reserved extents as nouse,
/// plus the space for the default superblock layout.
unsafe fn mark_unreserved_space(fs: *mut c::bch_fs, extents: &[CRange]) {
    let ca = (*fs).devs[0];
    let bucket_size = (*ca).mi.bucket_size as u32;
    let nbuckets = (*ca).mi.nbuckets;

    // Byte-addressed max for hole iteration
    let max_bytes = bucket_to_sector(bucket_size, nbuckets) << 9;

    for hole in for_each_hole(extents, max_bytes) {
        if hole.start == hole.end {
            continue;
        }
        mark_nouse_range(ca, hole.start >> 9, hole.end.div_ceil(1 << 9));
    }

    // Also mark space for the default sb layout
    let sb_size = 1u64 << (*(*ca).disk_sb.sb).layout.sb_max_size_bits;
    mark_nouse_range(ca, 0, c::BCH_SB_SECTOR as u64 + sb_size * 2);
}

/// Reserve space for bcachefs metadata file, return extents and inode number.
///
/// Nothing has been written to the block device yet, so on failure we put the
/// host filesystem back the way we found it rather than sitting on the space.
fn reserve_new_fs_space(
    file_path: &str,
    block_size: u32,  // in sectors
    dev_size: u64,
    dev: libc::dev_t,
    force: bool,
    mapper: &ExtentMapper,
) -> Result<(Vec<CRange>, u64)> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    let mut open_opts = std::fs::OpenOptions::new();
    open_opts.read(true).write(true).mode(0o600);
    if force {
        open_opts.create(true);
    } else {
        open_opts.create_new(true);
    }
    let file = open_opts.open(file_path)
        .map_err(|e| anyhow!("Error creating {} for bcachefs metadata: {}", file_path, e))?;

    let meta = file.metadata()
        .map_err(|e| anyhow!("fstat: {}", e))?;
    if meta.dev() != dev {
        bail!("bcachefs file has incorrect device");
    }
    let bcachefs_inum = meta.ino();

    // With --force this may be a reservation from a run that already got as far
    // as format(): those blocks hold bcachefs metadata, and releasing them to
    // the old filesystem's allocator would corrupt both filesystems. Roll back
    // to what we opened, not to empty.
    let orig_len = meta.len();

    // A tenth of the device is a fudge factor for the new filesystem's
    // metadata: computing what we really need means accounting for inodes,
    // extents (which depends on fragmentation) and alloc info.
    let extents = reserve_extents(&file, block_size, dev_size / 10, mapper);
    if extents.is_err() {
        let cleanup = if orig_len == 0 {
            fs::remove_file(file_path)
        } else {
            file.set_len(orig_len)
        };

        if let Err(e) = cleanup {
            eprintln!("Error releasing space in {}: {} - it is still in use", file_path, e);
        }
    }

    Ok((extents?, bcachefs_inum))
}

/// fallocate() the reservation and return the physical extents backing it.
fn reserve_extents(
    file: &fs::File,
    block_size: u32,
    size: u64,
    mapper: &ExtentMapper,
) -> Result<Vec<CRange>> {
    let vfs = rustix::fs::fstatvfs(file).map_err(|e| anyhow!("fstatvfs: {}", e))?;
    let free = vfs.f_bavail.saturating_mul(vfs.f_frsize);

    // fallocate() has no rollback: when it runs out of space partway through it
    // keeps everything it allocated, and on XFS flags it as explicit
    // preallocation so nothing reclaims it in the background. Our caller
    // truncates on error.
    rustix::fs::fallocate(file, rustix::fs::FallocateFlags::empty(), 0, size)
        .map_err(|e| if e == rustix::io::Errno::NOSPC {
            anyhow!("Not enough free space to migrate: bcachefs metadata needs {}, {} available",
                    fmt_bytes_human(size), fmt_bytes_human(free))
        } else {
            anyhow!("Error reserving space ({} bytes) for bcachefs metadata: {}", size, e)
        })?;

    rustix::fs::fsync(file).map_err(|e| anyhow!("fsync: {}", e))?;

    let extents = fiemap_iter(file.as_fd(), mapper)?;

    // Check alignment
    let block_bytes = (block_size as u64) << 9;
    for r in &extents {
        if (r.start & (block_bytes - 1)) != 0 || ((r.end - r.start) & (block_bytes - 1)) != 0 {
            bail!("Unable to continue: unaligned extents in metadata file");
        }
    }

    Ok(extents)
}

/// Find space within the reserved extents for the superblock.
///
/// @bucket_bytes is the bucket_size option, which is in bytes - only the
/// superblock member field it maps to is in sectors (OPT_SB_FIELD_SECTORS).
/// @sb_size is in sectors.
fn find_superblock_space(extents: &[CRange], sb_size: u32, bucket_bytes: u32) -> Result<(u64, u64)> {
    if bucket_bytes == 0 {
        bail!("superblock bucket size cannot be zero");
    }
    let bucket_bytes = bucket_bytes as u64;
    let sb_bytes = (sb_size as u64) << 9;
    let required_bytes = sb_bytes
        .checked_mul(2)
        .ok_or_else(|| anyhow!("superblock reservation size overflows"))?;
    let required_sectors = (sb_size as u64)
        .checked_mul(2)
        .ok_or_else(|| anyhow!("superblock reservation sector count overflows"))?;

    for r in extents {
        let start = std::cmp::max(256 << 10, r.start)
            .div_ceil(bucket_bytes)
            .checked_mul(bucket_bytes)
            .ok_or_else(|| anyhow!("superblock alignment overflows"))?;
        let end = (r.end / bucket_bytes) * bucket_bytes;

        // Need space for two superblocks
        if start
            .checked_add(required_bytes)
            .is_some_and(|candidate| candidate <= end)
        {
            let start_sector = start >> 9;
            let end_sector = start_sector
                .checked_add(required_sectors)
                .ok_or_else(|| anyhow!("superblock sector range overflows"))?;
            return Ok((start_sector, end_sector));
        }
    }
    bail!("Couldn't find a valid location for superblock");
}

/// Add default superblock layout entries (sector 8 and sector 8+sb_size).
/// Returns the sb_size.
fn add_default_sb_layout(sb: &mut c::bch_sb) -> Result<u32> {
    let sb_size = 1u32 << sb.layout.sb_max_size_bits;
    let bch_sb_sector = c::BCH_SB_SECTOR as u64;

    if sb.layout.nr_superblocks as usize >= sb.layout.sb_offset.len() {
        bail!("Can't add superblock: no space left in superblock layout");
    }

    for i in 0..sb.layout.nr_superblocks as usize {
        let off = u64::from_le(sb.layout.sb_offset[i]);
        if off == bch_sb_sector || off == bch_sb_sector + sb_size as u64 {
            bail!("Superblock layout already has default superblocks");
        }
    }

    // Shift existing entries down by 2
    let nr = sb.layout.nr_superblocks as usize;
    sb.layout.sb_offset.copy_within(0..nr, 2);
    sb.layout.nr_superblocks += 2;
    sb.layout.sb_offset[0] = bch_sb_sector.to_le();
    sb.layout.sb_offset[1] = (bch_sb_sector + sb_size as u64).to_le();

    Ok(sb_size)
}

const BTRFS_SUPER_MAGIC: u64 = c::BCH_BTRFS_SUPER_MAGIC as u64;

struct SourceInfo {
    source_fs_dev: u64,
    raw_device_path: PathBuf,
    raw_device_size: u64,
    raw_alignment: u64,
    mapper: ExtentMapper,
    btrfs: Option<BtrfsSourceSummary>,
}
#[derive(Clone, Debug)]
struct BtrfsSourceSummary {
    fsid: [u8; 16],
    devid: u64,
    subvolid: u64,
    num_devices: u64,
    incompat_flags: u64,
    sectorsize: u32,
    chunk_count: usize,
    subvolume_count: usize,
    data_chunk_count: usize,
}


fn migration_block_alignment(raw_device_path: &Path, raw_device_size: u64) -> u64 {
    if raw_device_size < 1 << 30 {
        return 512;
    }
    let physical_hint = File::open(raw_device_path)
        .map(|device| crate::wrappers::bdev::get_blocksize_physical_hint(device.as_raw_fd()))
        .unwrap_or(0) as u64;
    if physical_hint.is_power_of_two() {
        physical_hint.max(4096)
    } else {
        4096
    }
}

fn inspect_source(fs_path: &Path) -> Result<SourceInfo> {
    if !path_is_fs_root(fs_path.to_string_lossy().as_ref())? {
        bail!("{} is not a filesystem root", fs_path.display());
    }

    let source_meta = fs::symlink_metadata(fs_path)
        .map_err(|e| anyhow!("Error stating {}: {}", fs_path.display(), e))?;
    if !source_meta.file_type().is_dir() {
        bail!("{} is not a directory", fs_path.display());
    }
    let source_fs_dev = source_meta.dev();

    if path_is_btrfs(fs_path)? {
        let preflight = preflight_source(fs_path)?;
        let btrfs = BtrfsSourceSummary {
            fsid: preflight.fsid,
            devid: preflight.devid,
            subvolid: preflight.subvolid,
            num_devices: preflight.num_devices,
            incompat_flags: preflight.incompat_flags,
            sectorsize: preflight.sectorsize,
            chunk_count: preflight.mapper.chunks().len(),
            subvolume_count: preflight.subvolume_count,
            data_chunk_count: preflight
                .mapper
                .chunks()
                .iter()
                .filter(|chunk| chunk.chunk_type & BTRFS_BLOCK_GROUP_DATA_MASK != 0)
                .count(),
        };
        let raw_alignment =
            migration_block_alignment(&preflight.raw_device_path, preflight.raw_device_size);
        return Ok(SourceInfo {
            source_fs_dev: preflight.source_fs_dev,
            raw_device_size: preflight.raw_device_size,
            raw_device_path: preflight.raw_device_path,
            raw_alignment,
            mapper: ExtentMapper::Btrfs(preflight.mapper),
            btrfs: Some(btrfs),
        });
    }

    let raw_device_path = PathBuf::from(dev_t_to_path(source_fs_dev as libc::dev_t)?);
    let raw_device_size = match File::open(&raw_device_path) {
        Ok(raw_device) => crate::wrappers::bdev::get_size(raw_device.as_raw_fd()),
        Err(error) => {
            eprintln!(
                "migration preflight: raw device {} is not readable by this user ({}); continuing with identity mapping and no raw-size bound",
                raw_device_path.display(),
                error
            );
            0
        }
    };
    let raw_alignment = migration_block_alignment(&raw_device_path, raw_device_size);

    Ok(SourceInfo {
        source_fs_dev,
        raw_device_path,
        raw_device_size,
        raw_alignment,
        mapper: ExtentMapper::Identity,
        btrfs: None,
    })
}

fn path_is_btrfs(path: &Path) -> Result<bool> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|e| anyhow!("Error opening {}: {}", path.display(), e))?;
    let mut stat = std::mem::MaybeUninit::<libc::statfs>::zeroed();
    let ret = unsafe { libc::fstatfs(file.as_raw_fd(), stat.as_mut_ptr()) };
    if ret < 0 {
        bail!(
            "fstatfs({}) failed: {}",
            path.display(),
            io::Error::last_os_error()
        );
    }
    Ok(unsafe { stat.assume_init() }.f_type as u64 == BTRFS_SUPER_MAGIC)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NoChangeCopyReason {
    Shared,
    Encoded,
    Encrypted,
    NotAligned,
    Inline,
    Tail,
    Merged,
    FutureFlags,
    RawUnaligned,
}

#[derive(Default)]
struct NoChangeStats {
    directories: u64,
    files: u64,
    symlinks: u64,
    special: u64,
    total_input: u64,
    linked: u64,
    copied: u64,
    unwritten: u64,
    unstable: u64,
    mapped_extents: u64,
    mapped_segments: u64,
    chunk_splits: u64,
    copied_shared: u64,
    copied_encoded: u64,
    copied_encrypted: u64,
    copied_not_aligned: u64,
    copied_inline: u64,
    copied_tail: u64,
    copied_merged: u64,
    copied_future_flags: u64,
    copied_other_flag: u64,
    copied_raw_unaligned: u64,
    mapping_failures: u64,
    reservation_bytes: u64,
    free_bytes: u64,
    reservation_margin_bytes: u64,
    capacity_valid: bool,
    eligible: bool,
}

impl NoChangeStats {
    fn record_copy(&mut self, reason: NoChangeCopyReason, bytes: u64) {
        self.copied = self.copied.saturating_add(bytes);
        match reason {
            NoChangeCopyReason::Encoded => self.copied_encoded += bytes,
            NoChangeCopyReason::Encrypted => self.copied_encrypted += bytes,
            NoChangeCopyReason::Shared => self.copied_shared += bytes,
            NoChangeCopyReason::NotAligned => self.copied_not_aligned += bytes,
            NoChangeCopyReason::Inline => {
                self.copied_inline += bytes;
            }
            NoChangeCopyReason::Tail => {
                self.copied_tail += bytes;
                self.copied_other_flag += bytes;
            }
            NoChangeCopyReason::Merged => {
                self.copied_merged += bytes;
                self.copied_other_flag += bytes;
            }
            NoChangeCopyReason::FutureFlags => {
                self.copied_future_flags += bytes;
                self.copied_other_flag += bytes;
            }
            NoChangeCopyReason::RawUnaligned => self.copied_raw_unaligned += bytes,
        }
    }
}

fn no_change_copy_reason(flags: fiemap::FiemapExtentFlags) -> Option<NoChangeCopyReason> {
    let known = FiemapExtentFlags::LAST
        | FiemapExtentFlags::UNKNOWN
        | FiemapExtentFlags::DELALLOC
        | FiemapExtentFlags::ENCODED
        | FiemapExtentFlags::DATA_ENCRYPTED
        | FiemapExtentFlags::NOT_ALIGNED
        | FiemapExtentFlags::DATA_INLINE
        | FiemapExtentFlags::DATA_TAIL
        | FiemapExtentFlags::UNWRITTEN
        | FiemapExtentFlags::MERGED
        | FiemapExtentFlags::SHARED;
    if flags.contains(FiemapExtentFlags::ENCODED) {
        Some(NoChangeCopyReason::Encoded)
    } else if flags.contains(FiemapExtentFlags::DATA_ENCRYPTED) {
        Some(NoChangeCopyReason::Encrypted)
    } else if flags.contains(FiemapExtentFlags::SHARED) {
        Some(NoChangeCopyReason::Shared)
    } else if flags.contains(FiemapExtentFlags::DATA_INLINE) {
        Some(NoChangeCopyReason::Inline)
    } else if flags.contains(FiemapExtentFlags::DATA_TAIL) {
        Some(NoChangeCopyReason::Tail)
    } else if flags.contains(FiemapExtentFlags::NOT_ALIGNED) {
        Some(NoChangeCopyReason::NotAligned)
    } else if flags.contains(FiemapExtentFlags::MERGED) {
        Some(NoChangeCopyReason::Merged)
    } else if flags.bits() & !known.bits() != 0 {
        Some(NoChangeCopyReason::FutureFlags)
    } else {
        None
    }
}

fn scan_no_change_file(
    path: &Path,
    source: &SourceInfo,
    stats: &mut NoChangeStats,
) -> Result<()> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|e| anyhow!("cannot open {} read-only for FIEMAP: {}", path.display(), e))?;
    let metadata = file
        .metadata()
        .map_err(|e| anyhow!("cannot stat {}: {}", path.display(), e))?;
    if metadata.dev() != source.source_fs_dev {
        bail!(
            "source tree crosses filesystem device boundary at {} ({} -> {})",
            path.display(),
            source.source_fs_dev,
            metadata.dev()
        );
    }

    for extent in fiemap::Fiemap::new(&file) {
        let extent = extent
            .map_err(|e| anyhow!("cannot read FIEMAP for {}: {}", path.display(), e))?;
        if extent.fe_length == 0 {
            bail!("FIEMAP returned a zero-length extent for {}", path.display());
        }
        stats.total_input = stats
            .total_input
            .checked_add(extent.fe_length)
            .ok_or_else(|| anyhow!("no-change input byte counter overflow"))?;

        if extent.fe_flags.intersects(FiemapExtentFlags::UNKNOWN | FiemapExtentFlags::DELALLOC) {
            stats.unstable = stats
                .unstable
                .checked_add(extent.fe_length)
                .ok_or_else(|| anyhow!("no-change unstable byte counter overflow"))?;
            stats.eligible = false;
            continue;
        }

        let src_max = metadata.len().div_ceil(source.raw_alignment);
        let src_max = src_max
            .checked_mul(source.raw_alignment)
            .ok_or_else(|| anyhow!("no-change source extent bound overflow"))?;
        if extent.fe_logical > src_max {
            bail!(
                "FIEMAP extent starts past EOF for {}: {} > {}",
                path.display(),
                extent.fe_logical,
                src_max
            );
        }
        let length = extent.fe_length.min(src_max - extent.fe_logical);
        let visible_len = if extent.fe_logical >= metadata.len() {
            0
        } else {
            (metadata.len() - extent.fe_logical).min(length)
        };

        if extent.fe_flags.contains(FiemapExtentFlags::UNWRITTEN) {
            stats.unwritten = stats
                .unwritten
                .checked_add(visible_len)
                .ok_or_else(|| anyhow!("no-change unwritten byte counter overflow"))?;
            continue;
        }
        if let Some(reason) = no_change_copy_reason(extent.fe_flags) {
            stats.record_copy(reason, visible_len);
            continue;
        }

        let segments = match source
            .mapper
            .map_physical(extent.fe_physical, length)
        {
            Ok(segments) => segments,
            Err(error) => {
                eprintln!(
                    "no-change cannot map {} logical extent: {}",
                    path.display(),
                    error
                );
                stats.mapping_failures = stats.mapping_failures.saturating_add(1);
                stats.eligible = false;
                continue;
            }
        };
        stats.mapped_extents = stats.mapped_extents.saturating_add(1);
        stats.mapped_segments = stats
            .mapped_segments
            .saturating_add(segments.len() as u64);
        stats.chunk_splits = stats
            .chunk_splits
            .saturating_add((segments.len() as u64).saturating_sub(1));

        let mut raw_aligned = true;
        for segment in &segments {
            let segment_file_logical = segment
                .source_logical
                .checked_sub(extent.fe_physical)
                .and_then(|delta| extent.fe_logical.checked_add(delta))
                .ok_or_else(|| anyhow!("no-change file offset translation overflow"))?;
            if (segment_file_logical | segment.physical | segment.len)
                & (source.raw_alignment - 1)
                != 0
            {
                raw_aligned = false;
                break;
            }
        }
        if raw_aligned {
            stats.linked = stats.linked.saturating_add(visible_len);
        } else {
            stats.record_copy(NoChangeCopyReason::RawUnaligned, visible_len);
        }
    }
    Ok(())
}

fn scan_no_change_tree(source_path: &Path, source: &SourceInfo) -> Result<NoChangeStats> {
    let mut stats = NoChangeStats {
        eligible: true,
        ..Default::default()
    };
    let mut directories = vec![source_path.to_path_buf()];
    let mut visited = std::collections::HashSet::new();
    let mut seen_files = std::collections::HashSet::new();

    while let Some(path) = directories.pop() {
        let metadata = fs::symlink_metadata(&path)
            .map_err(|e| anyhow!("cannot inspect {}: {}", path.display(), e))?;
        if metadata.dev() != source.source_fs_dev {
            bail!(
                "source tree crosses filesystem device boundary at {} ({} -> {})",
                path.display(),
                source.source_fs_dev,
                metadata.dev()
            );
        }
        if !metadata.file_type().is_dir() {
            continue;
        }
        if !visited.insert((metadata.dev(), metadata.ino())) {
            continue;
        }
        stats.directories = stats.directories.saturating_add(1);

        for entry in fs::read_dir(&path)
            .map_err(|e| anyhow!("cannot enumerate {}: {}", path.display(), e))?
        {
            let entry = entry.map_err(|e| anyhow!("cannot read entry under {}: {}", path.display(), e))?;
            let child = entry.path();
            let child_meta = fs::symlink_metadata(&child)
                .map_err(|e| anyhow!("cannot inspect {}: {}", child.display(), e))?;
            let file_type = child_meta.file_type();
            if file_type.is_dir() {
                directories.push(child);
            } else if file_type.is_symlink() {
                stats.symlinks = stats.symlinks.saturating_add(1);
            } else if file_type.is_file() {
                stats.files = stats.files.saturating_add(1);
                if seen_files.insert((child_meta.dev(), child_meta.ino())) {
                    scan_no_change_file(&child, source, &mut stats)?;
                }
            } else {
                stats.special = stats.special.saturating_add(1);
            }
        }
    }
    Ok(stats)
}

const MIN_RESERVATION_MARGIN_BYTES: u64 = 8 * 1024 * 1024 * 1024;

fn no_change_capacity(source_path: &Path, source: &SourceInfo) -> Result<(u64, u64, u64, bool)> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(source_path)
        .map_err(|e| anyhow!("cannot open {} for statfs: {}", source_path.display(), e))?;
    let vfs = rustix::fs::fstatvfs(&file).map_err(|e| anyhow!("fstatvfs: {}", e))?;
    let free_bytes = vfs.f_bavail.saturating_mul(vfs.f_frsize);
    let reservation_bytes = source.raw_device_size / 10;
    let reservation_margin_bytes = free_bytes.saturating_sub(reservation_bytes);
    let valid = source.raw_device_size != 0
        && free_bytes >= reservation_bytes
        && reservation_margin_bytes >= MIN_RESERVATION_MARGIN_BYTES;
    Ok((
        reservation_bytes,
        free_bytes,
        reservation_margin_bytes,
        valid,
    ))
}

fn print_preflight(source_path: &Path, source: &SourceInfo) {
    if let Some(btrfs) = &source.btrfs {
        println!(
            "MIGRATE_PREFLIGHT source_fs=btrfs fs_type=btrfs source={} source_fs_dev={} raw_device={} raw_device_size={} fsid={} devid={} num_devices={} data_profile=SINGLE subvolid={} child_subvolumes={} subvolume_count={} incompat_flags=0x{:x} sectorsize={} chunk_count={} data_chunk_count={} raw_alignment={}",
            source_path.display(),
            source.source_fs_dev,
            source.raw_device_path.display(),
            source.raw_device_size,
            hex_bytes(&btrfs.fsid),
            btrfs.devid,
            btrfs.num_devices,
            btrfs.subvolid,
            btrfs.subvolume_count,
            btrfs.subvolume_count,
            btrfs.incompat_flags,
            btrfs.sectorsize,
            btrfs.chunk_count,
            btrfs.data_chunk_count,
            source.raw_alignment,
        );
    } else {
        println!(
            "MIGRATE_PREFLIGHT source_fs=identity fs_type=identity source={} source_fs_dev={} raw_device={} raw_device_size={} data_profile=identity child_subvolumes=0 subvolume_count=0 alignment={}",
            source_path.display(),
            source.source_fs_dev,
            source.raw_device_path.display(),
            source.raw_device_size,
            source.raw_alignment,
        );
    }
}

fn print_no_change_plan(stats: &NoChangeStats) {
    println!(
        "MIGRATE_PLAN mode=no-change directories={} files={} symlinks={} special={} total_input_bytes={} linked_bytes={} copied_bytes={} copied_shared_bytes={} copied_encoded_bytes={} copied_encrypted_bytes={} copied_inline_bytes={} copied_not_aligned_bytes={} copied_other_flag_bytes={} unwritten_hole_bytes={} unstable_extent_bytes={} mapped_extent_count={} mapped_segment_count={} chunk_split_count={} copied_tail_bytes={} copied_merged_bytes={} copied_future_flags_bytes={} copied_raw_unaligned_bytes={} mapping_failures={} reservation_bytes={} free_bytes={} reservation_margin_bytes={}",
        stats.directories,
        stats.files,
        stats.symlinks,
        stats.special,
        stats.total_input,
        stats.linked,
        stats.copied,
        stats.copied_shared,
        stats.copied_encoded,
        stats.copied_encrypted,
        stats.copied_inline,
        stats.copied_not_aligned,
        stats.copied_other_flag,
        stats.unwritten,
        stats.unstable,
        stats.mapped_extents,
        stats.mapped_segments,
        stats.chunk_splits,
        stats.copied_tail,
        stats.copied_merged,
        stats.copied_future_flags,
        stats.copied_raw_unaligned,
        stats.mapping_failures,
        stats.reservation_bytes,
        stats.free_bytes,
        stats.reservation_margin_bytes,
    );
}

fn run_no_change(source_path: &Path, source: &SourceInfo) -> Result<()> {
    let (reservation_bytes, free_bytes, reservation_margin_bytes, capacity_valid) =
        match no_change_capacity(source_path, source) {
            Ok(capacity) => capacity,
            Err(error) => {
                println!("MIGRATE_RESULT eligible=no");
                return Err(error);
            }
        };
    println!(
        "MIGRATE_PREFLIGHT reservation_bytes={} free_bytes={} reservation_margin_bytes={} reservation_margin_required_bytes={} capacity_valid={}",
        reservation_bytes,
        free_bytes,
        reservation_margin_bytes,
        MIN_RESERVATION_MARGIN_BYTES,
        if capacity_valid { "yes" } else { "no" },
    );
    let mut stats = match scan_no_change_tree(source_path, source) {
        Ok(stats) => stats,
        Err(error) => {
            println!("MIGRATE_RESULT eligible=no");
            return Err(error);
        }
    };
    stats.reservation_bytes = reservation_bytes;
    stats.free_bytes = free_bytes;
    stats.reservation_margin_bytes = reservation_margin_bytes;
    stats.capacity_valid = capacity_valid;
    print_no_change_plan(&stats);
    let eligible = stats.eligible && stats.mapping_failures == 0 && stats.capacity_valid;
    println!(
        "MIGRATE_RESULT eligible={}",
        if eligible { "yes" } else { "no" }
    );
    if !eligible {
        bail!("no-change preflight found unstable, unmappable, or insufficient-space conditions");
    }
    Ok(())
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

// ---- Main commands ----

fn migrate_usage() {
    print!("\
bcachefs migrate - migrate an existing filesystem to bcachefs
Usage: bcachefs migrate [OPTION]... [SOURCE]

Options:
  -f fs                        Root of filesystem to migrate (or positional SOURCE)
  -n, --no-change              Read-only preflight and migration plan; never mutate
      --encrypted              Enable whole filesystem encryption (chacha20/poly1305)
      --no_passphrase          Store master encryption key unencrypted in superblock
  -F                           Force, even if metadata file already exists
  -h, --help                   Display this help and exit

Report bugs to <$$EMAIL_X7DPVIQXXN8Q:L$$>
");
}

fn migrate_fs(
    fs_path: &str,
    source: SourceInfo,
    fs_opt_strs: c::bch_opt_strs,
    mut fs_opts: c::bch_opts,
    format_opts: c::format_opts,
    force: bool,
) -> Result<()> {
    let fs_path_cstr = CString::new(fs_path)?;
    let fs_file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOATIME | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(fs_path)
        .map_err(|e| anyhow!("Error opening {}: {}", fs_path, e))?;

    let fs_meta = fs_file.metadata()
        .map_err(|e| anyhow!("fstat: {}", e))?;
    if !fs_meta.is_dir() {
        bail!("{} is not a directory", fs_path);
    }
    if fs_meta.dev() != source.source_fs_dev {
        bail!(
            "{} changed filesystem device between preflight and migration",
            fs_path
        );
    }
    let fs_dev = source.source_fs_dev;

    let dev_path = source.raw_device_path.to_string_lossy().into_owned();
    // Set up DevOpts and open the device
    let mut c_dev = DevOpts::new(CString::new(dev_path.as_str())?);
    c_dev.open_no_blkid(
        crate::wrappers::bdev::BLK_OPEN_READ | crate::wrappers::bdev::BLK_OPEN_WRITE | crate::wrappers::bdev::BLK_OPEN_BUFFERED,
    ).map_err(|e| {
        anyhow!("Error opening device to format {}: {}", dev_path, io::Error::from_raw_os_error(e))
    })?;

    let file_path = format!("{}/bcachefs", fs_path);
    println!("Creating new filesystem on {} in space reserved at {}", dev_path, file_path);

    let dev_size = crate::wrappers::bdev::get_size(c_dev.fd());
    c_dev.fs_size = dev_size;

    let block_size = crate::commands::format_util::pick_block_size(&fs_opts, std::slice::from_ref(&c_dev));
    opt_set!(fs_opts, block_size, block_size as u16);

    let bucket_size = crate::commands::format_util::pick_bucket_size(&fs_opts, std::slice::from_ref(&c_dev));
    {
        let dev_opts = &mut c_dev.opts;
        opt_set!(dev_opts, bucket_size, bucket_size as u32);
    }
    c_dev.nbuckets = c_dev.fs_size / c_dev.opts.bucket_size as u64;

    crate::commands::format_util::check_bucket_size(&fs_opts, &c_dev);

    // Reserve space for bcachefs metadata — grab as much as we can
    let (extents, bcachefs_inum) = reserve_new_fs_space(
        &file_path,
        block_size >> 9,
        dev_size,
        fs_dev,
        force,
        &source.mapper,
    )?;

    // Find superblock location within reserved extents
    let (sb_offset, sb_end) = find_superblock_space(
        &extents,
        format_opts.superblock_size,
        c_dev.opts.bucket_size,
    )?;
    c_dev.sb_offset = sb_offset;
    c_dev.sb_end = sb_end;

    // format() consumes format_opts (it's no longer Copy — has a blocklisted
    // bch_opts field), so capture the fields used afterward first.
    let passphrase = format_opts.passphrase;
    let superblock_size = format_opts.superblock_size;

    let sb = crate::commands::format_util::format(fs_opt_strs, fs_opts, format_opts, std::slice::from_mut(&mut c_dev));
    if sb.is_null() {
        bail!("format failed");
    }

    let sb_offset_val = u64::from_le(unsafe { (*sb).layout.sb_offset[0] });

    // Add encryption key if needed
    if !passphrase.is_null() {
        let type_cstr = CString::new("user").unwrap();
        let desc_cstr = CString::new("user").unwrap();
        unsafe { c::bch2_add_key(sb, type_cstr.as_ptr(), desc_cstr.as_ptr(), passphrase) };
    }

    unsafe { libc::free(sb as *mut _) };

    // Open the new filesystem
    let dev_path_pb = std::path::PathBuf::from(&dev_path);
    let mut opts: c::bch_opts = Default::default();
    opt_set!(opts, sb, sb_offset_val);
    opt_set!(opts, nostart, true as u8);
    opt_set!(opts, noexcl, true as u8);
    opt_set!(opts, reconcile_enabled, false as u8);
    opt_set!(opts, copygc_enabled, false as u8);

    let fs = Fs::open(std::slice::from_ref(&dev_path_pb), opts)
        .map_err(|e| anyhow!("Error opening new filesystem: {}", e))?;

    fs.buckets_nouse_alloc()
        .map_err(|e| anyhow!("Error allocating buckets_nouse: {}", e))?;

    unsafe { mark_unreserved_space(fs.raw, &extents) };

    fs.start()
        .map_err(|e| anyhow!("Error starting new filesystem: {}", e))?;

    // Set no_default_sb feature — prevents rw mount until migrate-superblock is run
    let sb = unsafe { &mut *(*fs.raw).disk_sb.sb };
    sb.features[0] |= (1u64 << c::bch_sb_feature::BCH_FEATURE_no_default_sb as u64).to_le();
    {
        let _lock = fs.sb_lock();
        fs.write_super_ret()
            .map_err(|e| anyhow!("Error writing superblock: {}", e))?;
    }

    // Calculate reserve_start: round up (superblock_size * 2 + BCH_SB_SECTOR) sectors
    // to bucket boundary (in bytes)
    let bucket_bytes = unsafe { (*(*fs.raw).devs[0]).mi.bucket_size as u64 } << 9;
    let reserve_start = (((superblock_size as u64 * 2 + c::BCH_SB_SECTOR as u64) << 9)
        .div_ceil(bucket_bytes)) * bucket_bytes;

    // Copy the filesystem tree
    let mut s = crate::copy_fs::CopyFsState::new_migrate(
        bcachefs_inum,
        fs_dev,
        reserve_start,
        source.mapper.clone(),
    );
    // Pre-populate with metadata reservation ranges
    for e in &extents {
        s.extents.push((e.start, e.end));
    }

    let copy_ret = crate::copy_fs::copy_fs(&fs, &mut s, fs_file.as_fd(), &fs_path_cstr);

    let exit_ret = fs.exit();
    drop(fs_file);

    copy_ret.map_err(|e| anyhow!("Error copying filesystem: {}", e))?;
    if exit_ret != 0 {
        bail!("Error closing filesystem (ret={})", exit_ret);
    }

    // Run fsck on the new filesystem
    println!("Migrate complete, running fsck:");
    let mut fsck_opts: c::bch_opts = Default::default();
    opt_set!(fsck_opts, sb, sb_offset_val);
    opt_set!(fsck_opts, noexcl, true as u8);
    opt_set!(fsck_opts, nochanges, true as u8);
    opt_set!(fsck_opts, read_only, true as u8);

    let fs = Fs::open(std::slice::from_ref(&dev_path_pb), fsck_opts)
        .map_err(|e| anyhow!("Error opening new filesystem for fsck: {}", e))?;
    drop(fs);
    println!("fsck complete");

    println!(
        "To mount the new filesystem, run\n\
         \x20 mount -t bcachefs -o sb={sb} {dev} dir\n\
         \n\
         After verifying that the new filesystem is correct, to create a\n\
         superblock at the default offset and finish the migration run\n\
         \x20 bcachefs migrate-superblock -d {dev} -o {sb}\n\
         \n\
         The new filesystem will have a file at /old_migrated_filesystem\n\
         referencing all disk space that might be used by the existing\n\
         filesystem. That file can be deleted once the old filesystem is\n\
         no longer needed (and should be deleted prior to running\n\
         bcachefs migrate-superblock)",
        sb = sb_offset_val,
        dev = dev_path,
    );

    Ok(())
}

fn migrate_superblock(dev_path: &str, sb_offset: u64) -> Result<()> {
    use std::os::unix::fs::OpenOptionsExt;

    // Open device and read existing superblock
    let dev_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_EXCL)
        .open(dev_path)
        .map_err(|e| anyhow!("Error opening {}: {}", dev_path, e))?;

    let mut sb = super_io::super_read(dev_file.as_raw_fd(), sb_offset)?;

    // Validate and add default layout (catches errors early)
    let sb_size = add_default_sb_layout(sb.sb_mut())?;

    // Zero the start of the disk to blow away the old superblock
    let zeroes_len = ((c::BCH_SB_SECTOR as usize) << 9) + std::mem::size_of::<c::bch_sb>();
    let zeroes = vec![0u8; zeroes_len];
    {
        use std::os::unix::fs::FileExt;
        dev_file.write_all_at(&zeroes, 0)
            .map_err(|e| anyhow!("Error zeroing start of disk: {}", e))?;
    }
    drop(dev_file);
    drop(sb);

    // Open the filesystem with nostart + sb offset
    let dev_path_pb = std::path::PathBuf::from(dev_path);
    let mut opts: c::bch_opts = Default::default();
    opt_set!(opts, nostart, true as u8);
    opt_set!(opts, sb, sb_offset);
    opt_set!(opts, reconcile_enabled, false as u8);
    opt_set!(opts, copygc_enabled, false as u8);

    let fs = Fs::open(&[dev_path_pb], opts)
        .map_err(|e| anyhow!("Error opening filesystem: {}", e))?;

    fs.buckets_nouse_alloc()
        .map_err(|e| anyhow!("Error allocating buckets_nouse: {}", e))?;

    // Mark the new sb bucket range as nouse
    let ca = unsafe { (*fs.raw).devs[0] };
    unsafe { mark_nouse_range(ca, 0, c::BCH_SB_SECTOR as u64 + sb_size as u64 * 2) };

    // Clear no_default_sb feature so recovery doesn't force ro
    unsafe {
        let fs_raw = fs.raw;
        let _lock = fs.sb_lock();
        let sb = &mut *(*fs_raw).disk_sb.sb;
        sb.features[0] &= !(1u64 << c::bch_sb_feature::BCH_FEATURE_no_default_sb as u64).to_le();
        // Also update the cached copy — c->sb.features is host-endian
        (*fs_raw).sb.features &= !(1u64 << c::bch_sb_feature::BCH_FEATURE_no_default_sb as u64);
        fs.write_super_ret()
            .map_err(|e| anyhow!("Error writing superblock: {}", e))?;

        // This is normall set in fs_init if BCH_FEATURE_no_default_sb is
        // cleared - but fs_init already happened:
        (*fs_raw).flags |= (1 as c_ulong) << c::bch_fs_flags::BCH_FS_may_upgrade_downgrade as u32;
    }

    fs.start()
        .map_err(|e| anyhow!("Error starting filesystem: {}", e))?;

    // Verify sb_size consistency
    let actual_sb_size = 1u32 << unsafe { (*(*ca).disk_sb.sb).layout.sb_max_size_bits };
    assert_eq!(actual_sb_size, sb_size, "sb_max_size_bits mismatch after recovery");

    // Apply superblock layout changes (FS is already RW)
    let sb_ref = unsafe { &mut *(*ca).disk_sb.sb };
    add_default_sb_layout(sb_ref)?;

    // Mark the new sb buckets in FS metadata
    let ca_ref = fs.dev_get(0)
        .ok_or_else(|| anyhow!("device 0 not found"))?;
    fs.trans_mark_dev_sb(
        &ca_ref,
        bcachefs_kernel::btree::iter::UpdateTriggerFlags::TRANSACTIONAL,
    ).map_err(|e| anyhow!("Error marking superblock buckets: {}", e))?;
    drop(ca_ref);

    drop(fs);
    Ok(())
}

// ---- Public command entry points ----

fn cmd_migrate(argv: Vec<String>) -> Result<()> {
    let opt_flags = c::opt_flags::OPT_FORMAT as u32;

    let mut fs_path: Option<String> = None;
    let mut encrypted = false;
    let mut no_passphrase = false;
    let mut force = false;
    let mut no_change = false;

    let mut fs_opts: c::bch_opts = Default::default();
    let mut deferred_opts: Vec<(c::bch_opt_id, String)> = Vec::new();

    let mut i = 1;
    while i < argv.len() {
        let arg = &argv[i];

        if arg.starts_with("--") && arg.len() > 2 {
            let opt_part = &arg[2..];
            let (raw_name, inline_val) = match opt_part.split_once('=') {
                Some((n, v)) => (n, Some(v)),
                None => (opt_part, None),
            };
            let name = raw_name.replace('-', "_");

            if let Some((opt_id, opt, negated)) = bch_opt_lookup_negated(&name) {
                if opt.flags as u32 & opt_flags != 0 {
                    let val_str = if negated {
                        "0".to_string()
                    } else if let Some(v) = inline_val {
                        v.to_string()
                    } else if opt.type_ != c::opt_type::BCH_OPT_BOOL {
                        take_opt_value(None, &argv, &mut i, raw_name)?
                    } else {
                        "1".to_string()
                    };

                    match parse_opt_val(opt, &val_str)? {
                        None => deferred_opts.push((opt_id, val_str)),
                        Some(v) => bcachefs_kernel::opts::opt_set_by_id(&mut fs_opts, opt_id, v),
                    }
                    i += 1;
                    continue;
                }
            }

            match name.as_str() {
                "encrypted" => encrypted = true,
                "no_passphrase" => no_passphrase = true,
                "no_change" => no_change = true,
                "help" => {
                    migrate_usage();
                    return Ok(());
                }
                _ => bail!("unknown option: {}", arg),
            }

            i += 1;
            continue;
        }

        if arg.starts_with('-') && arg.len() > 1 {
            match arg.as_bytes()[1] {
                b'f' => {
                    i += 1;
                    if i >= argv.len() {
                        bail!("-f requires a value");
                    }
                    fs_path = Some(argv[i].clone());
                }
                b'F' => force = true,
                b'n' => no_change = true,
                b'h' => {
                    migrate_usage();
                    return Ok(());
                }
                _ => bail!("unknown option: {}", arg),
            }

            i += 1;
            continue;
        }

        if fs_path.is_some() {
            bail!("filesystem specified more than once");
        }
        fs_path = Some(arg.clone());
        i += 1;
        continue;
    }

    let fs_path = fs_path.ok_or_else(|| {
        migrate_usage();
        anyhow!("please specify a filesystem to migrate")
    })?;
    let source_path = Path::new(&fs_path);
    let source = match inspect_source(source_path) {
        Ok(source) => source,
        Err(error) if no_change => {
            println!("MIGRATE_RESULT eligible=no");
            return Err(error);
        }
        Err(error) => return Err(error),
    };
    print_preflight(source_path, &source);
    if no_change {
        return run_no_change(source_path, &source);
    }

    // Build format_opts with version detection
    let mut fmt_opts = format_opts_default();
    fmt_opts.encrypted = encrypted;

    // Handle encryption passphrase
    let passphrase: Option<Passphrase> = if encrypted && !no_passphrase {
        Some(Passphrase::ask_for_new_passphrase()?)
    } else {
        None
    };

    if let Some(ref p) = passphrase {
        fmt_opts.passphrase = p.get().as_ptr() as *mut c_char;
    }

    // Build bch_opt_strs for deferred options
    let mut fs_opt_strs: c::bch_opt_strs = Default::default();
    for &(id, ref val) in &deferred_opts {
        let cstr = CString::new(val.as_str())?;
        fs_opt_strs.set(id, &cstr);
    }

    let result = migrate_fs(&fs_path, source, fs_opt_strs, fs_opts, fmt_opts, force);

    fs_opt_strs.free();

    result
}

/// Migrate superblock to standard location
#[derive(Parser, Debug)]
#[command(about = "Create default superblock after migrating")]
pub struct MigrateSuperblockCli {
    /// Device to create superblock for
    #[arg(short = 'd', long = "dev")]
    device: String,

    /// Offset of existing superblock
    #[arg(short = 'o', long = "offset")]
    offset: u64,
}

fn cmd_migrate_superblock(cli: MigrateSuperblockCli) -> Result<()> {
    migrate_superblock(&cli.device, cli.offset)
}

pub const CMD_MIGRATE: super::CmdDef = raw_cmd!("migrate", "Migrate existing filesystem to bcachefs", cmd_migrate);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_change_classifies_every_copy_fallback_flag() {
        let cases = [
            (FiemapExtentFlags::SHARED, NoChangeCopyReason::Shared),
            (FiemapExtentFlags::ENCODED, NoChangeCopyReason::Encoded),
            (FiemapExtentFlags::DATA_ENCRYPTED, NoChangeCopyReason::Encrypted),
            (FiemapExtentFlags::NOT_ALIGNED, NoChangeCopyReason::NotAligned),
            (FiemapExtentFlags::DATA_INLINE, NoChangeCopyReason::Inline),
            (FiemapExtentFlags::DATA_TAIL, NoChangeCopyReason::Tail),
            (FiemapExtentFlags::MERGED, NoChangeCopyReason::Merged),
            (
                FiemapExtentFlags::from_bits_retain(1 << 15),
                NoChangeCopyReason::FutureFlags,
            ),
        ];
        for (flags, expected) in cases {
            assert_eq!(no_change_copy_reason(flags), Some(expected));
        }
        assert_eq!(no_change_copy_reason(FiemapExtentFlags::empty()), None);
        assert_eq!(
            no_change_copy_reason(FiemapExtentFlags::UNKNOWN | FiemapExtentFlags::DELALLOC),
            None
        );
    }

    #[test]
    fn no_change_range_holes_cover_only_unmapped_bytes() {
        let mut ranges = vec![
            CRange { start: 100, end: 200 },
            CRange { start: 0, end: 50 },
            CRange { start: 40, end: 120 },
        ];
        ranges_sort_merge(&mut ranges);
        assert_eq!(ranges, vec![CRange { start: 0, end: 200 }]);
        assert_eq!(
            for_each_hole(&ranges, 300),
            vec![CRange { start: 200, end: 300 }],
        );
    }
}
pub const CMD_MIGRATE_SUPERBLOCK: super::CmdDef = typed_cmd!("migrate-superblock", "Move superblock to standard location", MigrateSuperblockCli, cmd_migrate_superblock);
