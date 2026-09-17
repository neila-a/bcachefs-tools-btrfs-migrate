// SPDX-License-Identifier: GPL-2.0
//
// Copy a POSIX directory tree into a bcachefs filesystem.
//
// Two modes:
//   - Copy (format --source): data is written to the new filesystem
//   - Migrate (bcachefs migrate): data extents point at existing on-disk locations
//
// Converted from c_src/posix_to_bcachefs.c.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::time::{Duration, Instant};

use bch_bindgen::fs::FsExt;
use bch_bindgen::c;
use bch_bindgen::data::io::{block_on, MAX_IO_SIZE};
use bcachefs_kernel::btree;
use bcachefs_kernel::btree::iter::{CommitFlags, CommitOpts};
use bcachefs_kernel::data::io_misc;
use bcachefs_kernel::{dirent, inode, namei, str_hash, xattr};
use bcachefs_kernel::errcode::{ret_to_result_void as ret_to_result, BchError, bch_errcode};
use bcachefs_kernel::fs::Fs;

use crate::util::AlignedBuf;
use crate::btrfs_migrate::{ExtentMapper, PhysicalSegment};
use crate::migration_copy_policy::{
    classify_extent, plan_raw_segment, CopyAccounting, CopyReason, ExtentDisposition,
    MigrationCopyPolicy, RawSegmentDisposition,
};

const BCACHEFS_ROOT_INO: u64 = 4096;
const BCACHEFS_ROOT_SUBVOL: u64 = 1;

/// DT_* constants (from dirent.h)
const DT_DIR: u8  = 4;
const DT_REG: u8  = 8;
const DT_LNK: u8  = 10;

const PROGRESS_INTERVAL: Duration = Duration::from_secs(30);
const PROGRESS_ENTRY_INTERVAL: u64 = 10_000;

/// Xattr namespace indices matching KEY_TYPE_XATTR_INDEX_* in xattr_format.h.
const XATTR_INDEX_USER: i32     = 0;
const XATTR_INDEX_ACL_ACCESS: i32 = 1;
const XATTR_INDEX_ACL_DEFAULT: i32 = 2;
const XATTR_INDEX_TRUSTED: i32  = 3;
const XATTR_INDEX_SECURITY: i32 = 4;

const POSIX_ACL_XATTR_VERSION: u32 = 0x0002;
const BCH_ACL_VERSION: u32 = 0x0001;

const ACL_USER_OBJ: u16 = 0x01;
const ACL_USER: u16 = 0x02;
const ACL_GROUP_OBJ: u16 = 0x04;
const ACL_GROUP: u16 = 0x08;
const ACL_MASK: u16 = 0x10;
const ACL_OTHER: u16 = 0x20;
const ACL_UNDEFINED_ID: u32 = 0xffff_ffff;

const POSIX_ACL_XATTR_HEADER_SIZE: usize = 4;
const POSIX_ACL_XATTR_ENTRY_SIZE: usize = 8;

fn read_acl_bytes<const N: usize>(
    value: &[u8],
    offset: usize,
    entry_index: usize,
) -> anyhow::Result<[u8; N]> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| anyhow::anyhow!("POSIX ACL entry {} offset overflows", entry_index))?;
    let bytes = value.get(offset..end).ok_or_else(|| {
        anyhow::anyhow!(
            "POSIX ACL entry {} is truncated at byte {}",
            entry_index,
            offset
        )
    })?;
    Ok(bytes.try_into().unwrap())
}

fn linux_posix_acl_xattr_to_bcachefs(value: &[u8]) -> anyhow::Result<Vec<u8>> {
    if value.len() < POSIX_ACL_XATTR_HEADER_SIZE {
        anyhow::bail!(
            "Linux POSIX ACL xattr is shorter than its {}-byte header ({} bytes)",
            POSIX_ACL_XATTR_HEADER_SIZE,
            value.len()
        );
    }

    let version = u32::from_le_bytes(read_acl_bytes::<4>(value, 0, 0)?);
    if version != POSIX_ACL_XATTR_VERSION {
        anyhow::bail!(
            "unsupported Linux POSIX ACL xattr version {}",
            version
        );
    }

    let payload_len = value
        .len()
        .checked_sub(POSIX_ACL_XATTR_HEADER_SIZE)
        .ok_or_else(|| anyhow::anyhow!("POSIX ACL xattr header length overflows"))?;
    if payload_len % POSIX_ACL_XATTR_ENTRY_SIZE != 0 {
        anyhow::bail!(
            "Linux POSIX ACL xattr payload is {} bytes, not divisible by {}",
            payload_len,
            POSIX_ACL_XATTR_ENTRY_SIZE
        );
    }

    let entry_count = payload_len / POSIX_ACL_XATTR_ENTRY_SIZE;
    let output_capacity = POSIX_ACL_XATTR_HEADER_SIZE
        .checked_add(
            entry_count
                .checked_mul(POSIX_ACL_XATTR_ENTRY_SIZE)
                .ok_or_else(|| anyhow::anyhow!("POSIX ACL output size overflows"))?,
        )
        .ok_or_else(|| anyhow::anyhow!("POSIX ACL output size overflows"))?;
    let mut output = Vec::with_capacity(output_capacity);
    output.extend_from_slice(&BCH_ACL_VERSION.to_le_bytes());

    let mut offset = POSIX_ACL_XATTR_HEADER_SIZE;
    for entry_index in 0..entry_count {
        let tag = u16::from_le_bytes(read_acl_bytes::<2>(value, offset, entry_index)?);
        let perm = u16::from_le_bytes(read_acl_bytes::<2>(
            value,
            offset
                .checked_add(2)
                .ok_or_else(|| anyhow::anyhow!("POSIX ACL entry {} offset overflows", entry_index))?,
            entry_index,
        )?);
        let id = u32::from_le_bytes(read_acl_bytes::<4>(
            value,
            offset
                .checked_add(4)
                .ok_or_else(|| anyhow::anyhow!("POSIX ACL entry {} offset overflows", entry_index))?,
            entry_index,
        )?);

        if perm & !0x7 != 0 {
            anyhow::bail!(
                "Linux POSIX ACL entry {} has invalid permissions 0x{:x}",
                entry_index,
                perm
            );
        }

        let short_entry = match tag {
            ACL_USER_OBJ | ACL_GROUP_OBJ | ACL_MASK | ACL_OTHER => true,
            ACL_USER | ACL_GROUP => false,
            _ => anyhow::bail!(
                "Linux POSIX ACL entry {} has unknown tag 0x{:x}",
                entry_index,
                tag
            ),
        };

        if short_entry {
            if id != ACL_UNDEFINED_ID {
                anyhow::bail!(
                    "Linux POSIX ACL entry {} has concrete id {} for short tag 0x{:x}",
                    entry_index,
                    id,
                    tag
                );
            }
        } else if id == ACL_UNDEFINED_ID {
            anyhow::bail!(
                "Linux POSIX ACL entry {} has undefined id for named tag 0x{:x}",
                entry_index,
                tag
            );
        }

        output.extend_from_slice(&tag.to_le_bytes());
        output.extend_from_slice(&perm.to_le_bytes());
        if !short_entry {
            output.extend_from_slice(&id.to_le_bytes());
        }

        offset = offset
            .checked_add(POSIX_ACL_XATTR_ENTRY_SIZE)
            .ok_or_else(|| anyhow::anyhow!("POSIX ACL entry offset overflows"))?;
    }

    Ok(output)
}

fn xattr_value_for_storage<'a>(
    xattr_type: i32,
    value: &'a [u8],
) -> anyhow::Result<Cow<'a, [u8]>> {
    match xattr_type {
        XATTR_INDEX_ACL_ACCESS | XATTR_INDEX_ACL_DEFAULT => {
            linux_posix_acl_xattr_to_bcachefs(value).map(Cow::Owned)
        }
        _ => Ok(Cow::Borrowed(value)),
    }
}

fn acl_type_name(xattr_type: i32) -> Option<&'static str> {
    match xattr_type {
        XATTR_INDEX_ACL_ACCESS => Some("access"),
        XATTR_INDEX_ACL_DEFAULT => Some("default"),
        _ => None,
    }
}

fn acl_inode_flag(xattr_type: i32) -> Option<u32> {
    match xattr_type {
        XATTR_INDEX_ACL_ACCESS => {
            Some(c::bch_inode_flags::BCH_INODE_has_access_acl as u32)
        }
        XATTR_INDEX_ACL_DEFAULT => {
            Some(c::bch_inode_flags::BCH_INODE_has_default_acl as u32)
        }
        _ => None,
    }
}

fn add_acl_inode_flag(inode_flags: u32, xattr_type: i32) -> u32 {
    inode_flags | acl_inode_flag(xattr_type).unwrap_or(0)
}

fn mode_to_type(mode: u32) -> u8 {
    ((mode >> 12) & 15) as u8
}

/// Convert a rustix error into a BchError.
fn rustix_err(e: rustix::io::Errno) -> BchError {
    BchError::from_raw(-e.raw_os_error())
}

/// Convert a rustix FileType to a DT_* constant.
fn file_type_to_dtype(ft: rustix::fs::FileType) -> u8 {
    use rustix::fs::FileType;
    match ft {
        FileType::RegularFile     => DT_REG,
        FileType::Directory       => DT_DIR,
        FileType::Symlink         => DT_LNK,
        _                         => 0, // DT_UNKNOWN
    }
}

extern "C" {
    fn rust_link_data(
        c: *mut c::bch_fs,
        dst_inum: u64,
        sectors_delta: *mut i64,
        logical: u64,
        physical: u64,
        length: u64,
    ) -> i32;
}

/// Migrate type matching enum bch_migrate_type in posix_to_bcachefs.h.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MigrateType {
    Copy,
    Migrate,
}


/// State for a copy_fs operation.
pub struct CopyFsState {
    pub migrate_type:   MigrateType,
    pub bcachefs_inum:  u64,
    pub dev:            u64,
    pub reserve_start:  u64,
    pub mapper:          ExtentMapper,
    pub extents:        Vec<(u64, u64)>,  // (start, end) byte ranges
    pub verbosity:      u32,

    pub total_files:    u64,
    pub total_input:    u64,
    pub total_wrote:    u64,
    total_allocated_wrote: u64,
    pub total_linked:   u64,
    raw_link_salvaged_bytes: u64,
    pub total_unwritten: u64,
    pub total_unstable: u64,
    pub mapped_extents: u64,
    pub mapped_segments: u64,
    pub chunk_splits: u64,
    reservation_usable_full_bucket_bytes: u64,
    reservation_allocator_headroom_bytes: u64,
    copy_accounting: CopyAccounting,
    entries_total: u64,
    entries_done: u64,
    // All successfully copied non-directory dirents: regular files, symlinks,
    // and special files.
    non_directories_done: u64,
    directories_done: u64,
    progress_started: Instant,
    progress_last_emitted: Instant,
    progress_last_entries: u64,
    cancellation_probe: Option<fn() -> bool>,

    /// Hardlink tracking: source inode -> destination inode
    hardlinks: HashMap<u64, u64>,
}

impl CopyFsState {
    pub fn new_copy() -> Self {
        Self {
            migrate_type:   MigrateType::Copy,
            bcachefs_inum:  0,
            dev:            0,
            reserve_start:  0,
            mapper:         ExtentMapper::Identity,
            extents:        Vec::new(),
            verbosity:      0,
            total_files:    0,
            total_input:    0,
            total_wrote:    0,
            total_allocated_wrote: 0,
            total_linked:   0,
            raw_link_salvaged_bytes: 0,
            total_unwritten: 0,
            total_unstable:   0,
            mapped_extents:   0,
            mapped_segments:  0,
            chunk_splits:     0,
            reservation_usable_full_bucket_bytes: 0,
            reservation_allocator_headroom_bytes: 0,
            copy_accounting: CopyAccounting::default(),
            entries_total: 0,
            entries_done: 0,
            non_directories_done: 0,
            directories_done: 0,
            progress_started: Instant::now(),
            progress_last_emitted: Instant::now(),
            progress_last_entries: 0,
            hardlinks:      HashMap::new(),
            cancellation_probe: None,
        }
    }

    pub fn new_migrate(
        bcachefs_inum: u64,
        dev: u64,
        reserve_start: u64,
        mapper: ExtentMapper,
        reservation_usable_full_bucket_bytes: u64,
        reservation_allocator_headroom_bytes: u64,
    ) -> Self {
        Self {
            migrate_type:   MigrateType::Migrate,
            bcachefs_inum,
            dev,
            reserve_start,
            extents:        Vec::new(),
            mapper,
            verbosity:      0,
            total_files:    0,
            total_input:    0,
            total_wrote:    0,
            total_allocated_wrote: 0,
            total_linked:   0,
            raw_link_salvaged_bytes: 0,
            total_unwritten: 0,
            total_unstable:   0,
            mapped_extents:   0,
            mapped_segments:  0,
            chunk_splits:     0,
            reservation_usable_full_bucket_bytes,
            reservation_allocator_headroom_bytes,
            copy_accounting: CopyAccounting::default(),
            entries_total: 0,
            entries_done: 0,
            non_directories_done: 0,
            directories_done: 0,
            progress_started: Instant::now(),
            progress_last_emitted: Instant::now(),
            progress_last_entries: 0,
            hardlinks:      HashMap::new(),
            cancellation_probe: None,
        }
    }

    pub fn set_cancellation_probe(&mut self, probe: fn() -> bool) {
        self.cancellation_probe = Some(probe);
    }

    fn check_cancellation(&self) -> Result<(), BchError> {
        check_cancellation_probe(self.cancellation_probe)
    }
}

fn check_cancellation_probe(probe: Option<fn() -> bool>) -> Result<(), BchError> {
    if probe.is_some_and(|probe| probe()) {
        eprintln!("MIGRATE_CANCELLATION requested=true");
        return Err(BchError::from_raw(-libc::ECANCELED));
    }
    Ok(())
}

impl CopyFsState {
    fn record_copy(&mut self, reason: CopyReason, bytes: u64, allocated_bytes: u64) {
        self.copy_accounting
            .record_with_allocation(reason, bytes, allocated_bytes);
        self.total_wrote = self.copy_accounting.copied_bytes;
        self.total_allocated_wrote = self.copy_accounting.allocated_copy_bytes;
    }

    fn ensure_reservation_capacity_for_copy(&self, allocated_bytes: u64) -> Result<(), BchError> {
        if self.migrate_type != MigrateType::Migrate {
            return Ok(());
        }

        let padded_copy_bytes = self
            .total_allocated_wrote
            .checked_add(allocated_bytes)
            .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
        let required_reservation_bytes = padded_copy_bytes
            .checked_add(self.reservation_allocator_headroom_bytes)
            .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
        if required_reservation_bytes >= self.reservation_usable_full_bucket_bytes {
            eprintln!(
                "migration copy requires {} bytes ({} padded copy bytes + {} allocator headroom bytes), but the reservation exposes only {} usable full-bucket bytes",
                required_reservation_bytes,
                padded_copy_bytes,
                self.reservation_allocator_headroom_bytes,
                self.reservation_usable_full_bucket_bytes,
            );
            return Err(BchError::from_raw(-libc::ENOSPC));
        }

        Ok(())
    }

    fn start_progress(&mut self, fs: &Fs, entries_total: u64) {
        self.entries_total = entries_total;
        self.emit_progress(fs, true);
    }

    fn finish_entry(&mut self, fs: &Fs, is_directory: bool) -> Result<(), BchError> {
        self.entries_done = self
            .entries_done
            .checked_add(1)
            .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
        if is_directory {
            self.directories_done = self
                .directories_done
                .checked_add(1)
                .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
        } else {
            self.non_directories_done = self
                .non_directories_done
                .checked_add(1)
                .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
        }

        self.emit_progress(fs, false);
        Ok(())
    }

    fn finish_progress(&mut self, fs: &Fs) {
        self.emit_progress(fs, true);
    }

    fn should_emit_progress(&self) -> bool {
        self.entries_done - self.progress_last_entries >= PROGRESS_ENTRY_INTERVAL
            || self.progress_last_emitted.elapsed() >= PROGRESS_INTERVAL
    }

    fn emit_progress(&mut self, _fs: &Fs, force: bool) {
        if !force && !self.should_emit_progress() {
            return;
        }

        let required_reservation_bytes = self
            .total_allocated_wrote
            .saturating_add(self.reservation_allocator_headroom_bytes);
        let reservation_remaining_bytes = self
            .reservation_usable_full_bucket_bytes
            .saturating_sub(self.total_allocated_wrote);
        eprintln!(
            "MIGRATE_PROGRESS entries_done={} entries_total={} non_directories_done={} directories_done={} linked_bytes={} copied_bytes={} elapsed_seconds={} logical_copy_bytes={} padded_copy_bytes={} allocator_headroom_bytes={} required_reservation_bytes={} reservation_usable_full_bucket_bytes={} reservation_remaining_bytes={}",
            self.entries_done,
            self.entries_total,
            self.non_directories_done,
            self.directories_done,
            self.total_linked,
            self.total_wrote,
            self.progress_started.elapsed().as_secs(),
            self.total_wrote,
            self.total_allocated_wrote,
            self.reservation_allocator_headroom_bytes,
            required_reservation_bytes,
            self.reservation_usable_full_bucket_bytes,
            reservation_remaining_bytes,
        );
        self.progress_last_emitted = Instant::now();
        self.progress_last_entries = self.entries_done;
    }
}

fn root_subvol_inum() -> c::subvol_inum {
    c::subvol_inum { subvol: BCACHEFS_ROOT_SUBVOL, inum: BCACHEFS_ROOT_INO }
}

fn subvol_inum(inum: u64) -> c::subvol_inum {
    c::subvol_inum { subvol: 1, inum }
}

fn subvol_inum_eq(a: c::subvol_inum, b: c::subvol_inum) -> bool {
    a.subvol == b.subvol && a.inum == b.inum
}

fn zeroed_subvol_inum() -> c::subvol_inum {
    c::subvol_inum { subvol: 0, inum: 0 }
}

fn unlink_and_rm(
    fs: &Fs,
    dir_inum: c::subvol_inum,
    dir: &mut c::bch_inode_unpacked,
    child_name: &CStr,
) -> Result<(), BchError> {
    let qstr = dirent::qstr(child_name.to_bytes());
    let mut child: c::bch_inode_unpacked = Default::default();

    btree::iter::trans_commit_do(
        fs,
        None,
        CommitOpts::new().flags(CommitFlags::NO_ENOSPC),
        |t| {
            namei::unlink_trans(
                t,
                dir_inum,
                dir,
                zeroed_subvol_inum(),
                &mut child,
                &qstr,
                false,
            )
        },
    )?;

    if child.bi_flags & (c::bch_inode_flags::BCH_INODE_unlinked as u32) == 0 {
        return Ok(());
    }

    let child_inum = c::subvol_inum { subvol: dir_inum.subvol, inum: child.bi_inum };
    inode::rm(fs, child_inum)
}

fn update_inode(fs: &Fs, inode: &c::bch_inode_unpacked) -> Result<(), BchError> {
    inode::write_cached(fs, inode)
}

fn create_or_update_link(
    fs: &Fs,
    dir_inum: c::subvol_inum,
    dir: &mut c::bch_inode_unpacked,
    name: &CStr,
    inum: c::subvol_inum,
) -> Result<(), BchError> {
    let dir_hash = str_hash::hash_info_init(fs, dir)?;

    let qstr = dirent::qstr(name.to_bytes());

    match dirent::lookup(fs, dir_inum, &dir_hash, &qstr) {
        Err(e) if e.matches(bch_errcode::BCH_ERR_ENOENT_str_hash_lookup) => {
            // Fall through to create
        }
        Err(e) => return Err(e),
        Ok(old_inum) => {
            if subvol_inum_eq(inum, old_inum) {
                return Ok(());
            }
            unlink_and_rm(fs, dir_inum, dir, name)?;
        }
    }

    let mut dir_u: c::bch_inode_unpacked = Default::default();
    let mut inode: c::bch_inode_unpacked = Default::default();

    btree::iter::trans_commit_do(
        fs,
        None,
        CommitOpts::new(),
        |t| namei::link_trans(t, dir_inum, &mut dir_u, inum, &mut inode, &qstr),
    )
}

#[allow(clippy::too_many_arguments)]
fn create_or_update_file(
    fs: &Fs,
    dir_inum: c::subvol_inum,
    dir: &mut c::bch_inode_unpacked,
    name: &CStr,
    uid: u32,
    gid: u32,
    mode: u32,
    rdev: u64,
) -> Result<c::bch_inode_unpacked, BchError> {
    let dir_hash = str_hash::hash_info_init(fs, dir)?;

    let qname = dirent::qstr(name.to_bytes());
    let child_inum = dirent::lookup(fs, dir_inum, &dir_hash, &qname);

    let mut child_inode: c::bch_inode_unpacked = Default::default();
    let mut child_subvol : c::bch_subvolume = Default::default();

    if let Ok(child_inum) = child_inum {
        // Already exists — update
        child_inode = inode::find_by_inum(fs, child_inum)?;

        child_inode.bi_mode = mode as u16;
        child_inode.bi_uid = uid;
        child_inode.bi_gid = gid;
        child_inode.bi_dev = rdev as u32;

        // bch2_fsck_write_inode has its own commit_do loop internally —
        // don't wrap it in trans_commit_do or it double-nests.
        {
            let trans = btree::BtreeTrans::new(fs);
            inode::fsck_write_inode(&trans, &mut child_inode)?;
        }
    } else {
        inode::init_early(fs, &mut child_inode);

        btree::iter::trans_commit_do(
            fs,
            None,
            CommitOpts::new(),
            |t| {
                namei::create_trans(
                    t,
                    dir_inum,
                    dir,
                    &mut child_inode,
                    &mut child_subvol,
                    &qname,
                    uid,
                    gid,
                    mode as _,
                    rdev as _,
                    zeroed_subvol_inum(),
                    0,
                )
            },
        )?;
    }

    Ok(child_inode)
}

/// Resolve xattr name prefix to bcachefs xattr index.
/// Returns (index, stripped_name) or None if unsupported.
fn xattr_resolve_name(name: &[u8]) -> Option<(i32, &[u8])> {
    if let Some(rest) = name.strip_prefix(b"user.") {
        Some((XATTR_INDEX_USER, rest))
    } else if let Some(rest) = name.strip_prefix(b"trusted.") {
        Some((XATTR_INDEX_TRUSTED, rest))
    } else if let Some(rest) = name.strip_prefix(b"security.") {
        Some((XATTR_INDEX_SECURITY, rest))
    } else if name == b"system.posix_acl_access" {
        Some((XATTR_INDEX_ACL_ACCESS, b""))
    } else if name == b"system.posix_acl_default" {
        Some((XATTR_INDEX_ACL_DEFAULT, b""))
    } else {
        None
    }
}

fn copy_times(fs: &Fs, dst: &mut c::bch_inode_unpacked, src: &rustix::fs::Stat) {
    let to_bch_time = |sec: i64, nsec: i64| {
        let ts = c::timespec64 {
            tv_sec: sec as _,
            tv_nsec: nsec as _,
            ..unsafe { std::mem::zeroed() }
        };
        fs.timespec_to_time(ts) as u64
    };

    dst.bi_atime = to_bch_time(src.st_atime as i64, src.st_atime_nsec as i64);
    dst.bi_mtime = to_bch_time(src.st_mtime as i64, src.st_mtime_nsec as i64);
    dst.bi_ctime = to_bch_time(src.st_ctime as i64, src.st_ctime_nsec as i64);
}

fn copy_xattrs(
    fs: &Fs,
    dst: &mut c::bch_inode_unpacked,
    src: &CStr,
    display_path: &CStr,
) -> Result<(), BchError> {
    let mut attrs_buf = vec![0u8; 65536]; // XATTR_LIST_MAX
    let attrs_size = match rustix::fs::llistxattr(src, &mut attrs_buf) {
        Ok(n) => n,
        Err(_) => {
            return Ok(()); // silently skip if xattrs not supported
        }
    };

    if attrs_size == 0 {
        return Ok(()); // silently skip if xattrs not supported
    }

    let mut pos = 0usize;
    while pos < attrs_size {
        let end = attrs_buf[pos..attrs_size].iter().position(|&b| b == 0).unwrap() + pos;
        let attr_name = &attrs_buf[pos..end];
        pos = end + 1;

        let (xattr_type, stripped) = match xattr_resolve_name(attr_name) {
            Some(v) => v,
            None => continue,
        };
        let acl_kind = acl_type_name(xattr_type);
        let acl_flag = acl_inode_flag(xattr_type);

        let mut val_buf = vec![0u8; 65536]; // XATTR_SIZE_MAX
        let attr_cstr = CString::new(attr_name).unwrap();
        let val_size = match rustix::fs::lgetxattr(src, &attr_cstr, &mut val_buf) {
            Ok(n) => n,
            Err(error) => {
                if let Some(acl_kind) = acl_kind {
                    eprintln!(
                        "Unable to migrate POSIX {} ACL on {}: unable to read source xattr: {}",
                        acl_kind,
                        display_path.to_string_lossy(),
                        error
                    );
                    return Err(rustix_err(error));
                }
                continue;
            }
        };

        let value = match xattr_value_for_storage(xattr_type, &val_buf[..val_size]) {
            Ok(value) => value,
            Err(error) => {
                if let Some(acl_kind) = acl_kind {
                    eprintln!(
                        "Unable to migrate POSIX {} ACL on {}: {}",
                        acl_kind,
                        display_path.to_string_lossy(),
                        error
                    );
                }
                return Err(BchError::from_raw(-libc::EINVAL));
            }
        };
        let stripped_cstr = CString::new(stripped).unwrap();

        let result = btree::iter::trans_commit_do(
            fs,
            None,
            CommitOpts::new(),
            |t| {
                xattr::set(
                    t,
                    subvol_inum(dst.bi_inum),
                    dst,
                    &stripped_cstr,
                    value.as_ref(),
                    xattr_type,
                    0,
                )
            },
        );
        if let Err(error) = result {
            if let Some(acl_kind) = acl_kind {
                eprintln!(
                    "Unable to migrate POSIX {} ACL on {}: failed to store ACL: {}",
                    acl_kind,
                    display_path.to_string_lossy(),
                    error
                );
            }
            return Err(error);
        }

        if acl_flag.is_some() {
            // xattr::set() commits its transaction before returning.  The
            // userspace copy path has no suitable same-transaction inode
            // write API, so persist the ACL flag immediately instead of
            // relying on recursive copying's final inode update.
            dst.bi_flags = add_acl_inode_flag(dst.bi_flags, xattr_type);
            if let Err(error) = update_inode(fs, dst) {
                if let Some(acl_kind) = acl_kind {
                    eprintln!(
                        "Unable to migrate POSIX {} ACL on {}: failed to persist inode ACL flag: {}",
                        acl_kind,
                        display_path.to_string_lossy(),
                        error
                    );
                }
                return Err(error);
            }
        }
    }

    Ok(())
}

fn write_data(
    fs: &Fs,
    dst_inode: &mut c::bch_inode_unpacked,
    offset: u64,
    data: &[u8],
) -> Result<(), BchError> {
    let result = block_on(
        fs.write(dst_inode.bi_inum, offset, 1, 1, data, u64::MAX)
    )?;
    dst_inode.bi_sectors = (dst_inode.bi_sectors as i64 + result.sectors_delta) as u64;
    Ok(())
}

fn copy_data(
    fs: &Fs,
    s: &mut CopyFsState,
    dst_inode: &mut c::bch_inode_unpacked,
    src_fd: BorrowedFd<'_>,
    start: u64,
    end: u64,
    reason: CopyReason,
) -> Result<(), BchError> {
    let block_size = fs.block_bytes();
    let mut pos = start;

    while pos < end {
        s.check_cancellation()?;
        let len = std::cmp::min(end - pos, MAX_IO_SIZE as u64) as usize;
        let padded = ((len as u64).div_ceil(block_size) * block_size) as usize;

        let mut buf = AlignedBuf::new(padded);
        rustix::io::pread(src_fd, &mut buf[..len], pos)
            .map_err(rustix_err)?;

        s.ensure_reservation_capacity_for_copy(padded as u64)?;
        write_data(fs, dst_inode, pos, &buf)?;
        s.record_copy(reason, len as u64, padded as u64);
        s.emit_progress(fs, false);
        pos += len as u64;
    }

    Ok(())
}

fn link_data(
    fs: &Fs,
    dst: &mut c::bch_inode_unpacked,
    logical: u64,
    physical: u64,
    length: u64,
) -> Result<(), BchError> {
    let mut sectors_delta: i64 = 0;
    let ret = unsafe {
        rust_link_data(fs.raw, dst.bi_inum, &mut sectors_delta, logical, physical, length)
    };
    dst.bi_sectors = (dst.bi_sectors as i64 + sectors_delta) as u64;
    ret_to_result(ret)
}

#[derive(Clone, Copy)]
struct PendingRawLink {
    logical: u64,
    physical: u64,
    len: u64,
}

fn copy_link(
    fs: &Fs,
    s: &mut CopyFsState,
    dst_inum: c::subvol_inum,
    dst: &mut c::bch_inode_unpacked,
    src: &CStr,
) -> Result<(), BchError> {
    let block_size = fs.block_bytes();

    let mut i_sectors_delta: i64 = 0;
    io_misc::fpunch(fs, dst_inum, 0, u64::MAX, &mut i_sectors_delta)?;
    dst.bi_sectors = (dst.bi_sectors as i64 + i_sectors_delta) as u64;

    let target = rustix::fs::readlinkat(rustix::fs::CWD, src, Vec::new())
        .map_err(rustix_err)?;
    let target_bytes = target.as_bytes();
    let link_len = target_bytes.len() as u64;

    let mut buf = AlignedBuf::new(MAX_IO_SIZE);
    buf[..target_bytes.len()].copy_from_slice(target_bytes);
    let padded = (link_len.div_ceil(block_size) * block_size) as usize;

    s.ensure_reservation_capacity_for_copy(padded as u64)?;
    write_data(fs, dst, 0, &buf[..padded])?;
    s.record_copy(CopyReason::Symlink, link_len, padded as u64);
    Ok(())
}

fn file_logical_for_segment(
    extent_file_logical: u64,
    extent_physical: u64,
    segment: &PhysicalSegment,
) -> Result<u64, BchError> {
    let offset = segment
        .source_logical
        .checked_sub(extent_physical)
        .ok_or_else(|| BchError::from_raw(-libc::EINVAL))?;
    extent_file_logical
        .checked_add(offset)
        .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))
}

fn link_file_data(
    fs: &Fs,
    s: &mut CopyFsState,
    dst: &mut c::bch_inode_unpacked,
    src_fd: BorrowedFd<'_>,
    src_path: &CStr,
    src_size: u64,
) -> Result<(), BchError> {
    let block_size = fs.block_bytes();
    let mut extents = collect_fiemap(src_fd)?;

    if extents.iter().any(|extent| {
        extent
            .fe_flags
            .intersects(fiemap::FiemapExtentFlags::UNKNOWN | fiemap::FiemapExtentFlags::DELALLOC)
    }) {
        rustix::fs::fsync(src_fd).map_err(rustix_err)?;
        extents = collect_fiemap(src_fd)?;
    }
    if extents.iter().any(|extent| {
        extent
            .fe_flags
            .intersects(fiemap::FiemapExtentFlags::UNKNOWN | fiemap::FiemapExtentFlags::DELALLOC)
    }) {
        eprintln!(
            "Unstable FIEMAP extent remains in {}; quiesce the source and retry",
            src_path.to_string_lossy()
        );
        return Err(BchError::from_raw(-libc::EBUSY));
    }

    let policy = MigrationCopyPolicy {
        block_bytes: block_size,
        reserved_superblock_end: s.reserve_start,
    };
    let mut pending_links = Vec::new();
    for extent in extents {
        s.check_cancellation()?;
        if extent.fe_length == 0 {
            eprintln!(
                "Zero-length FIEMAP extent in {}",
                src_path.to_string_lossy()
            );
            return Err(BchError::from_raw(-libc::EIO));
        }

        s.total_input = s
            .total_input
            .checked_add(extent.fe_length)
            .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;

        let src_max = src_size.div_ceil(block_size);
        let src_max = src_max
            .checked_mul(block_size)
            .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
        if extent.fe_logical > src_max {
            eprintln!(
                "FIEMAP extent starts past EOF in {}: {} > {}",
                src_path.to_string_lossy(),
                extent.fe_logical,
                src_max
            );
            return Err(BchError::from_raw(-libc::EIO));
        }
        let length = std::cmp::min(extent.fe_length, src_max - extent.fe_logical);
        let visible_len = if extent.fe_logical >= src_size {
            0
        } else {
            std::cmp::min(src_size - extent.fe_logical, length)
        };

        if visible_len == 0 {
            continue;
        }
        let extent_end = extent
            .fe_logical
            .checked_add(visible_len)
            .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
        match classify_extent(extent.fe_flags) {
            ExtentDisposition::Hole => {
                s.total_unwritten = s
                    .total_unwritten
                    .checked_add(visible_len)
                    .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
                continue;
            }
            ExtentDisposition::Copy(reason) => {
                copy_data(
                    fs,
                    s,
                    dst,
                    src_fd,
                    extent.fe_logical,
                    extent_end,
                    reason,
                )?;
                continue;
            }
            ExtentDisposition::Unstable => {
                eprintln!(
                    "Unstable FIEMAP extent remains in {}; quiesce the source and retry",
                    src_path.to_string_lossy()
                );
                return Err(BchError::from_raw(-libc::EBUSY));
            }
            ExtentDisposition::Link => {}
        }

        let segments = match s.mapper.map_physical(extent.fe_physical, length) {
            Ok(segments) => segments,
            Err(error) => {
                eprintln!(
                    "Unable to map FIEMAP logical extent in {}; copying instead: {}",
                    src_path.to_string_lossy(),
                    error
                );
                copy_data(
                    fs,
                    s,
                    dst,
                    src_fd,
                    extent.fe_logical,
                    extent_end,
                    CopyReason::RawMappingUnavailable,
                )?;
                continue;
            }
        };
        if segments.is_empty() {
            continue;
        }
        s.mapped_extents = s
            .mapped_extents
            .checked_add(1)
            .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
        s.mapped_segments = s
            .mapped_segments
            .checked_add(segments.len() as u64)
            .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
        s.chunk_splits = s
            .chunk_splits
            .checked_add((segments.len() as u64).saturating_sub(1))
            .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;

        for segment in segments {
            s.check_cancellation()?;
            let segment_file_logical =
                file_logical_for_segment(extent.fe_logical, extent.fe_physical, &segment)?;
            let segment_file_end = segment_file_logical
                .checked_add(segment.len)
                .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
            let copy_start = std::cmp::max(segment_file_logical, extent.fe_logical);
            let copy_end = std::cmp::min(segment_file_end, extent_end);
            let copy_len = copy_end.saturating_sub(copy_start);
            if copy_len == 0 {
                continue;
            }
            let physical_offset = copy_start
                .checked_sub(segment_file_logical)
                .ok_or_else(|| BchError::from_raw(-libc::EINVAL))?;
            let link_physical = segment
                .physical
                .checked_add(physical_offset)
                .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
            let plan = plan_raw_segment(copy_start, link_physical, copy_len, src_size, policy)
                .map_err(|error| {
                    eprintln!(
                        "Unsafe raw FIEMAP segment in {}: {}",
                        src_path.to_string_lossy(),
                        error,
                    );
                    BchError::from_raw(-libc::EIO)
                })?;
            s.raw_link_salvaged_bytes = s
                .raw_link_salvaged_bytes
                .checked_add(plan.raw_link_salvaged_bytes)
                .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;

            for span in plan.spans() {
                match span.disposition {
                    RawSegmentDisposition::Copy(reason) => {
                        let copy_end = span
                            .file_offset
                            .checked_add(span.len)
                            .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
                        copy_data(fs, s, dst, src_fd, span.file_offset, copy_end, reason)?;
                    }
                    RawSegmentDisposition::Link => pending_links.push(PendingRawLink {
                        logical: span.file_offset,
                        physical: span.physical_offset,
                        len: span.len,
                    }),
                }
            }
        }
    }
    for link in pending_links {
        s.check_cancellation()?;
        range_add(&mut s.extents, link.physical, link.len)?;
        link_data(fs, dst, link.logical, link.physical, link.len)?;
        s.total_linked = s
            .total_linked
            .checked_add(link.len)
            .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
    }

    Ok(())
}

fn collect_fiemap(src_fd: BorrowedFd<'_>) -> Result<Vec<fiemap::FiemapExtent>, BchError> {
    fiemap::Fiemap::new(src_fd)
        .map(|extent| extent.map_err(|_| BchError::from_raw(-libc::EIO)))
        .collect()
}



/// Add a range to the extents list.
fn range_add(extents: &mut Vec<(u64, u64)>, start: u64, len: u64) -> Result<(), BchError> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
    extents.push((start, end));
    Ok(())
}

/// Sort and merge overlapping ranges.
fn ranges_sort_merge(extents: &mut Vec<(u64, u64)>) {
    extents.sort_by_key(|&(start, _)| start);
    let mut i = 0;
    while i + 1 < extents.len() {
        if extents[i].1 >= extents[i + 1].0 {
            extents[i].1 = std::cmp::max(extents[i].1, extents[i + 1].1);
            extents.remove(i + 1);
        } else {
            i += 1;
        }
    }
}

/// Iterate over holes in [0, end) not covered by extents.
fn for_each_hole(
    extents: &[(u64, u64)],
    end: u64,
    mut f: impl FnMut(u64, u64) -> Result<(), BchError>,
) -> Result<(), BchError> {
    let mut pos = 0u64;
    for &(start, stop) in extents {
        if pos < start {
            f(pos, start)?;
        }
        pos = stop;
    }
    if pos < end {
        f(pos, end)?;
    }
    Ok(())
}

/// Range of bytes [start, end).
struct Range {
    start: u64,
    end: u64,
}

fn align_range(r: Range, bs: u64) -> Range {
    Range {
        start: r.start / bs * bs,
        end: r.end.div_ceil(bs) * bs,
    }
}

fn seek_data(fd: BorrowedFd<'_>, i_size: u64, offset: u64) -> Range {
    use rustix::fs::{seek, SeekFrom};
    let s = match seek(fd, SeekFrom::Data(offset)) {
        Ok(s) => s,
        Err(_) => return Range { start: 0, end: 0 },
    };
    let e = seek(fd, SeekFrom::Hole(s)).unwrap_or(i_size);
    Range { start: s, end: e }
}

fn seek_data_aligned(fd: BorrowedFd<'_>, i_size: u64, offset: u64, bs: u64) -> Range {
    let mut r = align_range(seek_data(fd, i_size, offset), bs);
    if r.end == 0 {
        return r;
    }
    loop {
        let n = align_range(seek_data(fd, i_size, r.end), bs);
        if n.end == 0 || r.end < n.start {
            break;
        }
        r.end = n.end;
    }
    r
}

fn seek_mismatch(buf1: &[u8], buf2: &[u8], mut offset: usize, len: usize) -> Range {
    while offset < len && buf1[offset] == buf2[offset] {
        offset += 1;
    }
    if offset == len {
        return Range { start: 0, end: 0 };
    }
    let start = offset;
    while offset < len && buf1[offset] != buf2[offset] {
        offset += 1;
    }
    Range { start: start as u64, end: offset as u64 }
}

fn seek_mismatch_aligned(buf1: &[u8], buf2: &[u8], offset: usize, len: usize, bs: u64) -> Range {
    let mut r = align_range(seek_mismatch(buf1, buf2, offset, len), bs);
    if r.end != 0 {
        loop {
            let n = align_range(seek_mismatch(buf1, buf2, r.end as usize, len), bs);
            if n.end == 0 || r.end < n.start {
                break;
            }
            r.end = n.end;
        }
    }
    r
}

fn copy_sync_file_range(
    fs: &Fs,
    s: &mut CopyFsState,
    dst_inum: c::subvol_inum,
    dst: &mut c::bch_inode_unpacked,
    src_fd: BorrowedFd<'_>,
    src_size: u64,
    range: &Range,
) -> Result<(), BchError> {
    let block_size = fs.block_bytes();
    let mut start = range.start;

    while start != range.end {
        s.check_cancellation()?;
        let b = std::cmp::min(range.end - start, MAX_IO_SIZE as u64) as usize;

        let mut src_buf = AlignedBuf::new(b);
        let read_len = std::cmp::min(b, (src_size - start) as usize);
        rustix::io::pread(src_fd, &mut src_buf[..read_len], start)
            .map_err(rustix_err)?;

        let mut dst_buf = AlignedBuf::new(b);
        block_on(fs.read(dst_inum, start, dst, &mut dst_buf))?;

        let mut m = Range { start: 0, end: 0 };
        loop {
            m = seek_mismatch_aligned(&src_buf, &dst_buf, m.end as usize, b, block_size);
            if m.end == 0 {
                break;
            }
            s.ensure_reservation_capacity_for_copy(m.end - m.start)?;
            write_data(fs, dst, start + m.start, &src_buf[m.start as usize..m.end as usize])?;
            s.record_copy(CopyReason::Other, m.end - m.start, m.end - m.start);
        }

        start += b as u64;
    }

    Ok(())
}

fn copy_sync_file_data(
    fs: &Fs,
    s: &mut CopyFsState,
    dst_inum: c::subvol_inum,
    dst: &mut c::bch_inode_unpacked,
    src_fd: BorrowedFd<'_>,
    src_size: u64,
) -> Result<(), BchError> {
    let block_size = fs.block_bytes();
    let mut i_sectors_delta: i64 = 0;

    let mut prev = Range { start: 0, end: 0 };

    loop {
        s.check_cancellation()?;
        let next = seek_data_aligned(src_fd, src_size, prev.end, block_size);
        if next.end == 0 {
            break;
        }

        if next.start != 0 {
            io_misc::fpunch(fs, dst_inum, prev.end >> 9, next.start >> 9, &mut i_sectors_delta)?;
        }

        copy_sync_file_range(fs, s, dst_inum, dst, src_fd, src_size, &next)?;
        s.total_input += next.end - next.start;
        prev = next;
    }

    // End of file — truncate remaining
    io_misc::fpunch(fs, dst_inum, prev.end >> 9, u64::MAX, &mut i_sectors_delta)?;

    Ok(())
}

/// Directory entry from bcachefs readdir.
struct DirEntry {
    inum:   u64,
    dtype:  u8,
    name:   CString,
}

/// Trampoline for bch2_readdir callback.
#[repr(C)]
struct ReadDirCtx {
    ctx:     c::dir_context,
    entries: *mut Vec<DirEntry>,
}

unsafe extern "C" fn readdir_actor(
    ctx: *mut c::dir_context,
    name: *const libc::c_char,
    name_len: i32,
    _pos: i64,
    inum: u64,
    dtype: u32,
) -> i32 {
    let (entries, name_slice) = unsafe {
        let rctx = &mut *(ctx as *mut ReadDirCtx);
        (
            &mut *rctx.entries,
            std::slice::from_raw_parts(name as *const u8, name_len as usize),
        )
    };
    let cname = CString::new(name_slice).unwrap_or_else(|_| CString::new("?").unwrap());
    entries.push(DirEntry { inum, dtype: dtype as u8, name: cname });
    0
}

fn simple_readdir(
    fs: &Fs,
    dir_inum: c::subvol_inum,
    dir: &mut c::bch_inode_unpacked,
) -> Result<Vec<DirEntry>, BchError> {
    let mut hash_info = str_hash::hash_info_init(fs, dir)?;

    let mut entries = Vec::new();
    let mut rctx = ReadDirCtx {
        ctx: c::dir_context {
            actor: Some(readdir_actor),
            pos: 0,
        },
        entries: &mut entries,
    };

    dirent::readdir(fs, dir_inum, &mut hash_info, &mut rctx.ctx)?;

    // Sort by (type, name) like the C code
    entries.sort_by(|a, b| {
        a.dtype.cmp(&b.dtype).then_with(|| a.name.cmp(&b.name))
    });

    Ok(entries)
}

fn recursive_remove(
    fs: &Fs,
    dir_inum: c::subvol_inum,
    dir: &mut c::bch_inode_unpacked,
    d: &DirEntry,
) -> Result<(), BchError> {
    let child_inum = c::subvol_inum { subvol: dir_inum.subvol, inum: d.inum };

    let mut child = inode::find_by_inum(fs, child_inum)?;

    if (child.bi_mode as u32 & libc::S_IFMT) == libc::S_IFDIR {
        let child_dirents = simple_readdir(fs, child_inum, &mut child)?;
        for entry in &child_dirents {
            recursive_remove(fs, child_inum, &mut child, entry)?;
        }
    }

    unlink_and_rm(fs, dir_inum, dir, &d.name)
}

fn delete_non_matching_dirents(
    fs: &Fs,
    s: &CopyFsState,
    dst_dir_inum: c::subvol_inum,
    dst_dir: &mut c::bch_inode_unpacked,
    src_dirents: &[DirEntryInfo],
) -> Result<(), BchError> {
    let dst_dirents = simple_readdir(fs, dst_dir_inum, dst_dir)?;

    let mut src_idx = 0;
    for dst_d in &dst_dirents {
        while src_idx < src_dirents.len() && dirent_cmp_key(&src_dirents[src_idx]) < dirent_cmp_key_de(dst_d) {
            src_idx += 1;
        }

        let matches = src_idx < src_dirents.len()
            && dirent_cmp_key(&src_dirents[src_idx]) == dirent_cmp_key_de(dst_d);

        if !matches {
            if subvol_inum_eq(dst_dir_inum, root_subvol_inum())
                && dst_d.name.to_bytes() == b"lost+found"
            {
                continue;
            }

            if s.verbosity > 1 {
                println!("deleting {}", dst_d.name.to_string_lossy());
            }

            recursive_remove(fs, dst_dir_inum, dst_dir, dst_d)?;
        }
    }

    Ok(())
}

/// Comparison key for sorting: (dtype, name)
fn dirent_cmp_key(d: &DirEntryInfo) -> (u8, &[u8]) {
    (d.dtype, d.name.to_bytes())
}

fn dirent_cmp_key_de(d: &DirEntry) -> (u8, &[u8]) {
    (d.dtype, d.name.to_bytes())
}

/// Source directory entry info from host filesystem.
struct DirEntryInfo {
    _inum:  u64,
    dtype:  u8,
    name:   CString,
    stat:   rustix::fs::Stat,
}

fn source_dirents(src_fd: &OwnedFd) -> Result<Vec<DirEntryInfo>, BchError> {
    // Dir::read_from borrows the fd (creating an internal dup for iteration),
    // so src_fd remains available for fstatat/openat/fchdir below.
    let mut dir = rustix::fs::Dir::read_from(src_fd).map_err(rustix_err)?;
    let mut dirents = Vec::new();

    while let Some(entry) = dir.read() {
        let entry = entry.map_err(rustix_err)?;
        let name = entry.file_name();

        let stat = match rustix::fs::statat(
            src_fd,
            name,
            rustix::fs::AtFlags::SYMLINK_NOFOLLOW,
        ) {
            Ok(stat) => stat,
            Err(_) => continue,
        };

        dirents.push(DirEntryInfo {
            _inum: entry.ino(),
            dtype: file_type_to_dtype(entry.file_type()),
            name: name.to_owned(),
            stat,
        });
    }

    dirents.sort_by(|a, b| a.dtype.cmp(&b.dtype).then_with(|| a.name.cmp(&b.name)));
    Ok(dirents)
}

fn skip_source_entry(s: &CopyFsState, d: &DirEntryInfo) -> bool {
    (d.name.to_bytes() == b"."
        || d.name.to_bytes() == b".."
        || d.name.to_bytes() == b"lost+found")
        || (s.migrate_type == MigrateType::Migrate && d.stat.st_ino as u64 == s.bcachefs_inum)
}

fn count_source_entries(s: &CopyFsState, src_fd: OwnedFd) -> Result<u64, BchError> {
    let mut entries: u64 = 0;
    let oflags = rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::NOATIME;

    for d in source_dirents(&src_fd)? {
        s.check_cancellation()?;
        if skip_source_entry(s, &d) {
            continue;
        }
        if s.migrate_type == MigrateType::Migrate && d.stat.st_dev != s.dev {
            return Err(BchError::from_raw(-libc::EINVAL));
        }

        entries = entries
            .checked_add(1)
            .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
        if mode_to_type(d.stat.st_mode) == DT_DIR {
            let fd = rustix::fs::openat(&src_fd, &d.name, oflags, rustix::fs::Mode::empty())
                .map_err(rustix_err)?;
            entries = entries
                .checked_add(count_source_entries(s, fd)?)
                .ok_or_else(|| BchError::from_raw(-libc::EOVERFLOW))?;
        }
    }

    Ok(entries)
}

fn copy_dir(
    fs: &Fs,
    s: &mut CopyFsState,
    dst: &mut c::bch_inode_unpacked,
    src_fd: OwnedFd,
    src_path: &CStr,
) -> Result<(), BchError> {
    let dirents = source_dirents(&src_fd)?;

    let dir_inum = subvol_inum(dst.bi_inum);

    delete_non_matching_dirents(fs, s, dir_inum, dst, &dirents)?;

    let oflags = rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::NOATIME;

    for d in &dirents {
        s.check_cancellation()?;
        rustix::process::fchdir(&src_fd).map_err(rustix_err)?;

        let name_str = d.name.to_str().unwrap_or("?");
        if skip_source_entry(s, d) {
            continue;
        }

        s.total_files += 1;
        let is_directory = mode_to_type(d.stat.st_mode) == DT_DIR;

        let child_path = CString::new(format!(
            "{}/{}",
            src_path.to_string_lossy(),
            name_str
        )).unwrap();

        if s.migrate_type == MigrateType::Migrate && d.stat.st_dev != s.dev {
            eprintln!("{} does not have correct st_dev!", child_path.to_string_lossy());
            return Err(BchError::from_raw(-libc::EINVAL));
        }

        // Hardlink handling
        if (d.stat.st_mode & libc::S_IFMT) == libc::S_IFREG && d.stat.st_nlink > 1 {
            let src_ino = d.stat.st_ino as u64;
            if let Some(&dst_ino) = s.hardlinks.get(&src_ino) {
                create_or_update_link(
                    fs, dir_inum, dst, &d.name,
                    subvol_inum(dst_ino),
                )?;
                s.finish_entry(fs, false)?;
                continue;
            }
        }

        let mut inode = create_or_update_file(
            fs, dir_inum, dst, &d.name,
            d.stat.st_uid, d.stat.st_gid,
            d.stat.st_mode, d.stat.st_rdev,
        )?;

        let dst_child_inum = subvol_inum(inode.bi_inum);

        // Record hardlink destination
        if (d.stat.st_mode & libc::S_IFMT) == libc::S_IFREG && d.stat.st_nlink > 1 {
            s.hardlinks.insert(d.stat.st_ino as u64, inode.bi_inum);
        }

        let name_cstr = &d.name;
        copy_xattrs(fs, &mut inode, name_cstr, &child_path)?;

        match mode_to_type(d.stat.st_mode) {
            DT_DIR => {
                let fd = rustix::fs::openat(&src_fd, &d.name, oflags, rustix::fs::Mode::empty())
                    .map_err(rustix_err)?;
                copy_dir(fs, s, &mut inode, fd, &child_path)?;
            }
            DT_REG => {
                inode.bi_size = d.stat.st_size as u64;

                let fd = rustix::fs::openat(&src_fd, &d.name, oflags, rustix::fs::Mode::empty())
                    .map_err(rustix_err)?;

                if s.migrate_type == MigrateType::Migrate {
                    link_file_data(fs, s, &mut inode, fd.as_fd(), &child_path, d.stat.st_size as u64)?;
                } else {
                    copy_sync_file_data(fs, s, dst_child_inum, &mut inode, fd.as_fd(), d.stat.st_size as u64)?;
                }
                // fd dropped here — close is automatic
            }
            DT_LNK => {
                inode.bi_size = d.stat.st_size as u64;
                copy_link(fs, s, dst_child_inum, &mut inode, name_cstr)?;
            }
            _ => {
                // DT_FIFO, DT_CHR, DT_BLK, DT_SOCK, DT_WHT — nothing else to copy
            }
        }

        copy_times(fs, &mut inode, &d.stat);
        update_inode(fs, &inode)?;
        s.finish_entry(fs, is_directory)?;
    }

    Ok(())
}

fn first_device_total_sectors(fs: &Fs) -> u64 {
    let ca = fs.dev_get(0).expect("device 0 should exist");
    ca.mi.nbuckets * ca.mi.bucket_size as u64
}

fn reserve_old_fs_space(
    fs: &Fs,
    cancellation_probe: Option<fn() -> bool>,
    root_inode: &mut c::bch_inode_unpacked,
    extents: &mut Vec<(u64, u64)>,
    reserve_start: u64,
) -> Result<(), BchError> {
    let total_sectors = first_device_total_sectors(fs);

    let root_inum = subvol_inum(root_inode.bi_inum);
    let name = CString::new("old_migrated_filesystem").unwrap();
    let mut dst = create_or_update_file(
        fs, root_inum, root_inode,
        &name, 0, 0, libc::S_IFREG | 0o400, 0,
    )?;
    dst.bi_size = total_sectors << 9;

    ranges_sort_merge(extents);

    for_each_hole(extents, total_sectors << 9, |hole_start, hole_end| {
        check_cancellation_probe(cancellation_probe)?;
        if hole_end <= reserve_start {
            return Ok(());
        }
        let start = std::cmp::max(hole_start, reserve_start);
        link_data(fs, &mut dst, start, start, hole_end - start)
    })?;

    update_inode(fs, &dst)
}

/// Copy a POSIX directory tree into a bcachefs filesystem.
///
/// Entry point for both `bcachefs format --source` and `bcachefs migrate`.
pub fn copy_fs(
    fs: &Fs,
    s: &mut CopyFsState,
    src_fd: BorrowedFd<'_>,
    src_path: &CStr,
) -> Result<(), BchError> {
    let stat = rustix::fs::fstat(src_fd).map_err(rustix_err)?;
    if (stat.st_mode & libc::S_IFMT) != libc::S_IFDIR {
        eprintln!("{} is not a directory", src_path.to_string_lossy());
        return Err(BchError::from_raw(-libc::ENOTDIR));
    }

    if s.migrate_type == MigrateType::Migrate {
        rustix::fs::syncfs(src_fd).map_err(rustix_err)?;
    }

    let prepass_fd = rustix::io::dup(src_fd).map_err(rustix_err)?;
    let entries_total = count_source_entries(s, prepass_fd)?;
    s.start_progress(fs, entries_total);

    let mut root_inode = inode::find_by_inum(fs, root_subvol_inum())?;

    rustix::process::fchdir(src_fd).map_err(rustix_err)?;

    copy_times(fs, &mut root_inode, &stat);

    let dot = CString::new(".").unwrap();
    copy_xattrs(fs, &mut root_inode, &dot, src_path)?;

    let dup_fd = rustix::io::dup(src_fd).map_err(rustix_err)?;
    copy_dir(fs, s, &mut root_inode, dup_fd, src_path)?;

    if s.migrate_type == MigrateType::Migrate {
        let cancellation_probe = s.cancellation_probe;
        let reserve_start = s.reserve_start;
        reserve_old_fs_space(
            fs,
            cancellation_probe,
            &mut root_inode,
            &mut s.extents,
            reserve_start,
        )?;
    }

    update_inode(fs, &root_inode)?;
    s.finish_progress(fs);

    // Print summary
    println!("Total files:\t{}", s.total_files);
    print!("Total input:\t");
    print_human_readable(s.total_input);
    println!();

    if s.total_wrote > 0 {
        print!("Wrote:\t");
        print_human_readable(s.total_wrote);
        println!();
    }

    if s.total_linked > 0 {
        print!("Linked:\t");
        print_human_readable(s.total_linked);
        println!();
    }
    if !s.copy_accounting.is_consistent() {
        eprintln!(
            "MIGRATE_COPY_REASONS_INVALID copied_bytes={} copy_allocated_bytes={} reason_bytes={}",
            s.total_wrote,
            s.copy_accounting.allocated_copy_bytes,
            s.copy_accounting.reason_sum(),
        );
        return Err(BchError::from_raw(-libc::EIO));
    }
    println!(
        "MIGRATE_COPY_REASONS mode=migrate copied_bytes={} copy_allocated_bytes={} copied_shared_bytes={} copied_inline_bytes={} copied_encoded_bytes={} copied_encrypted_bytes={} copied_not_aligned_bytes={} copied_tail_bytes={} copied_merged_bytes={} copied_future_flags_bytes={} copied_reservation_overlap_bytes={} copied_raw_unaligned_bytes={} copied_raw_unaligned_head_bytes={} copied_raw_unaligned_tail_bytes={} copied_raw_unaligned_unsalvageable_bytes={} copied_raw_mapping_unavailable_bytes={} copied_unstable_bytes={} copied_symlink_bytes={} copied_other_bytes={}",
        s.total_wrote,
        s.copy_accounting.allocated_copy_bytes,
        s.copy_accounting.shared,
        s.copy_accounting.inline,
        s.copy_accounting.encoded,
        s.copy_accounting.encrypted,
        s.copy_accounting.not_aligned,
        s.copy_accounting.tail,
        s.copy_accounting.merged,
        s.copy_accounting.future_flags,
        s.copy_accounting.reservation_overlap,
        s.copy_accounting.raw_unaligned_bytes(),
        s.copy_accounting.raw_unaligned_head,
        s.copy_accounting.raw_unaligned_tail,
        s.copy_accounting.raw_unaligned_unsalvageable,
        s.copy_accounting.raw_mapping_unavailable,
        s.copy_accounting.unstable,
        s.copy_accounting.symlink,
        s.copy_accounting.other,
    );
    if std::env::var_os("BCACHEFS_TEST_COPY_DIAGNOSTICS").is_some() {
        let mut reasons = s.copy_accounting.reason_counts();
        reasons.sort_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| left.0.name().cmp(right.0.name()))
        });
        for (rank, (reason, bytes)) in reasons
            .into_iter()
            .filter(|(_, bytes)| *bytes != 0)
            .take(8)
            .enumerate()
        {
            println!(
                "MIGRATE_COPY_TOP mode=migrate rank={} reason={} copied_bytes={}",
                rank + 1,
                reason.name(),
                bytes,
            );
        }
    }
    println!(
        "MIGRATE_PLAN mode=migrate total_input_bytes={} linked_bytes={} copied_bytes={} copy_allocated_bytes={} copied_shared_bytes={} copied_inline_bytes={} copied_encoded_bytes={} copied_encrypted_bytes={} copied_not_aligned_bytes={} copied_tail_bytes={} copied_merged_bytes={} copied_future_flags_bytes={} copied_reservation_overlap_bytes={} copied_raw_unaligned_bytes={} copied_raw_unaligned_head_bytes={} copied_raw_unaligned_tail_bytes={} copied_raw_unaligned_unsalvageable_bytes={} copied_raw_mapping_unavailable_bytes={} copied_unstable_bytes={} copied_symlink_bytes={} copied_other_bytes={} raw_link_salvaged_bytes={} unwritten_hole_bytes={} unstable_extent_bytes={} mapped_extent_count={} mapped_segment_count={} chunk_split_count={}",
        s.total_input,
        s.total_linked,
        s.total_wrote,
        s.copy_accounting.allocated_copy_bytes,
        s.copy_accounting.shared,
        s.copy_accounting.inline,
        s.copy_accounting.encoded,
        s.copy_accounting.encrypted,
        s.copy_accounting.not_aligned,
        s.copy_accounting.tail,
        s.copy_accounting.merged,
        s.copy_accounting.future_flags,
        s.copy_accounting.reservation_overlap,
        s.copy_accounting.raw_unaligned_bytes(),
        s.copy_accounting.raw_unaligned_head,
        s.copy_accounting.raw_unaligned_tail,
        s.copy_accounting.raw_unaligned_unsalvageable,
        s.copy_accounting.raw_mapping_unavailable,
        s.copy_accounting.unstable,
        s.copy_accounting.symlink,
        s.copy_accounting.other,
        s.raw_link_salvaged_bytes,
        s.total_unwritten,
        s.total_unstable,
        s.mapped_extents,
        s.mapped_segments,
        s.chunk_splits,
    );
    println!("MIGRATE_RESULT eligible=yes");

    Ok(())
}

fn print_human_readable(bytes: u64) {
    if bytes >= 1 << 30 {
        print!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64);
    } else if bytes >= 1 << 20 {
        print!("{:.1} MiB", bytes as f64 / (1u64 << 20) as f64);
    } else if bytes >= 1 << 10 {
        print!("{:.1} KiB", bytes as f64 / (1u64 << 10) as f64);
    } else {
        print!("{} B", bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn emits_progress_after_interval_without_completed_entries() {
        let mut state = CopyFsState::new_migrate(1, 1, 0, ExtentMapper::Identity, 0, 0);
        state.progress_last_emitted = Instant::now() - PROGRESS_INTERVAL;

        assert!(state.should_emit_progress());
    }
    #[test]
    fn translates_btrfs_logical_segment_to_file_offset() {
        let segment = PhysicalSegment {
            source_logical: 0x20_1000,
            physical: 0x40_2000,
            len: 0x2000,
        };
        assert_eq!(
            file_logical_for_segment(0x8000, 0x20_0000, &segment).unwrap(),
            0x9000
        );
    }

    #[test]
    fn rejects_segment_before_extent_mapping_origin() {
        let segment = PhysicalSegment {
            source_logical: 0x1fff,
            physical: 0,
            len: 0x1000,
        };
        assert!(file_logical_for_segment(0, 0x2000, &segment).is_err());
    }
    const USER_OBJ: u16 = 0x01;
    const USER: u16 = 0x02;
    const GROUP_OBJ: u16 = 0x04;
    const GROUP: u16 = 0x08;
    const MASK: u16 = 0x10;
    const OTHER: u16 = 0x20;
    const UNDEFINED: u32 = 0xffff_ffff;

    fn linux_acl(entries: &[(u16, u16, u32)]) -> Vec<u8> {
        let mut value = 0x0002u32.to_le_bytes().to_vec();
        for &(tag, perm, id) in entries {
            value.extend_from_slice(&tag.to_le_bytes());
            value.extend_from_slice(&perm.to_le_bytes());
            value.extend_from_slice(&id.to_le_bytes());
        }
        value
    }

    #[test]
    fn converts_linux_posix_acl_fixture_exactly() {
        let source = [
            0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x06, 0x00, 0xff, 0xff, 0xff, 0xff, 0x02,
            0x00, 0x04, 0x00, 0x39, 0x30, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0xff, 0xff,
            0xff, 0xff, 0x10, 0x00, 0x04, 0x00, 0xff, 0xff, 0xff, 0xff, 0x20, 0x00, 0x04,
            0x00, 0xff, 0xff, 0xff, 0xff,
        ];
        let expected = [
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x06, 0x00, 0x02, 0x00, 0x04, 0x00, 0x39,
            0x30, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x10, 0x00, 0x04, 0x00, 0x20, 0x00,
            0x04, 0x00,
        ];
        assert_eq!(linux_posix_acl_xattr_to_bcachefs(&source).unwrap(), expected);
    }

    #[test]
    fn converts_minimal_linux_posix_acl() {
        let source = linux_acl(&[
            (USER_OBJ, 6, UNDEFINED),
            (GROUP_OBJ, 4, UNDEFINED),
            (OTHER, 4, UNDEFINED),
        ]);
        let expected = [
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x06, 0x00, 0x04, 0x00, 0x04, 0x00, 0x20, 0x00,
            0x04, 0x00,
        ];
        assert_eq!(linux_posix_acl_xattr_to_bcachefs(&source).unwrap(), expected);
    }

    #[test]
    fn converts_named_group_entry() {
        let source = linux_acl(&[
            (GROUP_OBJ, 4, UNDEFINED),
            (GROUP, 4, 12345),
            (OTHER, 4, UNDEFINED),
        ]);
        let expected = [
            0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x08, 0x00, 0x04, 0x00, 0x39, 0x30,
            0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
        ];
        assert_eq!(linux_posix_acl_xattr_to_bcachefs(&source).unwrap(), expected);
    }

    #[test]
    fn converts_access_acl_containing_mask() {
        let source = linux_acl(&[
            (USER_OBJ, 6, UNDEFINED),
            (GROUP_OBJ, 4, UNDEFINED),
            (MASK, 4, UNDEFINED),
            (OTHER, 4, UNDEFINED),
        ]);
        let expected = [
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x06, 0x00, 0x04, 0x00, 0x04, 0x00, 0x10, 0x00,
            0x04, 0x00, 0x20, 0x00, 0x04, 0x00,
        ];
        assert_eq!(linux_posix_acl_xattr_to_bcachefs(&source).unwrap(), expected);
    }

    #[test]
    fn default_acl_uses_identical_binary_conversion() {
        let source = linux_acl(&[
            (USER_OBJ, 6, UNDEFINED),
            (USER, 4, 12345),
            (GROUP_OBJ, 4, UNDEFINED),
            (MASK, 4, UNDEFINED),
            (OTHER, 4, UNDEFINED),
        ]);
        let expected = linux_posix_acl_xattr_to_bcachefs(&source).unwrap();
        assert_eq!(
            xattr_value_for_storage(XATTR_INDEX_ACL_DEFAULT, &source)
                .unwrap()
                .as_ref(),
            expected.as_slice()
        );
    }

    #[test]
    fn rejects_wrong_external_acl_version() {
        let mut source = linux_acl(&[]);
        source[..4].copy_from_slice(&3u32.to_le_bytes());
        let error = linux_posix_acl_xattr_to_bcachefs(&source).unwrap_err();
        assert!(error.to_string().contains("version"));
    }

    #[test]
    fn rejects_acl_shorter_than_header() {
        assert!(linux_posix_acl_xattr_to_bcachefs(&[0, 0, 0]).is_err());
    }

    #[test]
    fn rejects_acl_payload_not_divisible_by_entry_size() {
        assert!(linux_posix_acl_xattr_to_bcachefs(&[2, 0, 0, 0, 1, 2, 3]).is_err());
    }

    #[test]
    fn rejects_unknown_acl_tag() {
        assert!(linux_posix_acl_xattr_to_bcachefs(&linux_acl(&[(0x40, 4, UNDEFINED)])).is_err());
    }

    #[test]
    fn rejects_acl_permissions_above_seven() {
        assert!(linux_posix_acl_xattr_to_bcachefs(&linux_acl(&[(USER_OBJ, 8, UNDEFINED)])).is_err());
    }

    #[test]
    fn rejects_named_user_with_undefined_id() {
        assert!(linux_posix_acl_xattr_to_bcachefs(&linux_acl(&[(USER, 4, UNDEFINED)])).is_err());
    }

    #[test]
    fn rejects_named_group_with_undefined_id() {
        assert!(linux_posix_acl_xattr_to_bcachefs(&linux_acl(&[(GROUP, 4, UNDEFINED)])).is_err());
    }

    #[test]
    fn rejects_short_acl_entry_with_concrete_id() {
        assert!(linux_posix_acl_xattr_to_bcachefs(&linux_acl(&[(USER_OBJ, 4, 12345)])).is_err());
    }

    #[test]
    fn resolves_only_exact_acl_names() {
        assert_eq!(
            xattr_resolve_name(b"system.posix_acl_access"),
            Some((XATTR_INDEX_ACL_ACCESS, &[][..]))
        );
        assert_eq!(
            xattr_resolve_name(b"system.posix_acl_default"),
            Some((XATTR_INDEX_ACL_DEFAULT, &[][..]))
        );
        assert_eq!(
            xattr_resolve_name(b"system.posix_acl_access.foo"),
            None
        );
        assert_eq!(
            xattr_resolve_name(b"system.posix_acl_default.foo"),
            None
        );
    }

    #[test]
    fn generic_xattr_value_remains_byte_for_byte() {
        let value = [0, 1, 0xff, 0x7f];
        assert_eq!(
            xattr_value_for_storage(XATTR_INDEX_USER, &value)
                .unwrap()
                .as_ref(),
            value
        );
    }
    #[test]
    fn access_acl_maps_to_access_inode_flag() {
        assert_eq!(
            acl_inode_flag(XATTR_INDEX_ACL_ACCESS),
            Some(c::bch_inode_flags::BCH_INODE_has_access_acl as u32)
        );
    }

    #[test]
    fn default_acl_maps_to_default_inode_flag() {
        assert_eq!(
            acl_inode_flag(XATTR_INDEX_ACL_DEFAULT),
            Some(c::bch_inode_flags::BCH_INODE_has_default_acl as u32)
        );
    }

    #[test]
    fn generic_xattrs_do_not_map_to_acl_flags() {
        assert_eq!(acl_inode_flag(XATTR_INDEX_USER), None);
        assert_eq!(acl_inode_flag(XATTR_INDEX_TRUSTED), None);
        assert_eq!(acl_inode_flag(XATTR_INDEX_SECURITY), None);
    }

    #[test]
    fn setting_access_flag_preserves_unrelated_inode_flags() {
        let unrelated = c::bch_inode_flags::BCH_INODE_unlinked as u32;
        let flags = add_acl_inode_flag(unrelated, XATTR_INDEX_ACL_ACCESS);
        assert_eq!(flags & unrelated, unrelated);
        assert_ne!(
            flags & (c::bch_inode_flags::BCH_INODE_has_access_acl as u32),
            0
        );
    }

    #[test]
    fn setting_default_flag_preserves_unrelated_inode_flags() {
        let unrelated = c::bch_inode_flags::BCH_INODE_unlinked as u32;
        let flags = add_acl_inode_flag(unrelated, XATTR_INDEX_ACL_DEFAULT);
        assert_eq!(flags & unrelated, unrelated);
        assert_ne!(
            flags & (c::bch_inode_flags::BCH_INODE_has_default_acl as u32),
            0
        );
    }

    #[test]
    fn setting_both_acl_flags_sets_both_bits() {
        let flags = add_acl_inode_flag(
            add_acl_inode_flag(0, XATTR_INDEX_ACL_ACCESS),
            XATTR_INDEX_ACL_DEFAULT,
        );
        assert_ne!(
            flags & (c::bch_inode_flags::BCH_INODE_has_access_acl as u32),
            0
        );
        assert_ne!(
            flags & (c::bch_inode_flags::BCH_INODE_has_default_acl as u32),
            0
        );
    }

    #[test]
    fn directory_default_acl_receives_flag_for_immediate_persist() {
        let unrelated = c::bch_inode_flags::BCH_INODE_has_inode_opts as u32;
        let flags = add_acl_inode_flag(unrelated, XATTR_INDEX_ACL_DEFAULT);
        assert_eq!(
            flags & (c::bch_inode_flags::BCH_INODE_has_default_acl as u32),
            c::bch_inode_flags::BCH_INODE_has_default_acl as u32
        );
        assert_eq!(flags & unrelated, unrelated);
    }
}
