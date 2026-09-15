use anyhow::{bail, Result};
use std::cmp::{min, Ordering};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io;
pub(crate) const BTRFS_BLOCK_GROUP_DATA_MASK: u64 =
    bch_bindgen::c::BCH_BTRFS_BLOCK_GROUP_DATA as u64;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::ffi::OsStringExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};


const BTRFS_BLOCK_GROUP_DATA: u64 =
    bch_bindgen::c::BCH_BTRFS_BLOCK_GROUP_DATA as u64;
const BTRFS_BLOCK_GROUP_SYSTEM: u64 =
    bch_bindgen::c::BCH_BTRFS_BLOCK_GROUP_SYSTEM as u64;
const BTRFS_BLOCK_GROUP_METADATA: u64 =
    bch_bindgen::c::BCH_BTRFS_BLOCK_GROUP_METADATA as u64;
const BTRFS_BLOCK_GROUP_PROFILE_MASK: u64 =
    bch_bindgen::c::BCH_BTRFS_BLOCK_GROUP_PROFILE_MASK as u64;

const CHUNK_HEADER_BYTES: usize = 48;
const STRIPE_BYTES: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkMap {
    pub logical_start: u64,
    pub logical_end: u64,
    pub stripe_offset: u64,
    pub devid: u64,
    pub chunk_type: u64,
    pub num_stripes: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalSegment {
    pub source_logical: u64,
    pub physical: u64,
    pub len: u64,
}

#[derive(Clone, Debug)]
pub struct BtrfsMapper {
    pub fsid: [u8; 16],
    pub devid: u64,
    pub device_path: std::path::PathBuf,
    pub device_size: u64,
    pub sectorsize: u32,
    chunks: Vec<ChunkMap>,
}

impl BtrfsMapper {
    pub fn from_parts(
        fsid: [u8; 16],
        devid: u64,
        device_path: std::path::PathBuf,
        device_size: u64,
        sectorsize: u32,
        mut chunks: Vec<ChunkMap>,
    ) -> Result<Self> {
        if devid == 0 {
            bail!("Btrfs migration unsupported: device id 0 is not a real device");
        }
        if sectorsize == 0 || !sectorsize.is_power_of_two() {
            bail!(
                "Btrfs migration unsupported: invalid sectorsize {}; mapping alignment is unsafe",
                sectorsize
            );
        }

        for chunk in &chunks {
            if chunk.logical_start >= chunk.logical_end {
                bail!(
                    "Btrfs migration unsupported: chunk at logical {} has empty or reversed range",
                    chunk.logical_start
                );
            }
            if chunk.devid != devid {
                bail!(
                    "Btrfs migration unsupported: chunk at logical {} uses devid {}, expected sole devid {}; mapping is unsafe",
                    chunk.logical_start,
                    chunk.devid,
                    devid
                );
            }

            let type_bits = chunk.chunk_type
                & (BTRFS_BLOCK_GROUP_DATA
                    | BTRFS_BLOCK_GROUP_SYSTEM
                    | BTRFS_BLOCK_GROUP_METADATA);
            if !matches!(
                type_bits,
                BTRFS_BLOCK_GROUP_DATA
                    | BTRFS_BLOCK_GROUP_SYSTEM
                    | BTRFS_BLOCK_GROUP_METADATA
            ) {
                bail!(
                    "Btrfs migration unsupported: chunk at logical {} has invalid type 0x{:x}",
                    chunk.logical_start,
                    chunk.chunk_type
                );
            }
            if chunk.chunk_type & !(BTRFS_BLOCK_GROUP_PROFILE_MASK
                | BTRFS_BLOCK_GROUP_DATA
                | BTRFS_BLOCK_GROUP_SYSTEM
                | BTRFS_BLOCK_GROUP_METADATA)
                != 0
            {
                bail!(
                    "Btrfs migration unsupported: chunk at logical {} has unknown type bits 0x{:x}",
                    chunk.logical_start,
                    chunk.chunk_type
                );
            }
            if chunk.chunk_type & BTRFS_BLOCK_GROUP_DATA != 0
                && (chunk.chunk_type & BTRFS_BLOCK_GROUP_PROFILE_MASK != 0
                    || chunk.num_stripes != 1)
            {
                bail!(
                    "Btrfs migration unsupported: data chunk at logical {} uses profile 0x{:x} with {} stripes; only single-device Data SINGLE is supported",
                    chunk.logical_start,
                    chunk.chunk_type & BTRFS_BLOCK_GROUP_PROFILE_MASK,
                    chunk.num_stripes
                );
            }
        }

        chunks.sort_by_key(|chunk| chunk.logical_start);
        for pair in chunks.windows(2) {
            if pair[0].logical_end > pair[1].logical_start {
                bail!(
                    "Btrfs migration unsupported: overlapping chunks at logical {} and {}",
                    pair[0].logical_start,
                    pair[1].logical_start
                );
            }
        }

        Ok(Self {
            fsid,
            devid,
            device_path,
            device_size,
            sectorsize,
            chunks,
        })
    }

    pub fn chunks(&self) -> &[ChunkMap] {
        &self.chunks
    }

    pub fn validate_physical_ranges(&self) -> Result<()> {
        for chunk in &self.chunks {
            let length = chunk.logical_end - chunk.logical_start;
            let end = chunk.stripe_offset.checked_add(length).ok_or_else(|| {
                anyhow::anyhow!(
                    "Btrfs migration unsupported: chunk at logical {} overflows raw device offset",
                    chunk.logical_start
                )
            })?;
            if end > self.device_size {
                bail!(
                    "Btrfs migration unsupported: chunk at logical {} ends at raw offset {}, beyond device size {}; mapping is unsafe",
                    chunk.logical_start,
                    end,
                    self.device_size
                );
            }
        }
        Ok(())
    }

    pub fn map_physical(&self, source_logical: u64, length: u64) -> Result<Vec<PhysicalSegment>> {
        if length == 0 {
            return Ok(Vec::new());
        }

        let _source_end = source_logical.checked_add(length).ok_or_else(|| {
            anyhow::anyhow!(
                "Btrfs migration unsupported: logical extent {}+{} overflows",
                source_logical,
                length
            )
        })?;

        let mut logical = source_logical;
        let mut remaining = length;
        let mut segments = Vec::new();

        while remaining != 0 {
            let index = match self.chunks.binary_search_by_key(&logical, |chunk| {
                chunk.logical_start
            }) {
                Ok(index) => index,
                Err(0) => {
                    bail!(
                        "Btrfs migration unsupported: logical address {} is before the first chunk",
                        logical
                    )
                }
                Err(index) => index - 1,
            };
            let chunk = &self.chunks[index];
            if logical >= chunk.logical_end {
                bail!(
                    "Btrfs migration unsupported: logical address {} falls in a chunk gap",
                    logical
                );
            }

            let profile = chunk.chunk_type & BTRFS_BLOCK_GROUP_PROFILE_MASK;
            if chunk.chunk_type & BTRFS_BLOCK_GROUP_DATA == 0 {
                bail!(
                    "Btrfs migration unsupported: logical address {} is in a non-data chunk type 0x{:x}",
                    logical,
                    chunk.chunk_type
                );
            }
            if profile != 0 || chunk.num_stripes != 1 || chunk.devid != self.devid {
                bail!(
                    "Btrfs migration unsupported: data chunk at logical {} is not single-device Data SINGLE (type=0x{:x}, stripes={}, devid={}); raw linking is unsafe",
                    chunk.logical_start,
                    chunk.chunk_type,
                    chunk.num_stripes,
                    chunk.devid
                );
            }

            let available = chunk.logical_end - logical;
            let take = min(remaining, available);
            let delta = logical - chunk.logical_start;
            let physical = chunk.stripe_offset.checked_add(delta).ok_or_else(|| {
                anyhow::anyhow!(
                    "Btrfs migration unsupported: raw offset overflow for logical address {}",
                    logical
                )
            })?;
            let physical_end = physical.checked_add(take).ok_or_else(|| {
                anyhow::anyhow!(
                    "Btrfs migration unsupported: raw extent {}+{} overflows",
                    physical,
                    take
                )
            })?;
            if physical_end > self.device_size {
                bail!(
                    "Btrfs migration unsupported: mapped raw extent [{}, {}) exceeds device size {}; mapping is unsafe",
                    physical,
                    physical_end,
                    self.device_size
                );
            }

            segments.push(PhysicalSegment {
                source_logical: logical,
                physical,
                len: take,
            });
            logical = logical.checked_add(take).ok_or_else(|| {
                anyhow::anyhow!("Btrfs migration unsupported: logical split offset overflow")
            })?;
            remaining -= take;
        }

        Ok(segments)
    }
}

pub(crate) fn parse_chunk_item(logical_start: u64, payload: &[u8]) -> Result<ChunkMap> {
    if payload.len() < CHUNK_HEADER_BYTES {
        bail!(
            "Btrfs migration unsupported: chunk item at logical {} is truncated ({} bytes, need at least {})",
            logical_start,
            payload.len(),
            CHUNK_HEADER_BYTES
        );
    }

    let length = read_le_u64(payload, 0)?;
    let chunk_type = read_le_u64(payload, 24)?;
    let sector_size = read_le_u32(payload, 40)?;
    // btrfs_chunk stores sector_size at 40, num_stripes at 44, then stripe[0].
    let num_stripes = read_le_u16(payload, 44)?;
    if sector_size == 0 || !sector_size.is_power_of_two() {
        bail!(
            "Btrfs migration unsupported: chunk item at logical {} has invalid sector size {}; mapping alignment is unsafe",
            logical_start,
            sector_size
        );
    }
    if length == 0 {
        bail!(
            "Btrfs migration unsupported: chunk item at logical {} has zero length",
            logical_start
        );
    }
    if num_stripes == 0 {
        bail!(
            "Btrfs migration unsupported: chunk item at logical {} has zero stripes",
            logical_start
        );
    }

    let required = CHUNK_HEADER_BYTES
        .checked_add((num_stripes as usize).checked_mul(STRIPE_BYTES).ok_or_else(|| {
            anyhow::anyhow!("Btrfs migration unsupported: stripe count overflows payload size")
        })?)
        .ok_or_else(|| anyhow::anyhow!("Btrfs migration unsupported: chunk payload size overflow"))?;
    if payload.len() < required {
        bail!(
            "Btrfs migration unsupported: chunk item at logical {} declares {} stripes but payload has {} bytes; need {}",
            logical_start,
            num_stripes,
            payload.len(),
            required
        );
    }

    let logical_end = logical_start.checked_add(length).ok_or_else(|| {
        anyhow::anyhow!(
            "Btrfs migration unsupported: chunk logical range {}+{} overflows",
            logical_start,
            length
        )
    })?;
    let devid = read_le_u64(payload, CHUNK_HEADER_BYTES)?;
    let stripe_offset = read_le_u64(payload, CHUNK_HEADER_BYTES + 8)?;
    stripe_offset.checked_add(length).ok_or_else(|| {
        anyhow::anyhow!(
            "Btrfs migration unsupported: chunk at logical {} overflows raw stripe range",
            logical_start
        )
    })?;

    Ok(ChunkMap {
        logical_start,
        logical_end,
        stripe_offset,
        devid,
        chunk_type,
        num_stripes,
    })
}

fn read_le_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| anyhow::anyhow!("Btrfs migration unsupported: payload offset overflow"))?;
    let value = bytes.get(offset..end).ok_or_else(|| {
        anyhow::anyhow!("Btrfs migration unsupported: truncated little-endian u16 field")
    })?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_le_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| anyhow::anyhow!("Btrfs migration unsupported: payload offset overflow"))?;
    let value = bytes.get(offset..end).ok_or_else(|| {
        anyhow::anyhow!("Btrfs migration unsupported: truncated little-endian u64 field")
    })?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}
#[derive(Clone, Debug)]
pub enum ExtentMapper {
    Identity,
    Btrfs(BtrfsMapper),
}

impl ExtentMapper {
    pub fn map_physical(&self, source_logical: u64, length: u64) -> Result<Vec<PhysicalSegment>> {
        match self {
            Self::Identity => {
                if source_logical.checked_add(length).is_none() {
                    bail!(
                        "migration mapping failed: identity extent {}+{} overflows",
                        source_logical,
                        length
                    );
                }
                if length == 0 {
                    Ok(Vec::new())
                } else {
                    Ok(vec![PhysicalSegment {
                        source_logical,
                        physical: source_logical,
                        len: length,
                    }])
                }
            }
            Self::Btrfs(mapper) => mapper.map_physical(source_logical, length),
        }
    }
}


const BTRFS_FS_TREE_OBJECTID: u64 =
    bch_bindgen::c::BCH_BTRFS_FS_TREE_OBJECTID as u64;
const BTRFS_ROOT_TREE_OBJECTID: u64 =
    bch_bindgen::c::BCH_BTRFS_ROOT_TREE_OBJECTID as u64;
const BTRFS_FIRST_FREE_OBJECTID: u64 =
    bch_bindgen::c::BCH_BTRFS_FIRST_FREE_OBJECTID as u64;
const BTRFS_LAST_FREE_OBJECTID: u64 =
    bch_bindgen::c::BCH_BTRFS_LAST_FREE_OBJECTID as u64;
const BTRFS_CHUNK_TREE_OBJECTID: u64 =
    bch_bindgen::c::BCH_BTRFS_CHUNK_TREE_OBJECTID as u64;
const BTRFS_FIRST_CHUNK_TREE_OBJECTID: u64 =
    bch_bindgen::c::BCH_BTRFS_FIRST_CHUNK_TREE_OBJECTID as u64;
const BTRFS_ROOT_ITEM_KEY: u64 =
    bch_bindgen::c::BCH_BTRFS_ROOT_ITEM_KEY as u64;
const BTRFS_CHUNK_ITEM_KEY: u64 =
    bch_bindgen::c::BCH_BTRFS_CHUNK_ITEM_KEY as u64;
const BTRFS_SUPER_MAGIC: u64 =
    bch_bindgen::c::BCH_BTRFS_SUPER_MAGIC as u64;
const BTRFS_IOC_FS_INFO: libc::Ioctl =
    bch_bindgen::c::BCH_BTRFS_IOC_FS_INFO as libc::Ioctl;
const BTRFS_IOC_DEV_INFO: libc::Ioctl =
    bch_bindgen::c::BCH_BTRFS_IOC_DEV_INFO as libc::Ioctl;
const BTRFS_IOC_GET_FEATURES: libc::Ioctl =
    bch_bindgen::c::BCH_BTRFS_IOC_GET_FEATURES as libc::Ioctl;
const BTRFS_IOC_GET_SUBVOL_INFO: libc::Ioctl =
    bch_bindgen::c::BCH_BTRFS_IOC_GET_SUBVOL_INFO as libc::Ioctl;
const BTRFS_IOC_TREE_SEARCH_V2: libc::Ioctl =
    bch_bindgen::c::BCH_BTRFS_IOC_TREE_SEARCH_V2 as libc::Ioctl;

const BTRFS_FEATURE_INCOMPAT_MIXED_GROUPS: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_MIXED_GROUPS as u64;
const BTRFS_FEATURE_INCOMPAT_RAID56: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_RAID56 as u64;
const BTRFS_FEATURE_INCOMPAT_RAID1C34: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_RAID1C34 as u64;
const BTRFS_FEATURE_INCOMPAT_ZONED: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_ZONED as u64;
const BTRFS_FEATURE_INCOMPAT_EXTENT_TREE_V2: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_EXTENT_TREE_V2 as u64;
const BTRFS_FEATURE_INCOMPAT_RAID_STRIPE_TREE: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_RAID_STRIPE_TREE as u64;
const BTRFS_FEATURE_INCOMPAT_REMAP_TREE: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_REMAP_TREE as u64;
const BTRFS_FEATURE_INCOMPAT_MIXED_BACKREF: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_MIXED_BACKREF as u64;
const BTRFS_FEATURE_INCOMPAT_DEFAULT_SUBVOL: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_DEFAULT_SUBVOL as u64;
const BTRFS_FEATURE_INCOMPAT_COMPRESS_LZO: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_COMPRESS_LZO as u64;
const BTRFS_FEATURE_INCOMPAT_COMPRESS_ZSTD: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_COMPRESS_ZSTD as u64;
const BTRFS_FEATURE_INCOMPAT_BIG_METADATA: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_BIG_METADATA as u64;
const BTRFS_FEATURE_INCOMPAT_EXTENDED_IREF: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_EXTENDED_IREF as u64;
const BTRFS_FEATURE_INCOMPAT_SKINNY_METADATA: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_SKINNY_METADATA as u64;
const BTRFS_FEATURE_INCOMPAT_NO_HOLES: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_NO_HOLES as u64;
const BTRFS_FEATURE_INCOMPAT_METADATA_UUID: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_METADATA_UUID as u64;
const BTRFS_FEATURE_INCOMPAT_SIMPLE_QUOTA: u64 =
    bch_bindgen::c::BCH_BTRFS_FEATURE_INCOMPAT_SIMPLE_QUOTA as u64;

const BTRFS_SUPPORTED_INCOMPAT: u64 = BTRFS_FEATURE_INCOMPAT_MIXED_BACKREF
    | BTRFS_FEATURE_INCOMPAT_DEFAULT_SUBVOL
    | BTRFS_FEATURE_INCOMPAT_COMPRESS_LZO
    | BTRFS_FEATURE_INCOMPAT_COMPRESS_ZSTD
    | BTRFS_FEATURE_INCOMPAT_BIG_METADATA
    | BTRFS_FEATURE_INCOMPAT_EXTENDED_IREF
    | BTRFS_FEATURE_INCOMPAT_SKINNY_METADATA
    | BTRFS_FEATURE_INCOMPAT_NO_HOLES
    | BTRFS_FEATURE_INCOMPAT_METADATA_UUID
    | BTRFS_FEATURE_INCOMPAT_SIMPLE_QUOTA;
const BTRFS_HARD_REJECT_INCOMPAT: u64 = BTRFS_FEATURE_INCOMPAT_RAID56
    | BTRFS_FEATURE_INCOMPAT_RAID1C34
    | BTRFS_FEATURE_INCOMPAT_ZONED
    | BTRFS_FEATURE_INCOMPAT_EXTENT_TREE_V2
    | BTRFS_FEATURE_INCOMPAT_RAID_STRIPE_TREE
    | BTRFS_FEATURE_INCOMPAT_REMAP_TREE;

const SEARCH_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
pub struct BtrfsPreflight {
    pub source_fs_dev: u64,
    pub raw_device_path: PathBuf,
    pub raw_device_size: u64,
    pub fsid: [u8; 16],
    pub devid: u64,
    pub sectorsize: u32,
    pub num_devices: u64,
    pub subvolid: u64,
    pub incompat_flags: u64,
    pub subvolume_count: usize,
    pub mapper: BtrfsMapper,
}

pub fn preflight_source(source: &Path) -> Result<BtrfsPreflight> {
    let source_meta = fs::symlink_metadata(source).map_err(|e| {
        anyhow::anyhow!(
            "Btrfs migration preflight failed: cannot stat source {}: {}",
            source.display(),
            e
        )
    })?;
    if !source_meta.file_type().is_dir() {
        bail!(
            "Btrfs migration unsupported: source {} is not a directory",
            source.display()
        );
    }

    let root = open_directory(source)?;
    ensure_btrfs_statfs(root.as_raw_fd(), source)?;

    let subvol = ioctl_subvol_info(root.as_raw_fd(), source)?;
    if subvol.treeid as u64 != BTRFS_FS_TREE_OBJECTID {
        bail!(
            "Btrfs migration unsupported: source {} is subvolume {}, not top-level subvolid {}; child subvolumes are not allowed",
            source.display(),
            subvol.treeid,
            BTRFS_FS_TREE_OBJECTID
        );
    }
    let subvolume_count = read_subvolume_count(root.as_raw_fd())?;
    scan_for_child_subvolumes(source, source_meta.dev())?;

    let fs_info: bch_bindgen::c::btrfs_ioctl_fs_info_args =
        ioctl_readwrite(root.as_raw_fd(), BTRFS_IOC_FS_INFO, "BTRFS_IOC_FS_INFO")?;
    if fs_info.num_devices != 1 {
        bail!(
            "Btrfs migration unsupported: source has {} devices; exactly one device is required",
            fs_info.num_devices
        );
    }
    if fs_info.max_id == 0 {
        bail!("Btrfs migration unsupported: Btrfs reported no valid device id");
    }

    let mut dev_info: bch_bindgen::c::btrfs_ioctl_dev_info_args =
        bch_bindgen::c::btrfs_ioctl_dev_info_args::default();
    dev_info.devid = fs_info.max_id;
    ioctl_readwrite_in_place(
        root.as_raw_fd(),
        BTRFS_IOC_DEV_INFO,
        &mut dev_info,
        "BTRFS_IOC_DEV_INFO",
    )?;
    if dev_info.devid != fs_info.max_id {
        bail!(
            "Btrfs migration unsupported: device query returned devid {}, expected sole devid {}",
            dev_info.devid,
            fs_info.max_id
        );
    }
    if dev_info.fsid != fs_info.fsid {
        bail!(
            "Btrfs migration unsupported: device fsid does not match filesystem fsid; raw mapping is unsafe"
        );
    }
    let raw_device_path = kernel_path(&dev_info.path)?;
    if dev_info.total_bytes == 0 {
        bail!(
            "Btrfs migration unsupported: Btrfs reported a zero-sized raw device"
        );
    }
    let raw_device_size = match File::open(&raw_device_path) {
        Ok(raw_device) => {
            let observed = crate::wrappers::bdev::get_size(raw_device.as_raw_fd());
            if observed == 0 {
                dev_info.total_bytes
            } else if dev_info.total_bytes > observed {
                bail!(
                    "Btrfs migration unsupported: Btrfs device size {} exceeds raw device size {}; mapping is unsafe",
                    dev_info.total_bytes,
                    observed
                );
            } else {
                observed
            }
        }
        Err(error) => {
            eprintln!(
                "Btrfs migration preflight: raw device {} is not readable by this user ({}); using the kernel-reported device size {} for read-only mapping checks",
                raw_device_path.display(),
                error,
                dev_info.total_bytes
            );
            dev_info.total_bytes
        }
    };

    let features: bch_bindgen::c::btrfs_ioctl_feature_flags =
        ioctl_readwrite(root.as_raw_fd(), BTRFS_IOC_GET_FEATURES, "BTRFS_IOC_GET_FEATURES")?;
    validate_incompat_features(features.incompat_flags as u64)?;

    let chunks = read_chunk_tree(root.as_raw_fd())?;
    let mapper = BtrfsMapper::from_parts(
        fs_info.fsid,
        dev_info.devid,
        raw_device_path.clone(),
        raw_device_size,
        fs_info.sectorsize,
        chunks,
    )?;
    mapper.validate_physical_ranges()?;
    let mapper_fsid = mapper.fsid;
    let mapper_devid = mapper.devid;
    let mapper_device_path = mapper.device_path.clone();
    let mapper_sectorsize = mapper.sectorsize;
    if !mapper
        .chunks()
        .iter()
        .any(|chunk| chunk.chunk_type & BTRFS_BLOCK_GROUP_DATA != 0)
    {
        bail!(
            "Btrfs migration unsupported: Chunk Tree contains no Data chunks; source data cannot be mapped"
        );
    }

    Ok(BtrfsPreflight {
        source_fs_dev: source_meta.dev(),
        raw_device_path: mapper_device_path,
        raw_device_size,
        fsid: mapper_fsid,
        devid: mapper_devid,
        sectorsize: mapper_sectorsize,
        num_devices: fs_info.num_devices,
        subvolid: subvol.treeid as u64,
        subvolume_count,
        incompat_flags: features.incompat_flags as u64,
        mapper,
    })
}

fn open_directory(path: &Path) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|e| {
            anyhow::anyhow!(
                "Btrfs migration preflight failed: cannot open directory {} without following symlinks: {}",
                path.display(),
                e
            )
        })
}

fn ensure_btrfs_statfs(fd: RawFd, source: &Path) -> Result<()> {
    let mut stat = std::mem::MaybeUninit::<libc::statfs>::zeroed();
    let ret = unsafe { libc::fstatfs(fd, stat.as_mut_ptr()) };
    if ret < 0 {
        bail!(
            "Btrfs migration preflight failed: fstatfs({}) failed: {}",
            source.display(),
            io::Error::last_os_error()
        );
    }
    let stat = unsafe { stat.assume_init() };
    if stat.f_type as u64 != BTRFS_SUPER_MAGIC {
        bail!(
            "Btrfs migration unsupported: source {} is not Btrfs (filesystem magic 0x{:x})",
            source.display(),
            stat.f_type as u64
        );
    }
    Ok(())
}

fn scan_for_child_subvolumes(source: &Path, source_fs_dev: u64) -> Result<()> {
    let mut stack = vec![source.to_path_buf()];
    let mut visited = std::collections::HashSet::new();
    while let Some(path) = stack.pop() {
        let metadata = fs::symlink_metadata(&path).map_err(|e| {
            anyhow::anyhow!(
                "Btrfs migration preflight failed: cannot inspect {}: {}",
                path.display(),
                e
            )
        })?;
        if !metadata.file_type().is_dir() {
            continue;
        }
        if metadata.dev() != source_fs_dev {
            bail!(
                "Btrfs migration unsupported: {} crosses filesystem device boundary {} -> {}; source tree is not a single top-level Btrfs subvolume",
                path.display(),
                source_fs_dev,
                metadata.dev()
            );
        }
        if !visited.insert((metadata.dev(), metadata.ino())) {
            continue;
        }

        let dir = open_directory(&path)?;
        let subvol = ioctl_subvol_info(dir.as_raw_fd(), &path)?;
        if subvol.treeid as u64 != BTRFS_FS_TREE_OBJECTID {
            bail!(
                "Btrfs migration unsupported: child subvolume {} has subvolid {}; only top-level subvolid {} is supported",
                path.display(),
                subvol.treeid,
                BTRFS_FS_TREE_OBJECTID
            );
        }

        for entry in fs::read_dir(&path).map_err(|e| {
            anyhow::anyhow!(
                "Btrfs migration preflight failed: cannot enumerate {}: {}",
                path.display(),
                e
            )
        })? {
            let entry = entry.map_err(|e| {
                anyhow::anyhow!(
                    "Btrfs migration preflight failed: cannot read directory entry under {}: {}",
                    path.display(),
                    e
                )
            })?;
            let child = entry.path();
            let child_metadata = fs::symlink_metadata(&child).map_err(|e| {
                anyhow::anyhow!(
                    "Btrfs migration preflight failed: cannot inspect {}: {}",
                    child.display(),
                    e
                )
            })?;
            if child_metadata.file_type().is_dir() {
                stack.push(child);
            }
        }
    }
    Ok(())
}

fn ioctl_readwrite<T: Default>(fd: RawFd, opcode: libc::Ioctl, name: &str) -> Result<T> {
    let mut arg = T::default();
    ioctl_readwrite_in_place(fd, opcode, &mut arg, name)?;
    Ok(arg)
}

fn ioctl_readwrite_in_place<T>(
    fd: RawFd,
    opcode: libc::Ioctl,
    arg: &mut T,
    name: &str,
) -> Result<()> {
    let ret = unsafe { libc::ioctl(fd, opcode, arg as *mut T) };
    if ret < 0 {
        bail!(
            "Btrfs migration preflight failed: {} on fd {}: {}",
            name,
            fd,
            io::Error::last_os_error()
        );
    }
    Ok(())
}

fn ioctl_subvol_info(fd: RawFd, path: &Path) -> Result<bch_bindgen::c::btrfs_ioctl_get_subvol_info_args> {
    ioctl_readwrite(fd, BTRFS_IOC_GET_SUBVOL_INFO, "BTRFS_IOC_GET_SUBVOL_INFO").map_err(|e| {
        anyhow::anyhow!(
            "Btrfs migration preflight failed: cannot query subvolume {}: {}",
            path.display(),
            e
        )
    })
}

fn validate_incompat_features(flags: u64) -> Result<()> {
    let hard_reject = flags & BTRFS_HARD_REJECT_INCOMPAT;
    if hard_reject != 0 {
        bail!(
            "Btrfs migration unsupported: incompat feature bits 0x{:x} change Chunk Tree or subvolume mapping assumptions",
            hard_reject
        );
    }
    if flags & BTRFS_FEATURE_INCOMPAT_MIXED_GROUPS != 0 {
        bail!(
            "Btrfs migration unsupported: MIXED_GROUPS incompat feature is enabled; Data SINGLE mapping assumptions do not hold"
        );
    }
    let unsupported = flags & !BTRFS_SUPPORTED_INCOMPAT;
    if unsupported != 0 {
        bail!(
            "Btrfs migration unsupported: incompat feature bits 0x{:x} are not in the audited mapping whitelist (including RAID56/RAID1C34/ZONED/EXTENT_TREE_V2/RAID_STRIPE_TREE/REMAP_TREE); refusing unsafe raw mapping",
            unsupported
        );
    }
    Ok(())
}

fn kernel_path(bytes: &[u8]) -> Result<PathBuf> {
    let end = bytes.iter().position(|byte| *byte == 0).unwrap_or(bytes.len());
    if end == 0 {
        bail!("Btrfs migration preflight failed: Btrfs returned an empty device path");
    }
    Ok(PathBuf::from(OsString::from_vec(bytes[..end].to_vec())))
}

/// Btrfs tree keys are ordered lexicographically as one 136-bit key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BtrfsTreeKey {
    objectid: u64,
    ty: u32,
    offset: u64,
}

impl Ord for BtrfsTreeKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.objectid
            .cmp(&other.objectid)
            .then(self.ty.cmp(&other.ty))
            .then(self.offset.cmp(&other.offset))
    }
}

impl PartialOrd for BtrfsTreeKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl BtrfsTreeKey {
    fn in_range(self, min: Self, max: Self) -> bool {
        min <= self && self <= max
    }
}

#[derive(Debug)]
struct SearchRecord {
    objectid: u64,
    offset: u64,
    type_: u32,
    payload: Vec<u8>,
}

fn read_tree_items(
    fd: RawFd,
    tree_id: u64,
    min_objectid: u64,
    max_objectid: u64,
    min_type: u32,
    max_type: u32,
    tree_name: &str,
) -> Result<Vec<SearchRecord>> {
    let mut records = Vec::new();
    let mut key = bch_bindgen::c::btrfs_ioctl_search_key::default();
    key.tree_id = tree_id;
    key.min_objectid = min_objectid;
    key.max_objectid = max_objectid;
    key.min_offset = 0;
    key.max_offset = u64::MAX;
    key.min_transid = 0;
    key.max_transid = u64::MAX;
    key.min_type = min_type;
    key.max_type = max_type;

    loop {
        let request_min = BtrfsTreeKey {
            objectid: key.min_objectid,
            ty: key.min_type,
            offset: key.min_offset,
        };
        let request_max = BtrfsTreeKey {
            objectid: key.max_objectid,
            ty: key.max_type,
            offset: key.max_offset,
        };

        let mut args =
            crate::wrappers::ioctl::IoctlBuf::<bch_bindgen::c::btrfs_ioctl_search_args_v2>::new::<
                u8,
            >(SEARCH_BUFFER_BYTES);
        args.hdr_mut().key = copy_search_key(&key);
        args.hdr_mut().buf_size = SEARCH_BUFFER_BYTES as u64;
        let ret = unsafe { libc::ioctl(fd, BTRFS_IOC_TREE_SEARCH_V2, args.as_mut_ptr()) };
        if ret < 0 {
            bail!(
                "Btrfs migration preflight failed: BTRFS_IOC_TREE_SEARCH_V2 on {} failed: {}",
                tree_name,
                io::Error::last_os_error()
            );
        }

        let returned_nr_items = args.hdr().key.nr_items as usize;
        let used = usize::try_from(args.hdr().buf_size).map_err(|_| {
            anyhow::anyhow!(
                "Btrfs migration preflight failed: {} returned an unrepresentable buffer size",
                tree_name
            )
        })?;
        if used > SEARCH_BUFFER_BYTES {
            bail!(
                "Btrfs migration preflight failed: {} returned {} bytes into a {} byte buffer",
                tree_name,
                used,
                SEARCH_BUFFER_BYTES
            );
        }
        if returned_nr_items == 0 {
            break;
        }
        if used == 0 {
            bail!(
                "Btrfs migration preflight failed: {} reported items but returned no bytes",
                tree_name
            );
        }

        let bytes = args.trailing_bytes(used);
        let mut cursor = 0usize;
        let mut last = None;
        for _ in 0..returned_nr_items {
            let header_end = cursor.checked_add(32).ok_or_else(|| {
                anyhow::anyhow!(
                    "Btrfs migration preflight failed: {} record offset overflow",
                    tree_name
                )
            })?;
            if header_end > bytes.len() {
                bail!(
                    "Btrfs migration preflight failed: truncated {} search header at byte {}",
                    tree_name,
                    cursor
                );
            }
            let objectid = read_le_u64(bytes, cursor + 8)?;
            let offset = read_le_u64(bytes, cursor + 16)?;
            let type_ = read_le_u32(bytes, cursor + 24)?;
            let len = read_le_u32(bytes, cursor + 28)? as usize;
            let payload_start = header_end;
            let record_end = payload_start.checked_add(len).ok_or_else(|| {
                anyhow::anyhow!(
                    "Btrfs migration preflight failed: {} record length overflow",
                    tree_name
                )
            })?;
            if record_end > bytes.len() {
                bail!(
                    "Btrfs migration preflight failed: {} record at byte {} exceeds returned buffer",
                    tree_name,
                    cursor
                );
            }
            let returned_key = BtrfsTreeKey {
                objectid,
                ty: type_,
                offset,
            };
            if !returned_key.in_range(request_min, request_max) {
                bail!(
                    "Btrfs migration preflight failed: {} returned a record outside the search key",
                    tree_name
                );
            }
            records.push(SearchRecord {
                objectid,
                offset,
                type_,
                payload: bytes[payload_start..record_end].to_vec(),
            });
            last = Some((objectid, type_, offset));
            cursor = record_end;
        }

        let Some((objectid, type_, offset)) = last else {
            bail!(
                "Btrfs migration preflight failed: {} returned {} items without records",
                tree_name,
                returned_nr_items
            );
        };
        key.nr_items = 0;
        if offset == u64::MAX {
            if objectid == u64::MAX {
                break;
            }
            key.min_objectid = objectid.checked_add(1).ok_or_else(|| {
                anyhow::anyhow!(
                    "Btrfs migration preflight failed: {} pagination overflow",
                    tree_name
                )
            })?;
            key.min_type = min_type;
            key.min_offset = 0;
        } else {
            key.min_objectid = objectid;
            key.min_type = type_;
            key.min_offset = offset.checked_add(1).ok_or_else(|| {
                anyhow::anyhow!(
                    "Btrfs migration preflight failed: {} pagination overflow",
                    tree_name
                )
            })?;
        }
    }

    Ok(records)
}

fn is_user_subvolume_root_objectid(objectid: u64) -> bool {
    (BTRFS_FIRST_FREE_OBJECTID..=BTRFS_LAST_FREE_OBJECTID).contains(&objectid)
}

fn count_user_subvolume_roots(records: &[SearchRecord]) -> usize {
    let mut objectids = std::collections::HashSet::new();
    for record in records {
        if record.type_ != BTRFS_ROOT_ITEM_KEY as u32
            || !is_user_subvolume_root_objectid(record.objectid)
        {
            continue;
        }
        objectids.insert(record.objectid);
    }
    objectids.len()
}

fn read_subvolume_count(fd: RawFd) -> Result<usize> {
    let records = read_tree_items(
        fd,
        BTRFS_ROOT_TREE_OBJECTID,
        BTRFS_FIRST_FREE_OBJECTID,
        BTRFS_LAST_FREE_OBJECTID,
        BTRFS_ROOT_ITEM_KEY as u32,
        BTRFS_ROOT_ITEM_KEY as u32,
        "Root Tree",
    )?;
    let subvolume_count = count_user_subvolume_roots(&records);
    if subvolume_count != 0 {
        bail!(
            "Btrfs migration unsupported: Root Tree contains {} child subvolume/snapshot root(s)",
            subvolume_count
        );
    }
    Ok(subvolume_count)
}

fn parse_chunk_search_record(record: &SearchRecord) -> Result<ChunkMap> {
    if record.objectid != BTRFS_FIRST_CHUNK_TREE_OBJECTID
        || record.type_ as u64 != BTRFS_CHUNK_ITEM_KEY
    {
        bail!(
            "Btrfs migration preflight failed: invalid Chunk Tree item key ({}, {}, {})",
            record.objectid,
            record.type_,
            record.offset
        );
    }
    // btrfs_chunk keys use objectid=FIRST_CHUNK_TREE_OBJECTID and
    // offset=the chunk's logical start.
    parse_chunk_item(record.offset, &record.payload)
}

fn read_chunk_tree(fd: RawFd) -> Result<Vec<ChunkMap>> {
    let records = read_tree_items(
        fd,
        BTRFS_CHUNK_TREE_OBJECTID,
        BTRFS_FIRST_CHUNK_TREE_OBJECTID,
        u64::MAX,
        BTRFS_CHUNK_ITEM_KEY as u32,
        BTRFS_CHUNK_ITEM_KEY as u32,
        "Chunk Tree",
    )?;
    if records.is_empty() {
        bail!(
            "Btrfs migration preflight failed: Chunk Tree contains no CHUNK_ITEM records"
        );
    }

    let mut chunks = Vec::with_capacity(records.len());
    for record in &records {
        chunks.push(parse_chunk_search_record(record)?);
    }
    Ok(chunks)
}

fn copy_search_key(
    source: &bch_bindgen::c::btrfs_ioctl_search_key,
) -> bch_bindgen::c::btrfs_ioctl_search_key {
    let mut target = bch_bindgen::c::btrfs_ioctl_search_key::default();
    target.tree_id = source.tree_id;
    target.min_objectid = source.min_objectid;
    target.max_objectid = source.max_objectid;
    target.min_offset = source.min_offset;
    target.max_offset = source.max_offset;
    target.min_transid = source.min_transid;
    target.max_transid = source.max_transid;
    target.min_type = source.min_type;
    target.max_type = source.max_type;
    target.nr_items = source.nr_items;
    target.unused = source.unused;
    target.unused1 = source.unused1;
    target.unused2 = source.unused2;
    target.unused3 = source.unused3;
    target.unused4 = source.unused4;
    target
}

fn read_le_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| anyhow::anyhow!("Btrfs migration unsupported: payload offset overflow"))?;
    let value = bytes.get(offset..end).ok_or_else(|| {
        anyhow::anyhow!("Btrfs migration unsupported: truncated little-endian u32 field")
    })?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}


#[cfg(test)]
mod tests {
    use super::*;

    const DATA: u64 = 1 << 0;
    const METADATA: u64 = 1 << 2;
    const RAID0: u64 = 1 << 3;
    const RAID1: u64 = 1 << 4;
    const DUP: u64 = 1 << 5;

    fn chunk(
        logical_start: u64,
        logical_end: u64,
        stripe_offset: u64,
        devid: u64,
        chunk_type: u64,
        num_stripes: u16,
    ) -> ChunkMap {
        ChunkMap {
            logical_start,
            logical_end,
            stripe_offset,
            devid,
            chunk_type,
            num_stripes,
        }
    }

    fn mapper(chunks: Vec<ChunkMap>, device_size: u64) -> BtrfsMapper {
        BtrfsMapper::from_parts(
            [0; 16],
            1,
            PathBuf::from("/dev/test"),
            device_size,
            4096,
            chunks,
        )
        .unwrap()
    }

    #[test]
    fn maps_extent_inside_single_chunk() {
        let m = mapper(vec![chunk(100, 1_100, 10_000, 1, DATA, 1)], 20_000);
        assert_eq!(
            m.map_physical(200, 300).unwrap(),
            vec![PhysicalSegment { source_logical: 200, physical: 10_100, len: 300 }]
        );
    }

    #[test]
    fn maps_extent_ending_at_chunk_end() {
        let m = mapper(vec![chunk(100, 1_100, 10_000, 1, DATA, 1)], 20_000);
        assert_eq!(
            m.map_physical(700, 400).unwrap(),
            vec![PhysicalSegment { source_logical: 700, physical: 10_600, len: 400 }]
        );
    }

    #[test]
    fn splits_extent_across_two_chunks() {
        let m = mapper(
            vec![
                chunk(0, 1_000, 10_000, 1, DATA, 1),
                chunk(1_000, 2_000, 30_000, 1, DATA, 1),
            ],
            40_000,
        );
        assert_eq!(
            m.map_physical(800, 400).unwrap(),
            vec![
                PhysicalSegment { source_logical: 800, physical: 10_800, len: 200 },
                PhysicalSegment { source_logical: 1_000, physical: 30_000, len: 200 },
            ]
        );
    }

    #[test]
    fn splits_extent_across_three_chunks() {
        let m = mapper(
            vec![
                chunk(0, 100, 1_000, 1, DATA, 1),
                chunk(100, 200, 2_000, 1, DATA, 1),
                chunk(200, 300, 3_000, 1, DATA, 1),
            ],
            4_000,
        );
        assert_eq!(
            m.map_physical(50, 200).unwrap(),
            vec![
                PhysicalSegment { source_logical: 50, physical: 1_050, len: 50 },
                PhysicalSegment { source_logical: 100, physical: 2_000, len: 100 },
                PhysicalSegment { source_logical: 200, physical: 3_000, len: 50 },
            ]
        );
    }

    #[test]
    fn rejects_logical_gap() {
        let m = mapper(
            vec![
                chunk(0, 100, 1_000, 1, DATA, 1),
                chunk(200, 300, 2_000, 1, DATA, 1),
            ],
            4_000,
        );
        assert!(m.map_physical(50, 200).is_err());
    }

    #[test]
    fn rejects_logical_before_first_chunk() {
        let m = mapper(vec![chunk(100, 200, 1_000, 1, DATA, 1)], 4_000);
        assert!(m.map_physical(0, 1).is_err());
    }

    #[test]
    fn rejects_logical_length_overflow() {
        let m = mapper(vec![chunk(0, u64::MAX, 0, 1, DATA, 1)], u64::MAX);
        assert!(m.map_physical(u64::MAX - 1, 2).is_err());
    }

    #[test]
    fn rejects_stripe_offset_overflow() {
        let m = mapper(vec![chunk(0, 16, u64::MAX - 4, 1, DATA, 1)], u64::MAX);
        assert!(m.map_physical(8, 8).is_err());
    }

    #[test]
    fn rejects_raw_end_beyond_device() {
        let m = mapper(vec![chunk(0, 100, 900, 1, DATA, 1)], 999);
        assert!(m.map_physical(0, 100).is_err());
    }

    #[test]
    fn rejects_wrong_device_id() {
        let m = BtrfsMapper::from_parts(
            [0; 16],
            1,
            PathBuf::from("/dev/test"),
            4_000,
            4096,
            vec![chunk(0, 100, 1_000, 2, DATA, 1)],
        );
        assert!(m.is_err());
    }

    #[test]
    fn rejects_data_raid1() {
        assert!(BtrfsMapper::from_parts(
            [0; 16], 1, PathBuf::from("/dev/test"), 4_000, 4096,
            vec![chunk(0, 100, 1_000, 1, DATA | RAID1, 2)],
        ).is_err());
    }

    #[test]
    fn rejects_data_raid0() {
        assert!(BtrfsMapper::from_parts(
            [0; 16], 1, PathBuf::from("/dev/test"), 4_000, 4096,
            vec![chunk(0, 100, 1_000, 1, DATA | RAID0, 2)],
        ).is_err());
    }

    #[test]
    fn rejects_data_dup() {
        assert!(BtrfsMapper::from_parts(
            [0; 16], 1, PathBuf::from("/dev/test"), 4_000, 4096,
            vec![chunk(0, 100, 1_000, 1, DATA | DUP, 2)],
        ).is_err());
    }

    #[test]
    fn allows_metadata_dup() {
        let m = mapper(vec![chunk(0, 100, 1_000, 1, METADATA | DUP, 2)], 4_000);
        assert!(m.map_physical(0, 1).is_err());
    }

    #[test]
    fn rejects_truncated_chunk_item() {
        assert!(parse_chunk_item(0, &[0; 47]).is_err());
    }

    #[test]
    fn rejects_stripe_count_payload_mismatch() {
        let mut payload = vec![0u8; 48 + 32];
        payload[0..8].copy_from_slice(&100u64.to_le_bytes());
        payload[40..44].copy_from_slice(&4096u32.to_le_bytes());
        payload[44..46].copy_from_slice(&2u16.to_le_bytes());
        assert!(parse_chunk_item(0, &payload).is_err());
    }

    #[test]
    fn parses_chunk_payload_as_little_endian() {
        let mut payload = vec![0u8; 48 + 32];
        payload[0..8].copy_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes());
        payload[8..16].copy_from_slice(&0x1112_1314_1516_1718u64.to_le_bytes());
        payload[24..32].copy_from_slice(&DATA.to_le_bytes());
        payload[40..44].copy_from_slice(&4096u32.to_le_bytes());
        payload[44..46].copy_from_slice(&1u16.to_le_bytes());
        payload[48..56].copy_from_slice(&1u64.to_le_bytes());
        payload[56..64].copy_from_slice(&0x2122_2324_2526_2728u64.to_le_bytes());
        let parsed = parse_chunk_item(0x3031, &payload).unwrap();
        assert_eq!(parsed.logical_start, 0x3031);
        assert_eq!(parsed.logical_end, 0x0102_0304_0506_0708 + 0x3031);
        assert_eq!(parsed.stripe_offset, 0x2122_2324_2526_2728);
        assert_eq!(parsed.devid, 1);
        assert_eq!(parsed.chunk_type, DATA);
    }

    #[test]
    fn chunk_search_key_uses_offset_as_logical_start() {
        let mut payload = vec![0u8; 48 + 32];
        payload[0..8].copy_from_slice(&4096u64.to_le_bytes());
        payload[24..32].copy_from_slice(&DATA.to_le_bytes());
        payload[40..44].copy_from_slice(&4096u32.to_le_bytes());
        payload[44..46].copy_from_slice(&1u16.to_le_bytes());
        payload[48..56].copy_from_slice(&1u64.to_le_bytes());
        payload[56..64].copy_from_slice(&8192u64.to_le_bytes());
        let record = SearchRecord {
            objectid: BTRFS_FIRST_CHUNK_TREE_OBJECTID,
            offset: 0x1234,
            type_: BTRFS_CHUNK_ITEM_KEY as u32,
            payload,
        };
        assert_eq!(
            parse_chunk_search_record(&record).unwrap().logical_start,
            0x1234
        );
    }

    #[test]
    fn rejects_zero_length_chunk() {
        let mut payload = vec![0u8; 48 + 32];
        payload[40..44].copy_from_slice(&4096u32.to_le_bytes());
        payload[44..46].copy_from_slice(&1u16.to_le_bytes());
        assert!(parse_chunk_item(0, &payload).is_err());
    }

    #[test]
    fn rejects_overlapping_chunk_entries() {
        let result = BtrfsMapper::from_parts(
            [0; 16], 1, PathBuf::from("/dev/test"), 4_000, 4096,
            vec![
                chunk(0, 200, 1_000, 1, DATA, 1),
                chunk(100, 300, 2_000, 1, DATA, 1),
            ],
        );
        assert!(result.is_err());
    }

    #[test]
    fn sorts_unsorted_source_entries_before_mapping() {
        let m = mapper(
            vec![
                chunk(100, 200, 2_000, 1, DATA, 1),
                chunk(0, 100, 1_000, 1, DATA, 1),
            ],
            4_000,
        );
        assert_eq!(
            m.map_physical(50, 100).unwrap(),
            vec![
                PhysicalSegment { source_logical: 50, physical: 1_050, len: 50 },
                PhysicalSegment { source_logical: 100, physical: 2_000, len: 50 },
            ]
        );
    }
    #[test]
    fn maps_single_device_fixture_chunk_boundary() {
        let mapper = mapper(
            vec![
                chunk(298_844_160, 1_372_585_984, 575_668_224, 1, DATA, 1),
                chunk(1_372_585_984, 2_445_327_808, 1_649_410_048, 1, DATA, 1),
            ],
            17_179_869_184,
        );
        assert_eq!(
            mapper.map_physical(1_372_581_888, 8192).unwrap(),
            vec![
                PhysicalSegment {
                    source_logical: 1_372_581_888,
                    physical: 1_649_405_952,
                    len: 4096,
                },
                PhysicalSegment {
                    source_logical: 1_372_585_984,
                    physical: 1_649_410_048,
                    len: 4096,
                },
            ]
        );
    }

    #[test]
    fn identity_mapper_keeps_physical_address() {
        let mapper = ExtentMapper::Identity;
        assert_eq!(
            mapper.map_physical(4096, 8192).unwrap(),
            vec![PhysicalSegment {
                source_logical: 4096,
                physical: 4096,
                len: 8192,
            }]
        );
    }

    #[test]
    fn rejects_mapping_features_outside_audited_whitelist() {
        assert!(validate_incompat_features(BTRFS_FEATURE_INCOMPAT_RAID56).is_err());
        assert!(validate_incompat_features(BTRFS_FEATURE_INCOMPAT_EXTENT_TREE_V2).is_err());
        assert!(validate_incompat_features(1u64 << 63).is_err());
        assert!(validate_incompat_features(BTRFS_FEATURE_INCOMPAT_MIXED_GROUPS).is_err());
        assert!(validate_incompat_features(BTRFS_FEATURE_INCOMPAT_NO_HOLES).is_ok());
    }
    #[test]
    fn classifies_allocatable_subvolume_root_objectids() {
        assert!(is_user_subvolume_root_objectid(256));
        assert!(is_user_subvolume_root_objectid(257));
        assert!(is_user_subvolume_root_objectid(BTRFS_LAST_FREE_OBJECTID));
        assert!(!is_user_subvolume_root_objectid(BTRFS_FS_TREE_OBJECTID));
        assert!(!is_user_subvolume_root_objectid((-9i64) as u64));
        assert!(!is_user_subvolume_root_objectid((-255i64) as u64));
    }

    const ROOT_BACKREF_KEY: u32 = BTRFS_ROOT_ITEM_KEY as u32 + 12;
    const ROOT_REF_KEY: u32 = BTRFS_ROOT_ITEM_KEY as u32 + 24;

    fn root_search_bounds() -> (BtrfsTreeKey, BtrfsTreeKey) {
        (
            BtrfsTreeKey {
                objectid: BTRFS_FIRST_FREE_OBJECTID,
                ty: BTRFS_ROOT_ITEM_KEY as u32,
                offset: 0,
            },
            BtrfsTreeKey {
                objectid: BTRFS_LAST_FREE_OBJECTID,
                ty: BTRFS_ROOT_ITEM_KEY as u32,
                offset: u64::MAX,
            },
        )
    }

    fn root_record(objectid: u64, type_: u32) -> SearchRecord {
        SearchRecord {
            objectid,
            offset: 0,
            type_,
            payload: vec![0xff],
        }
    }

    #[test]
    fn root_search_min_key_is_in_composite_range() {
        let (min, max) = root_search_bounds();
        assert!(BtrfsTreeKey {
            objectid: BTRFS_FIRST_FREE_OBJECTID,
            ty: BTRFS_ROOT_ITEM_KEY as u32,
            offset: 0,
        }
        .in_range(min, max));
    }

    #[test]
    fn root_backref_at_first_objectid_is_in_composite_range() {
        let (min, max) = root_search_bounds();
        assert!(BtrfsTreeKey {
            objectid: BTRFS_FIRST_FREE_OBJECTID,
            ty: ROOT_BACKREF_KEY,
            offset: 5,
        }
        .in_range(min, max));
    }

    #[test]
    fn higher_objectid_dominates_key_type() {
        let (min, max) = root_search_bounds();
        assert!(BtrfsTreeKey {
            objectid: BTRFS_FIRST_FREE_OBJECTID + 1,
            ty: 0,
            offset: 0,
        }
        .in_range(min, max));
    }

    #[test]
    fn objectid_below_min_is_out_of_composite_range() {
        let (min, max) = root_search_bounds();
        assert!(!BtrfsTreeKey {
            objectid: BTRFS_FIRST_FREE_OBJECTID - 1,
            ty: u32::MAX,
            offset: u64::MAX,
        }
        .in_range(min, max));
    }

    #[test]
    fn root_search_max_key_is_in_composite_range() {
        let (min, max) = root_search_bounds();
        assert!(BtrfsTreeKey {
            objectid: BTRFS_LAST_FREE_OBJECTID,
            ty: BTRFS_ROOT_ITEM_KEY as u32,
            offset: u64::MAX,
        }
        .in_range(min, max));
    }

    #[test]
    fn higher_type_at_max_objectid_is_out_of_composite_range() {
        let (min, max) = root_search_bounds();
        assert!(!BtrfsTreeKey {
            objectid: BTRFS_LAST_FREE_OBJECTID,
            ty: ROOT_BACKREF_KEY,
            offset: 0,
        }
        .in_range(min, max));
    }

    #[test]
    fn data_reloc_objectid_is_outside_user_root_interval() {
        let (min, max) = root_search_bounds();
        assert!(!BtrfsTreeKey {
            objectid: (-9i64) as u64,
            ty: BTRFS_ROOT_ITEM_KEY as u32,
            offset: 0,
        }
        .in_range(min, max));
    }

    #[test]
    fn multiple_objectids_is_outside_user_root_interval() {
        let (min, max) = root_search_bounds();
        assert!(!BtrfsTreeKey {
            objectid: (-255i64) as u64,
            ty: BTRFS_ROOT_ITEM_KEY as u32,
            offset: 0,
        }
        .in_range(min, max));
    }

    #[test]
    fn root_tree_count_ignores_non_root_items_and_special_roots() {
        let records = [
            root_record(BTRFS_FIRST_FREE_OBJECTID, BTRFS_ROOT_ITEM_KEY as u32),
            root_record(BTRFS_FIRST_FREE_OBJECTID, ROOT_BACKREF_KEY),
            root_record(BTRFS_FIRST_FREE_OBJECTID, ROOT_REF_KEY),
            root_record(BTRFS_FIRST_FREE_OBJECTID, BTRFS_ROOT_ITEM_KEY as u32),
            root_record((-9i64) as u64, BTRFS_ROOT_ITEM_KEY as u32),
            root_record((-255i64) as u64, BTRFS_ROOT_ITEM_KEY as u32),
        ];

        assert_eq!(count_user_subvolume_roots(&records), 1);
    }
}
