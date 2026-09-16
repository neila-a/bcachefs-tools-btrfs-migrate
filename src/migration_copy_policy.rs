use fiemap::FiemapExtentFlags;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CopyReason {
    Shared,
    Inline,
    Encoded,
    Encrypted,
    NotAligned,
    Tail,
    Merged,
    FutureFlags,
    ReservationOverlap,
    RawUnaligned,
    RawMappingUnavailable,
    Symlink,
    Other,
}

impl CopyReason {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Inline => "inline",
            Self::Encoded => "encoded",
            Self::Encrypted => "encrypted",
            Self::NotAligned => "not_aligned",
            Self::Tail => "tail",
            Self::Merged => "merged",
            Self::FutureFlags => "future_flags",
            Self::ReservationOverlap => "reservation_overlap",
            Self::RawUnaligned => "raw_unaligned",
            Self::RawMappingUnavailable => "raw_mapping_unavailable",
            Self::Symlink => "symlink",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExtentDisposition {
    Link,
    Hole,
    Unstable,
    Copy(CopyReason),
}

pub(crate) fn classify_extent(flags: FiemapExtentFlags) -> ExtentDisposition {
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

    if flags.intersects(FiemapExtentFlags::UNKNOWN | FiemapExtentFlags::DELALLOC) {
        ExtentDisposition::Unstable
    } else if flags.contains(FiemapExtentFlags::UNWRITTEN) {
        ExtentDisposition::Hole
    } else if flags.contains(FiemapExtentFlags::ENCODED) {
        ExtentDisposition::Copy(CopyReason::Encoded)
    } else if flags.contains(FiemapExtentFlags::DATA_ENCRYPTED) {
        ExtentDisposition::Copy(CopyReason::Encrypted)
    } else if flags.contains(FiemapExtentFlags::SHARED) {
        ExtentDisposition::Copy(CopyReason::Shared)
    } else if flags.contains(FiemapExtentFlags::DATA_INLINE) {
        ExtentDisposition::Copy(CopyReason::Inline)
    } else if flags.contains(FiemapExtentFlags::DATA_TAIL) {
        ExtentDisposition::Copy(CopyReason::Tail)
    } else if flags.contains(FiemapExtentFlags::NOT_ALIGNED) {
        ExtentDisposition::Copy(CopyReason::NotAligned)
    } else if flags.contains(FiemapExtentFlags::MERGED) {
        ExtentDisposition::Copy(CopyReason::Merged)
    } else if flags.bits() & !known.bits() != 0 {
        ExtentDisposition::Copy(CopyReason::FutureFlags)
    } else {
        ExtentDisposition::Link
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MigrationCopyPolicy {
    pub block_bytes: u64,
    pub reserved_superblock_end: u64,
}

pub(crate) fn classify_segment(
    file_offset: u64,
    physical_offset: u64,
    len: u64,
    policy: MigrationCopyPolicy,
) -> Option<CopyReason> {
    if len != 0 && physical_offset < policy.reserved_superblock_end {
        Some(CopyReason::ReservationOverlap)
    } else if !policy.block_bytes.is_power_of_two()
        || (file_offset | physical_offset | len) & (policy.block_bytes - 1) != 0
    {
        Some(CopyReason::RawUnaligned)
    } else {
        None
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct CopyAccounting {
    pub copied_bytes: u64,
    pub shared: u64,
    pub inline: u64,
    pub encoded: u64,
    pub encrypted: u64,
    pub not_aligned: u64,
    pub tail: u64,
    pub merged: u64,
    pub future_flags: u64,
    pub reservation_overlap: u64,
    pub raw_unaligned: u64,
    pub raw_mapping_unavailable: u64,
    pub unstable: u64,
    pub symlink: u64,
    pub other: u64,
}

impl CopyAccounting {
    pub(crate) fn record(&mut self, reason: CopyReason, bytes: u64) {
        self.copied_bytes = self.copied_bytes.saturating_add(bytes);
        match reason {
            CopyReason::Shared => self.shared = self.shared.saturating_add(bytes),
            CopyReason::Inline => self.inline = self.inline.saturating_add(bytes),
            CopyReason::Encoded => self.encoded = self.encoded.saturating_add(bytes),
            CopyReason::Encrypted => self.encrypted = self.encrypted.saturating_add(bytes),
            CopyReason::NotAligned => self.not_aligned = self.not_aligned.saturating_add(bytes),
            CopyReason::Tail => self.tail = self.tail.saturating_add(bytes),
            CopyReason::Merged => self.merged = self.merged.saturating_add(bytes),
            CopyReason::FutureFlags => self.future_flags = self.future_flags.saturating_add(bytes),
            CopyReason::ReservationOverlap => {
                self.reservation_overlap = self.reservation_overlap.saturating_add(bytes)
            }
            CopyReason::RawUnaligned => {
                self.raw_unaligned = self.raw_unaligned.saturating_add(bytes)
            }
            CopyReason::RawMappingUnavailable => {
                self.raw_mapping_unavailable = self.raw_mapping_unavailable.saturating_add(bytes)
            }
            CopyReason::Symlink => self.symlink = self.symlink.saturating_add(bytes),
            CopyReason::Other => self.other = self.other.saturating_add(bytes),
        }
    }

    pub(crate) fn reason_sum(&self) -> u64 {
        self.shared
            .saturating_add(self.inline)
            .saturating_add(self.encoded)
            .saturating_add(self.encrypted)
            .saturating_add(self.not_aligned)
            .saturating_add(self.tail)
            .saturating_add(self.merged)
            .saturating_add(self.future_flags)
            .saturating_add(self.reservation_overlap)
            .saturating_add(self.raw_unaligned)
            .saturating_add(self.raw_mapping_unavailable)
            .saturating_add(self.unstable)
            .saturating_add(self.symlink)
            .saturating_add(self.other)
    }

    pub(crate) fn is_consistent(&self) -> bool {
        self.copied_bytes == self.reason_sum()
    }

    pub(crate) fn reason_counts(&self) -> [(CopyReason, u64); 13] {
        [
            (CopyReason::Shared, self.shared),
            (CopyReason::Inline, self.inline),
            (CopyReason::Encoded, self.encoded),
            (CopyReason::Encrypted, self.encrypted),
            (CopyReason::NotAligned, self.not_aligned),
            (CopyReason::Tail, self.tail),
            (CopyReason::Merged, self.merged),
            (CopyReason::FutureFlags, self.future_flags),
            (CopyReason::ReservationOverlap, self.reservation_overlap),
            (CopyReason::RawUnaligned, self.raw_unaligned),
            (CopyReason::RawMappingUnavailable, self.raw_mapping_unavailable),
            (CopyReason::Symlink, self.symlink),
            (CopyReason::Other, self.other),
        ]
    }
}
