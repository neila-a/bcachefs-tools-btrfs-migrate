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
    RawUnalignedHead,
    RawUnalignedTail,
    RawUnalignedUnsalvageable,
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
            Self::RawUnalignedHead => "raw_unaligned_head",
            Self::RawUnalignedTail => "raw_unaligned_tail",
            Self::RawUnalignedUnsalvageable => "raw_unaligned_unsalvageable",
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RawSegmentDisposition {
    Copy(CopyReason),
    Link,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RawSegmentSpan {
    pub file_offset: u64,
    pub physical_offset: u64,
    pub len: u64,
    pub disposition: RawSegmentDisposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RawSegmentPlan {
    spans: [RawSegmentSpan; 4],
    len: usize,
    pub raw_link_salvaged_bytes: u64,
}

const EMPTY_RAW_SEGMENT_SPAN: RawSegmentSpan = RawSegmentSpan {
    file_offset: 0,
    physical_offset: 0,
    len: 0,
    disposition: RawSegmentDisposition::Copy(CopyReason::Other),
};

impl Default for RawSegmentPlan {
    fn default() -> Self {
        Self {
            spans: [EMPTY_RAW_SEGMENT_SPAN; 4],
            len: 0,
            raw_link_salvaged_bytes: 0,
        }
    }
}

impl RawSegmentPlan {
    pub(crate) fn spans(&self) -> &[RawSegmentSpan] {
        &self.spans[..self.len]
    }

    fn push(&mut self, span: RawSegmentSpan) {
        if span.len == 0 {
            return;
        }
        debug_assert!(self.len < self.spans.len());
        self.spans[self.len] = span;
        self.len += 1;
    }

    fn copy(&mut self, file_offset: u64, physical_offset: u64, len: u64, reason: CopyReason) {
        self.push(RawSegmentSpan {
            file_offset,
            physical_offset,
            len,
            disposition: RawSegmentDisposition::Copy(reason),
        });
    }

    fn link(&mut self, file_offset: u64, physical_offset: u64, len: u64) {
        self.push(RawSegmentSpan {
            file_offset,
            physical_offset,
            len,
            disposition: RawSegmentDisposition::Link,
        });
    }
}

pub(crate) fn plan_raw_segment(
    file_offset: u64,
    physical_offset: u64,
    len: u64,
    i_size: u64,
    policy: MigrationCopyPolicy,
) -> Result<RawSegmentPlan, &'static str> {
    let file_end = file_offset
        .checked_add(len)
        .ok_or("raw segment file range overflows")?;
    if file_end > i_size {
        return Err("raw segment crosses i_size");
    }
    physical_offset
        .checked_add(len)
        .ok_or("raw segment physical range overflows")?;

    let mut plan = RawSegmentPlan::default();
    if len == 0 {
        return Ok(plan);
    }

    let mut copy_file_offset = file_offset;
    let mut copy_physical_offset = physical_offset;
    let mut remaining = len;
    if copy_physical_offset < policy.reserved_superblock_end {
        let reserved_len = policy
            .reserved_superblock_end
            .checked_sub(copy_physical_offset)
            .unwrap_or(0)
            .min(remaining);
        plan.copy(
            copy_file_offset,
            copy_physical_offset,
            reserved_len,
            CopyReason::ReservationOverlap,
        );
        copy_file_offset = copy_file_offset
            .checked_add(reserved_len)
            .ok_or("raw segment file range overflows")?;
        copy_physical_offset = copy_physical_offset
            .checked_add(reserved_len)
            .ok_or("raw segment physical range overflows")?;
        remaining -= reserved_len;
    }
    if remaining == 0 {
        return Ok(plan);
    }

    let block_bytes = policy.block_bytes;
    if !block_bytes.is_power_of_two()
        || (copy_file_offset ^ copy_physical_offset) & (block_bytes - 1) != 0
    {
        plan.copy(
            copy_file_offset,
            copy_physical_offset,
            remaining,
            CopyReason::RawUnalignedUnsalvageable,
        );
        return Ok(plan);
    }

    let link_file_offset = copy_file_offset
        .checked_add(block_bytes - 1)
        .ok_or("raw segment alignment overflows")?
        & !(block_bytes - 1);
    let head_len = link_file_offset
        .checked_sub(copy_file_offset)
        .ok_or("raw segment alignment underflows")?;
    if head_len >= remaining {
        plan.copy(
            copy_file_offset,
            copy_physical_offset,
            remaining,
            CopyReason::RawUnalignedUnsalvageable,
        );
        return Ok(plan);
    }

    let link_len = ((remaining - head_len) / block_bytes) * block_bytes;
    if link_len == 0 {
        plan.copy(
            copy_file_offset,
            copy_physical_offset,
            remaining,
            CopyReason::RawUnalignedUnsalvageable,
        );
        return Ok(plan);
    }

    if head_len != 0 {
        plan.copy(
            copy_file_offset,
            copy_physical_offset,
            head_len,
            CopyReason::RawUnalignedHead,
        );
    }
    let link_physical_offset = copy_physical_offset
        .checked_add(head_len)
        .ok_or("raw segment physical range overflows")?;
    plan.link(link_file_offset, link_physical_offset, link_len);
    let tail_len = remaining - head_len - link_len;
    if tail_len != 0 {
        plan.copy(
            link_file_offset
                .checked_add(link_len)
                .ok_or("raw segment file range overflows")?,
            link_physical_offset
                .checked_add(link_len)
                .ok_or("raw segment physical range overflows")?,
            tail_len,
            CopyReason::RawUnalignedTail,
        );
    }
    if head_len != 0 || tail_len != 0 {
        plan.raw_link_salvaged_bytes = link_len;
    }

    Ok(plan)
}

pub(crate) fn copy_write_bytes(bytes: u64, block_bytes: u64) -> Result<u64, &'static str> {
    if bytes == 0 {
        return Ok(0);
    }
    if !block_bytes.is_power_of_two() {
        return Err("copy block size is not a power of two");
    }

    let max_io_bytes = bch_bindgen::data::io::MAX_IO_SIZE as u64;
    let full_chunks = bytes / max_io_bytes;
    let tail = bytes % max_io_bytes;
    let full_chunk_bytes = max_io_bytes
        .div_ceil(block_bytes)
        .checked_mul(block_bytes)
        .ok_or("copy write byte count overflows")?;
    let full_bytes = full_chunks
        .checked_mul(full_chunk_bytes)
        .ok_or("copy write byte count overflows")?;
    let tail_bytes = if tail == 0 {
        0
    } else {
        tail.div_ceil(block_bytes)
            .checked_mul(block_bytes)
            .ok_or("copy write byte count overflows")?
    };
    full_bytes
        .checked_add(tail_bytes)
        .ok_or("copy write byte count overflows")
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct CopyAccounting {
    pub copied_bytes: u64,
    pub allocated_copy_bytes: u64,
    pub shared: u64,
    pub inline: u64,
    pub encoded: u64,
    pub encrypted: u64,
    pub not_aligned: u64,
    pub tail: u64,
    pub merged: u64,
    pub future_flags: u64,
    pub reservation_overlap: u64,
    pub raw_unaligned_head: u64,
    pub raw_unaligned_tail: u64,
    pub raw_unaligned_unsalvageable: u64,
    pub raw_mapping_unavailable: u64,
    pub unstable: u64,
    pub symlink: u64,
    pub other: u64,
}

impl CopyAccounting {

    pub(crate) fn record_with_allocation(
        &mut self,
        reason: CopyReason,
        bytes: u64,
        allocated_bytes: u64,
    ) {
        self.copied_bytes = self.copied_bytes.saturating_add(bytes);
        self.allocated_copy_bytes = self.allocated_copy_bytes.saturating_add(allocated_bytes);
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
            CopyReason::RawUnalignedHead => {
                self.raw_unaligned_head = self.raw_unaligned_head.saturating_add(bytes)
            }
            CopyReason::RawUnalignedTail => {
                self.raw_unaligned_tail = self.raw_unaligned_tail.saturating_add(bytes)
            }
            CopyReason::RawUnalignedUnsalvageable => {
                self.raw_unaligned_unsalvageable =
                    self.raw_unaligned_unsalvageable.saturating_add(bytes)
            }
            CopyReason::RawMappingUnavailable => {
                self.raw_mapping_unavailable = self.raw_mapping_unavailable.saturating_add(bytes)
            }
            CopyReason::Symlink => self.symlink = self.symlink.saturating_add(bytes),
            CopyReason::Other => self.other = self.other.saturating_add(bytes),
        }
    }

    pub(crate) fn raw_unaligned_bytes(&self) -> u64 {
        self.raw_unaligned_head
            .saturating_add(self.raw_unaligned_tail)
            .saturating_add(self.raw_unaligned_unsalvageable)
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
            .saturating_add(self.raw_unaligned_bytes())
            .saturating_add(self.raw_mapping_unavailable)
            .saturating_add(self.unstable)
            .saturating_add(self.symlink)
            .saturating_add(self.other)
    }

    pub(crate) fn is_consistent(&self) -> bool {
        self.copied_bytes == self.reason_sum() && self.allocated_copy_bytes >= self.copied_bytes
    }

    pub(crate) fn reason_counts(&self) -> [(CopyReason, u64); 15] {
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
            (CopyReason::RawUnalignedHead, self.raw_unaligned_head),
            (CopyReason::RawUnalignedTail, self.raw_unaligned_tail),
            (
                CopyReason::RawUnalignedUnsalvageable,
                self.raw_unaligned_unsalvageable,
            ),
            (CopyReason::RawMappingUnavailable, self.raw_mapping_unavailable),
            (CopyReason::Symlink, self.symlink),
            (CopyReason::Other, self.other),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> MigrationCopyPolicy {
        MigrationCopyPolicy {
            block_bytes: 4096,
            reserved_superblock_end: 0,
        }
    }

    #[test]
    fn salvages_the_aligned_middle_of_a_residue_matched_segment() {
        let plan = plan_raw_segment(512, 8704, 12_288, 12_800, policy()).unwrap();

        assert_eq!(
            plan.spans(),
            &[
                RawSegmentSpan {
                    file_offset: 512,
                    physical_offset: 8704,
                    len: 3584,
                    disposition: RawSegmentDisposition::Copy(CopyReason::RawUnalignedHead),
                },
                RawSegmentSpan {
                    file_offset: 4096,
                    physical_offset: 12_288,
                    len: 8192,
                    disposition: RawSegmentDisposition::Link,
                },
                RawSegmentSpan {
                    file_offset: 12_288,
                    physical_offset: 20_480,
                    len: 512,
                    disposition: RawSegmentDisposition::Copy(CopyReason::RawUnalignedTail),
                },
            ]
        );
        assert_eq!(plan.raw_link_salvaged_bytes, 8192);
    }

    #[test]
    fn refuses_to_link_mismatched_logical_and_physical_residues() {
        let plan = plan_raw_segment(512, 8192, 8192, 8704, policy()).unwrap();

        assert_eq!(
            plan.spans(),
            &[RawSegmentSpan {
                file_offset: 512,
                physical_offset: 8192,
                len: 8192,
                disposition: RawSegmentDisposition::Copy(
                    CopyReason::RawUnalignedUnsalvageable,
                ),
            }]
        );
        assert_eq!(plan.raw_link_salvaged_bytes, 0);
    }

    #[test]
    fn keeps_reserved_raw_bytes_out_of_an_otherwise_linkable_segment() {
        let policy = MigrationCopyPolicy {
            block_bytes: 4096,
            reserved_superblock_end: 8192,
        };
        let plan = plan_raw_segment(0, 0, 16_384, 16_384, policy).unwrap();

        assert_eq!(
            plan.spans(),
            &[
                RawSegmentSpan {
                    file_offset: 0,
                    physical_offset: 0,
                    len: 8192,
                    disposition: RawSegmentDisposition::Copy(CopyReason::ReservationOverlap),
                },
                RawSegmentSpan {
                    file_offset: 8192,
                    physical_offset: 8192,
                    len: 8192,
                    disposition: RawSegmentDisposition::Link,
                },
            ]
        );
        assert_eq!(plan.raw_link_salvaged_bytes, 0);
    }

    #[test]
    fn rejects_a_segment_that_would_cross_i_size() {
        assert!(plan_raw_segment(4096, 4096, 4096, 4095, policy()).is_err());
    }

    #[test]
    fn reserves_a_full_target_block_for_each_partial_copy_write() {
        let max_io_bytes = bch_bindgen::data::io::MAX_IO_SIZE as u64;

        assert_eq!(copy_write_bytes(4097, 4096), Ok(8192));
        assert_eq!(
            copy_write_bytes(max_io_bytes + 1, 4096),
            Ok(max_io_bytes + 4096)
        );
        assert_eq!(
            copy_write_bytes(max_io_bytes + 1, max_io_bytes * 2),
            Ok(max_io_bytes * 4)
        );
    }
}
