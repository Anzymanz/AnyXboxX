use super::{OnceLock, Step};
use std::ffi::OsStr;
use std::ops::Range;
use std::sync::atomic::AtomicU64;

pub(in crate::host) static FORWARDED: AtomicU64 = AtomicU64::new(0);

pub(in crate::host) fn fill_after_copy() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| enabled(std::env::var_os("VICTORY_FILL_AFTER_COPY").as_deref()))
}

fn enabled(value: Option<&OsStr>) -> bool {
    value == Some(OsStr::new("1"))
}

pub(in crate::host) fn forward(steps: &[Step], src: usize, len: usize) -> Option<usize> {
    fill_after_copy().then_some(())?;
    forwarded(steps.iter().rev().map(Seen::from), src, len)
}

/// List steps projected onto the ranges relevant to forwarding.
enum Seen {
    Placed {
        dst: usize,
        src: usize,
        len: usize,
        upload: Option<usize>,
    },
    Writes(usize, usize),
    Anything,
    Nothing,
}

impl From<&Step> for Seen {
    fn from(step: &Step) -> Self {
        match *step {
            Step::Placed(ref p) => Self::Placed {
                dst: p.dst,
                src: p.src,
                len: p.len,
                upload: p.from,
            },
            Step::Copy(dst, _, len)
            | Step::GpuWrite(_, dst, len)
            | Step::Fill(dst, _, len, _)
            | Step::ForwardedFill(dst, _, len) => Self::Writes(dst, len),
            Step::Run(_) => Self::Anything,
            Step::Refresh(..) | Step::Host(..) | Step::Texture(..) | Step::Plane(..) => {
                Self::Nothing
            }
        }
    }
}

fn overlaps(a: &Range<usize>, b: &Range<usize>) -> bool {
    !a.is_empty() && !b.is_empty() && a.start < b.end && b.start < a.end
}

/// Finds the upload address for a fill by walking the list backwards.
fn forwarded(steps: impl IntoIterator<Item = Seen>, src: usize, len: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let fill = src..src.checked_add(len)?;
    let mut later = Vec::new();
    for step in steps {
        let written = match step {
            Seen::Placed {
                dst,
                src,
                len,
                upload,
            } => {
                let dest = dst..dst.checked_add(len)?;
                if overlaps(&dest, &fill) {
                    let source = src..src.checked_add(len)?;
                    let upload = upload?;
                    upload.checked_add(len)?;
                    if dest.start > fill.start
                        || dest.end < fill.end
                        || overlaps(&source, &dest)
                        || later
                            .iter()
                            .any(|r| overlaps(r, &source) || overlaps(r, &dest))
                    {
                        return None;
                    }
                    return Some(upload + (fill.start - dst));
                }
                dest
            }
            Seen::Writes(at, len) => at..at.checked_add(len)?,
            Seen::Anything => return None,
            Seen::Nothing => continue,
        };
        if overlaps(&written, &fill) {
            return None;
        }
        if !written.is_empty() {
            later.push(written);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placed(dst: usize, src: usize, len: usize) -> Seen {
        Seen::Placed {
            dst,
            src,
            len,
            upload: Some(300),
        }
    }

    #[test]
    fn whole_and_offset() {
        assert_eq!(forwarded([placed(100, 200, 20)], 100, 20), Some(300));
        assert_eq!(forwarded([placed(100, 200, 20)], 105, 10), Some(305));
    }

    #[test]
    fn unrelated_steps_allow_forwarding() {
        let steps = [
            Seen::Nothing,
            Seen::Writes(120, 80),
            Seen::Writes(205, 0),
            placed(100, 200, 20),
        ];
        assert_eq!(forwarded(steps, 105, 10), Some(305));
    }

    #[test]
    fn missing_or_partial_copy() {
        assert_eq!(forwarded([], 100, 20), None);
        assert_eq!(forwarded([placed(50, 200, 20)], 100, 20), None);
        assert_eq!(forwarded([placed(105, 200, 10)], 100, 20), None);
    }

    #[test]
    fn host_copy_and_overlapping_copy() {
        let host = Seen::Placed {
            dst: 100,
            src: 200,
            len: 20,
            upload: None,
        };
        assert_eq!(forwarded([host], 100, 20), None);
        assert_eq!(forwarded([placed(100, 100, 20)], 100, 20), None);
        assert_eq!(forwarded([placed(100, 110, 20)], 100, 20), None);
    }

    #[test]
    fn later_writes_to_either_range() {
        // These writes miss the fill itself.
        for address in [100, 115, 200, 215] {
            assert_eq!(
                forwarded([Seen::Writes(address, 1), placed(100, 200, 20)], 105, 10),
                None
            );
        }
        assert_eq!(
            forwarded([Seen::Writes(108, 1), placed(100, 200, 20)], 105, 10),
            None
        );
        assert_eq!(
            forwarded([placed(200, 400, 5), placed(100, 200, 20)], 105, 10),
            None
        );
    }

    #[test]
    fn anything_stops_look_back() {
        assert_eq!(
            forwarded([Seen::Anything, placed(100, 200, 20)], 100, 20),
            None
        );
    }

    #[test]
    fn last_copy_only_writes_part() {
        assert_eq!(
            forwarded([placed(110, 400, 5), placed(100, 200, 20)], 100, 20),
            None
        );
    }

    #[test]
    fn empty_and_overflowing_ranges() {
        assert_eq!(forwarded([placed(100, 200, 20)], 100, 0), None);
        assert_eq!(forwarded([placed(100, 200, 20)], usize::MAX, 1), None);
        assert_eq!(forwarded([placed(100, usize::MAX, 20)], 100, 20), None);
        assert_eq!(
            forwarded([placed(usize::MAX - 1, 200, 20)], usize::MAX - 1, 1),
            None
        );
        let bad_upload = Seen::Placed {
            dst: 100,
            src: 200,
            len: 20,
            upload: Some(usize::MAX),
        };
        assert_eq!(forwarded([bad_upload], 100, 20), None);
        assert_eq!(
            forwarded([Seen::Writes(usize::MAX, 1), placed(100, 200, 20)], 100, 20),
            None
        );
    }

    #[test]
    fn switch_is_off_by_default() {
        assert!(!enabled(None));
        assert!(!enabled(Some(OsStr::new("0"))));
        assert!(!enabled(Some(OsStr::new("true"))));
        assert!(enabled(Some(OsStr::new("1"))));
    }
}
