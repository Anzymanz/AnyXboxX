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

/// Where a fill of `[src, src + len)` recorded now takes its bytes
/// ([`forwarded`]), given the list's steps and its copies at their place out
/// of upload memory since its last `Step::Run` (`placed`: index, destination,
/// bytes). Only such a copy that holds the whole fill can give it its bytes,
/// so the walk starts at the earliest of those (none: nothing to walk), and a
/// fill in a list without one costs nothing more than the look at `placed`.
/// What it finds is what a walk of every step finds: the steps before that
/// copy are a `Run` (which ends the walk with nothing) or steps that cannot
/// give the fill its bytes.
pub(in crate::host) fn forward(steps: &[Step], placed: &[(usize, usize, usize)], src: usize, len: usize) -> Option<usize> {
    fill_after_copy().then_some(())?;
    let from = start(placed, src, len)?;
    forwarded(steps.get(from..)?.iter().rev().map(Seen::from), src, len)
}

/// The index of the earliest of `placed` whose destination holds all of
/// `[src, src + len)`.
fn start(placed: &[(usize, usize, usize)], src: usize, len: usize) -> Option<usize> {
    let to = src.checked_add(len)?;
    placed
        .iter()
        .filter(|&&(_, dst, n)| dst <= src && dst.checked_add(n).is_some_and(|end| to <= end))
        .map(|&(at, _, _)| at)
        .min()
}

/// List steps projected onto the ranges relevant to forwarding.
#[derive(Clone, Copy, Debug)]
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

    /// What the list keeps for [`forward`] as `steps` are recorded: copies
    /// out of upload memory since the last "anything" step.
    fn placements(steps: &[Seen]) -> Vec<(usize, usize, usize)> {
        let mut placed = Vec::new();
        for (at, step) in steps.iter().enumerate() {
            match *step {
                Seen::Placed { dst, len, upload: Some(_), .. } => placed.push((at, dst, len)),
                Seen::Anything => placed.clear(),
                _ => {}
            }
        }
        placed
    }

    /// [`forward`]'s walk: from the earliest copy that holds the fill.
    fn bounded(steps: &[Seen], src: usize, len: usize) -> Option<usize> {
        let from = start(&placements(steps), src, len)?;
        forwarded(steps[from..].iter().rev().copied(), src, len)
    }

    #[test]
    fn the_bounded_walk_finds_what_the_whole_walk_finds() {
        // Lists of every kind of step over a few ranges, the fill at each.
        let mut seed = 0x2545_f491_u64;
        let mut next = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n) as usize
        };
        for _ in 0..4000 {
            let steps: Vec<Seen> = (0..next(12))
                .map(|_| match next(6) {
                    0 | 1 => placed(100 + 10 * next(4), 200 + 10 * next(4), 10 + 10 * next(3)),
                    2 => Seen::Placed { dst: 100 + 10 * next(4), src: 200, len: 20, upload: None },
                    3 => Seen::Writes(100 + 10 * next(15), 1 + next(20)),
                    4 => Seen::Anything,
                    _ => Seen::Nothing,
                })
                .collect();
            let (src, len) = (100 + 5 * next(10), 1 + next(30));
            let whole = forwarded(steps.iter().rev().copied(), src, len);
            assert_eq!(bounded(&steps, src, len), whole, "{steps:?} fill {src}+{len}");
        }
    }

    #[test]
    fn a_long_list_without_a_copy_that_holds_the_fill_is_not_walked() {
        let mut steps: Vec<Seen> = (0..100_000).map(|i| Seen::Writes(10_000 + 8 * i, 4)).collect();
        assert_eq!(start(&placements(&steps), 100, 20), None);
        // A copy elsewhere, or one that holds only part of the fill: still no walk.
        steps.push(placed(500, 600, 20));
        steps.push(placed(100, 300, 10));
        assert_eq!(start(&placements(&steps), 100, 20), None);
        // One that holds it, near the end: the walk is the few steps behind it.
        steps.push(placed(100, 200, 20));
        steps.push(Seen::Nothing);
        assert_eq!(start(&placements(&steps), 100, 20), Some(100_002));
        assert_eq!(bounded(&steps, 100, 20), Some(300));
        // Behind a step that may write anything there is nothing to walk.
        steps.push(Seen::Anything);
        assert_eq!(start(&placements(&steps), 100, 20), None);
    }

    #[test]
    fn switch_is_off_by_default() {
        assert!(!enabled(None));
        assert!(!enabled(Some(OsStr::new("0"))));
        assert!(!enabled(Some(OsStr::new("true"))));
        assert!(enabled(Some(OsStr::new("1"))));
    }
}
