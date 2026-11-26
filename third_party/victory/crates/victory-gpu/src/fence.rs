//! Fences hold a 64-bit value and satisfy waits when the CPU or a queue
//! signals that value.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{OsString, c_void};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::object::{self, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) const FENCE_IIDS: [Iid; 4] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    /// Events to set once the value reaches theirs (handles as integers).
    waiters: Mutex<Vec<(u64, usize, std::time::Instant)>>,
}

/// (Neptune) Since they were last taken: events asked for a value the fence had reached
/// already, events asked for one it had not, and the microseconds those then waited to be set.
static ASKED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// [`ASKED`], taken.
pub(crate) fn asked_taken() -> [u64; 3] {
    std::array::from_fn(|i| ASKED[i].swap(0, Ordering::Relaxed))
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
    });
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

/// `VICTORY_FENCE_WAKE=1` (off by default; cool_fox `e05d463`): a blocking
/// wait also watches the fence word for writes made directly to its address.
#[cfg(windows)]
pub(crate) fn wake_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch(std::env::var_os("VICTORY_FENCE_WAKE")))
}

/// `VICTORY_VALUE_WAKE=1` (off by default; cool_fox `e05d463`): a list's
/// bottom-of-pipe write to memory wakes the threads sleeping on that word.
#[cfg(windows)]
pub(crate) fn value_wake_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch(std::env::var_os("VICTORY_VALUE_WAKE")))
}

/// Whether a switch's variable is set to `1`.
#[cfg(windows)]
fn switch(value: Option<OsString>) -> bool {
    value.is_some_and(|v| v == "1")
}

/// The longest single sleep, in ms, so that held work is looked at between.
#[cfg(windows)]
const WAIT_SLICE_MS: u128 = 10;

/// Waits until `word` holds `value` or `timeout` has passed, sleeping on
/// the word (a [`set`] wakes it). Whether it got there.
#[cfg(windows)]
pub(crate) fn wait_reached(word: &AtomicU64, value: u64, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        let current = word.load(Ordering::Acquire);
        if current >= value {
            return true;
        }
        let now = Instant::now();
        if now >= deadline {
            return false;
        }
        let ms = (deadline - now).as_millis().clamp(1, WAIT_SLICE_MS) as u32;
        // SAFETY: sleeps only while the word still holds `current`, for
        // `size_of::<u64>()` bytes at the address of a live u64.
        unsafe {
            windows_sys::Win32::System::Threading::WaitOnAddress(
                (word as *const AtomicU64).cast(),
                (&raw const current).cast(),
                std::mem::size_of::<u64>(),
                ms,
            );
        }
        crate::queue::release_held(false);
    }
}

/// Wakes the threads sleeping on the word at `at`.
#[cfg(windows)]
pub(crate) fn wake_word(at: *const c_void) {
    // SAFETY: waking an address nobody waits on does nothing.
    unsafe { windows_sys::Win32::System::Threading::WakeByAddressAll(at) };
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
pub(crate) extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}
/// The title's [`completed`]. (Neptune) A signal of the fence carried behind
/// merged title submissions (`VICTORY_MERGE_SUBMISSIONS=2`) goes to the card
/// with them first, so that the title, asking again, sees it once they have
/// run.
extern "system" fn completed_asked(this: This) -> u64 {
    #[cfg(windows)]
    crate::host::merge_fence_asked(this as usize);
    completed(this)
}
/// Where the fence's value is (address, bytes): titles read it there, and a
/// list may wait for it as a memory value.
#[cfg(windows)]
pub(crate) fn value_place(this: This) -> (usize, usize) {
    (gpu_address(this) as usize, 8)
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    #[cfg(windows)]
    use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0, WAIT_TIMEOUT};
    #[cfg(windows)]
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};

    let fence = fence(this);
    // (Neptune) As [`completed_asked`], for a value not reached yet.
    #[cfg(windows)]
    if fence.value.load(Ordering::Acquire) < value {
        crate::host::merge_fence_asked(this as usize);
    }
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        ASKED[0].fetch_add(1, Ordering::Relaxed);
        return S_OK;
    }
    #[cfg(not(windows))]
    if event.is_null() {
        tracing::warn!(value, "blocking fence wait requires Windows");
        return object::E_NOTIMPL;
    }
    // A private event uses the same waiters as a caller's event. Once a
    // signal satisfies it, a later rewind cannot undo the wakeup.
    #[cfg(windows)]
    let blocking = if event.is_null() {
        // SAFETY: a new, initially unsignaled event with default security.
        match unsafe { CreateEventW(None, false, false, None) } {
            Ok(event) => Some(event),
            Err(error) => return error.code().0,
        }
    } else {
        None
    };
    #[cfg(windows)]
    let event = blocking.map_or(event, |event| event.0);
    ASKED[1].fetch_add(1, Ordering::Relaxed);
    waiters.push((value, event as usize, std::time::Instant::now()));
    drop(waiters);

    #[cfg(windows)]
    if let Some(handle) = blocking {
        // A queue signal may already be behind in-flight work. Settle
        // only after releasing the fence lock so the signal can move it.
        crate::host::settle_for_fence();
        let watch_address = wake_on();
        let wait_ms = if watch_address { 0 } else { INFINITE };
        let result = loop {
            // SAFETY: this call owns the live event until the wait completes.
            match unsafe { WaitForSingleObject(handle, wait_ms) } {
                WAIT_OBJECT_0 => break S_OK,
                WAIT_TIMEOUT => {
                    // A write to the fence word may satisfy the wait without setting the event.
                    if wait_reached(&fence.value, value, Duration::from_millis(WAIT_SLICE_MS as u64)) {
                        break S_OK;
                    }
                }
                _ => break windows::core::Error::from_thread().code().0,
            }
        };
        if watch_address || result != S_OK {
            let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
            waiters.retain(|w| w.1 != event as usize);
        }
        // SAFETY: the waiter has been consumed or removed under the fence lock.
        let _ = unsafe { CloseHandle(handle) };
        return result;
    }
    S_OK
}
/// Moves the fence to `value` and sets the events that waited for it.
pub(crate) fn set(this: This, value: u64) {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    fence.value.store(value, Ordering::Release);
    waiters.retain(|&(wanted, event, since)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
            ASKED[2].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        !reached
    });
    drop(waiters);
    // (cool_fox `e05d463`) A wait sleeping on the value is woken now, after
    // the events are set.
    #[cfg(windows)]
    if wake_on() {
        wake_word((&raw const fence.value).cast());
    }
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    set(this, value);
    // A queue may have waited for it: what it held runs now.
    #[cfg(windows)]
    crate::queue::release_held(false);
    S_OK
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed_asked as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[cfg(windows)]
    #[test]
    fn a_waiter_is_woken_by_a_set_from_another_thread() {
        let word = AtomicU64::new(0);
        std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(Duration::from_millis(20));
                word.store(7, Ordering::Release);
                wake_word((&raw const word).cast());
            });
            assert!(wait_reached(&word, 5, Duration::from_secs(5)));
        });
    }

    #[cfg(windows)]
    #[test]
    fn a_wait_gives_up_at_its_deadline() {
        let word = AtomicU64::new(1);
        let started = Instant::now();
        assert!(!wait_reached(&word, 5, Duration::from_millis(30)));
        assert!(started.elapsed() >= Duration::from_millis(30));
    }

    #[cfg(windows)]
    #[test]
    fn a_reached_value_is_not_waited_for() {
        assert!(wait_reached(&AtomicU64::new(9), 5, Duration::ZERO));
    }

    #[cfg(windows)]
    #[test]
    fn the_switches_are_off_unless_set_to_one() {
        assert!(!switch(None));
        assert!(!switch(Some("0".into())));
        assert!(switch(Some("1".into())));
    }

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal = slot::<extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, std::ptr::null_mut()), S_OK, "a reached fence returns immediately");
        // The title reads the value at the fence's address.
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }

    #[cfg(windows)]
    #[test]
    fn null_event_waits_for_a_queued_gpu_copy() {
        use std::ptr::null;
        use std::sync::mpsc;
        use std::time::{Duration, Instant};
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RANGE, ID3D12CommandAllocator, ID3D12CommandQueue, ID3D12GraphicsCommandList,
        };
        use windows::core::Interface;

        let Some(host) = crate::host::host() else {
            eprintln!("skipped queued fence test: no D3D12 device");
            return;
        };
        let (adapter, _) = crate::host::adapter_summary().expect("identified test adapter");
        const MARKER: u32 = 0x1357_9bdf;
        const TARGET: u64 = 1;
        let bytes = size_of::<u32>() as u64;
        let (upload, data) = host.upload_buffer(bytes).unwrap();
        let output = host.read_back_buffer(bytes).unwrap();
        // SAFETY: the upload mapping holds one word.
        unsafe { data.cast::<u32>().write(MARKER) };

        fn iid<T: Interface>() -> Iid {
            let i = T::IID;
            guid(i.data1, i.data2, i.data3, i.data4)
        }
        let (mut allocator, mut list, mut queue, mut gate, mut done) = (null(), null(), null(), null(), null());
        assert_eq!(crate::list::create_allocator(null(), 0, &iid::<ID3D12CommandAllocator>(), &mut allocator), S_OK);
        assert_eq!(
            crate::list::create_list(null(), 0, 0, allocator, null(), &iid::<ID3D12GraphicsCommandList>(), &mut list),
            S_OK
        );
        let queue_desc = [0u32; 4];
        assert_eq!(crate::queue::create(null(), queue_desc.as_ptr().cast(), &iid::<ID3D12CommandQueue>(), &mut queue), S_OK);
        assert_eq!(create(null(), 0, 0, &FENCE_IIDS[0], &mut gate), S_OK);
        assert_eq!(create(null(), 0, 0, &FENCE_IIDS[0], &mut done), S_OK);

        let host_list = crate::list::host_list(list).unwrap();
        // SAFETY: an open list and live upload/readback buffers of the given size.
        unsafe { host_list.list.CopyBufferRegion(&output, 0, &upload, 0, bytes) };
        assert_eq!(slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list), S_OK);
        let signal_offset = if crate::device::newer() { 0x80 } else { 0x70 };
        type QueueFence = extern "system" fn(This, This, u64) -> HRESULT;
        assert_eq!(slot::<QueueFence>(queue, signal_offset + 8)(queue, gate, TARGET), S_OK);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        assert_eq!(slot::<QueueFence>(queue, signal_offset)(queue, done, TARGET), S_OK);
        assert_eq!(completed(done), 0, "the gate must hold the queue");

        let (sent, result) = mpsc::channel();
        let (registered, early, outcome) = std::thread::scope(|scope| {
            let address = done as usize;
            scope.spawn(move || {
                let done = address as This;
                let hr = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(done, 0x48)(
                    done, TARGET, std::ptr::null_mut(),
                );
                sent.send((hr, completed(done))).unwrap();
            });
            // Wait for registration or an early return before opening the gate.
            let deadline = Instant::now() + Duration::from_secs(5);
            let (registered, early) = loop {
                if let Ok(outcome) = result.try_recv() {
                    break (false, Some(outcome));
                }
                if fence(done).waiters.lock().unwrap().iter().any(|w| w.0 == TARGET) {
                    break (true, result.try_recv().ok());
                }
                if Instant::now() >= deadline {
                    break (false, None);
                }
                std::thread::sleep(Duration::from_millis(1));
            };
            signal(gate, TARGET);
            let outcome = early.or_else(|| result.recv_timeout(Duration::from_secs(5)).ok());
            if outcome.is_none() {
                // Release a pending test thread before reporting a timeout.
                set(done, TARGET);
            }
            (registered, early, outcome)
        });
        crate::host::settle_for_fence();
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the queue's list finished, and the readback holds one word.
        let gpu = unsafe {
            output.Map(0, None, Some(&mut mapped)).unwrap();
            let word = mapped.cast::<u32>().read();
            output.Unmap(0, Some(&D3D12_RANGE::default()));
            word
        };
        for object in [queue, list, allocator, gate, done] {
            slot::<extern "system" fn(This) -> u32>(object, 0x10)(object);
        }
        let (hr, completed) = outcome.expect("null-event waiter did not finish after the queue signal");
        println!(
            "adapter={adapter}, late={}, registered={registered}, returned_before_signal={}, HRESULT={:#010x}, fence={completed}, GPU={gpu:#010x}",
            crate::host::late(), early.is_some(), hr as u32,
        );
        assert!(registered && early.is_none(), "null-event wait returned before the signal: HRESULT {:#010x}", hr as u32);
        assert_eq!((hr, completed, gpu), (S_OK, TARGET, MARKER));
    }
}
