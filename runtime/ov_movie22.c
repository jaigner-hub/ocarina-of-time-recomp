// Written for ocarina-of-time-recomp (MIT, Copyright (c) 2026 Jeff Aigner).
//
// SDK 22's movie library (subsdk0, movie 22.2.8) gives the game movie::VideoDecoder and movie::AudioDecoder (not the
// movie::Decoder of ov_movie.c, which this game never imports). Their Initialize opens the Switch 2's codec services
// through nn::codec (movie::detail::InitializeDecoderByRemoteMemory), which this host does not serve: the SDK aborts.
// The game calls the decoders through imports of main, so every public method it imports is replaced here by name,
// and the library's object is never touched (its constructor, destructor and Finalize are replaced as well).
//
// Video is decoded on the host with dav1d (the game's movies are AV1: the title screen is "video/av01" 1920x1080 at
// 60 fps, 321 samples per loop, build/logs/run-64.log), on a host thread per decoder (decode_thread), into the
// VideoFrames the game registered, and handed back on the playback clock (acquire). No movie of this game has an audio
// track (no run has created an AudioDecoder); one would get a null decoder: the stream is taken, and the registered
// frame comes back once with its end-of-stream state set when the end-of-stream buffer arrives. The events the game
// waits on (input ready, output ready, format changed) are interprocess nn::os system events, created through the
// SDK on the calling guest thread, whose kernel events any host thread can signal.
//
// The game's protocol (OoT 1.1.1, main+0x3b5f0c4): C1, Initialize(work memory, size, mime type, output format, codec
// mode), Configure(MediaProperty, codec data, size), the three event getters, RegisterVideoFrame(frames, count),
// Start, then DecodeAsync(nn::movie::Buffer) per extracted sample and AcquireVideoFrame / ReleaseVideoFrame per
// output; Stop, Flush, Finalize, D1 at the end; after the last sample an empty end-of-stream buffer (flag 8; flag 4
// marks the first sample of a stream, also each restart of the title loop), and for the title loop the samples start
// over. 2026-11-06: the end was taken for flag 4, so no movie ever ended: after the 154 s opening cutscene the game
// waited for the end-of-stream frame and gameplay stayed black (run-248 logged the empty flag-8 buffer). The track's MediaProperty
// (read by the game before Configure through nn::movie::MediaProperty::Get*, observed below) carries "mime",
// "width", "height", "frame-rate", "color-standard", "color-range", "color-transfer".
//
// Layouts measured in the lifted SDK (generated/survey-v131072): nn::movie::Buffer (sdk, GetBuffer / GetTimeStamp /
// GetBufferFlags): +0 data, +8 capacity, +0x10 size, +0x18 time stamp (nn::TimeSpan, ns), +0x20 flags.
// movie::VideoFrame (subsdk0, 0x90 bytes; SetBuffer, GetBuffer, SetPixelPlaneData, GetPixelPlaneFilledDataSize,
// SetTimeStamp, SetEndOfStreamState, SetPixelStorageType): +0x8+8*plane plane data (plane 0 is the buffer the game
// set with SetBuffer), +0x28+8*plane plane filled size (plane 0's is the buffer's capacity), +0x78 time stamp, +0x84
// pixel storage type (3 = native image handle, where SetBuffer is refused), +0x88 end of stream. The game gives each
// frame a buffer of width*height*3/2 bytes and reads the planes back by GetBuffer and GetPixelPlaneFilledDataSize, so
// the planes are laid out in that buffer one after the other; it asks for planes 0 and 1 only (run-66), so Y, then
// U and V interleaved (NV12, tight rows).
#include <stdatomic.h>
#include "platform/platform.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dav1d/dav1d.h>

#include "platform/platform.h"
#include "dispatch.h"
#include "cfg.h"
#include "host.h"

#define MAX_DECODERS 8
#define MAX_FRAMES 16
#define MAX_READY 16
#define SYSEVENT_BYTES 0x100             // nn::os::SystemEventType is 0x28 bytes on SDK 22; room to spare
#define FLAG_END_OF_STREAM 8u
#define FLAG_STREAM_START 4u
#define MOVIE_R_NO_FRAME (342u | (2u << 9))   // module 342 (movie): nothing to acquire

typedef struct { Dav1dPicture pic; int eos; } Ready;
typedef struct { uint8_t *data; size_t size; int64_t ts; int eos; } Sample;   // a compressed sample waiting for the ring

typedef struct {
    uint64_t guest;                      // the game's movie::VideoDecoder / AudioDecoder object (0: free)
    int video;
    int started;
    uint8_t *ev_in, *ev_out, *ev_fmt;    // nn::os::SystemEventType objects, host memory the guest addresses directly
    uint64_t frames[MAX_FRAMES];
    int nframes;
    uint32_t busy;                       // bit per registered frame: handed to the game, not released yet
    uint64_t decoded, acquired, dropped;
    // video
    Dav1dContext *dav;
    uint8_t *config; size_t config_len;  // the sequence header OBUs from Configure's codec data (av1C), sent first
    int config_sent;
    Ready ready[MAX_READY];
    int rhead, rcount;
    Sample *queue;                       // samples taken by DecodeAsync and not yet given to dav1d, in order
    int qhead, qcount, qcap;
    int64_t last_pts;
    int eos_pending;                     // audio: the registered frame comes back once as end of stream
    int warned;
    int64_t pace_pts; uint64_t pace_ns;  // pacing log: the picture time stamp and host time at the last line
    uint64_t skipped, pace_skipped, decode_ns, pace_decode_ns;
    int64_t clock_pts; uint64_t clock_ns;  // playback clock: the picture time stamp shown at host time clock_ns
    int skip, skip_sent, skip_end;       // OOT_MOVIE_SKIP: the last picture and the end once the game sent its end
                                         // buffer (skip_sent: 1 the picture handed back, 2 the end too)
    int64_t skip_pts;                    // the last sample's time stamp, for that end frame
    PlatThread *worker;                  // the host thread that runs dav1d (decode_thread)
    PlatCond wake;             // under lock: a sample queued, a ring slot freed, or quit
    PlatMutex dav_lock;                    // held around every dav1d call
    int quit;
    uint32_t gen;                        // bumped by Stop/Flush/close: pictures decoded before it are discarded
} Dec;

static Dec decs[MAX_DECODERS];
static PlatMutex lock = PLAT_MUTEX_INIT;
static struct { uint64_t create_event, signal_event, set_plane, set_ts, video_eos, audio_eos; } G;
static atomic_int g_ready;

static uint64_t need(const char *name) {
    uint64_t a = ldr_symbol(name);
    if (!a) rt_fatal("movie22: guest symbol %s is not loaded", name);
    return a;
}

static void guest_init(void) {
    if (atomic_load(&g_ready)) return;
    plat_mutex_lock(&lock);
    if (!g_ready) {
        G.create_event = need("_ZN2nn2os17CreateSystemEventEPNS0_15SystemEventTypeENS0_14EventClearModeEb");
        G.signal_event = need("_ZN2nn2os17SignalSystemEventEPNS0_15SystemEventTypeE");
        G.set_plane = need("_ZN5movie10VideoFrame17SetPixelPlaneDataEPKvmNS_10PixelPlaneE");
        G.set_ts = need("_ZN5movie10VideoFrame12SetTimeStampEN2nn8TimeSpanE");
        G.video_eos = need("_ZN5movie10VideoFrame19SetEndOfStreamStateEb");
        G.audio_eos = need("_ZN5movie10AudioFrame19SetEndOfStreamStateEb");
        atomic_store(&g_ready, 1);
    }
    plat_mutex_unlock(&lock);
}

#define U(x) ((uint64_t)(uintptr_t)(x))

// nn::os::CreateSystemEvent(event, EventClearMode_AutoClear, interprocess) on the calling guest thread, through
// kern_bind_system_event: an interprocess event is a kernel event whose writable handle the kernel keeps, so the
// decoding thread (a host thread) can signal the output event when a picture is ready.
static uint8_t *event_new(Cpu *c) {
    uint8_t *e = calloc(1, SYSEVENT_BYTES);
    if (!e) rt_fatal("movie22: out of memory");
    kern_bind_system_event(c, U(e), 1);
    return e;
}

static void event_signal(Cpu *c, const uint8_t *e) { (void)c; if (e) kern_signal_system_event(U(e)); }

static Dec *find(uint64_t guest) {
    for (int i = 0; i < MAX_DECODERS; i++)
        if (decs[i].guest == guest) return &decs[i];
    return NULL;
}

static void ready_clear(Dec *d) {
    while (d->rcount) {
        Ready *r = &d->ready[d->rhead];
        if (!r->eos) dav1d_picture_unref(&r->pic);
        d->rhead = (d->rhead + 1) % MAX_READY, d->rcount--;
    }
}

static void queue_clear(Dec *d) {
    while (d->qcount) free(d->queue[d->qhead].data), d->qhead = (d->qhead + 1) % d->qcap, d->qcount--;
    d->qhead = 0;
}

// lock held; the worker is stopped first (it takes the lock to finish), so the lock is let go while it ends.
static void dav_close(Dec *d) {
    if (d->worker) {
        d->quit = 1, d->gen++;
        plat_cond_broadcast(&d->wake);
        plat_mutex_unlock(&lock);
        plat_thread_join(d->worker);
        plat_mutex_lock(&lock);
        d->worker = NULL;
    }
    ready_clear(d);
    queue_clear(d);
    if (d->dav) dav1d_close(&d->dav);
    free(d->config), d->config = NULL, d->config_len = 0;
}

static void decode_thread(void *arg);

static Dec *open_dec(Cpu *c, uint64_t guest, int video) {
    guest_init();
    plat_mutex_lock(&lock);
    Dec *d = find(guest);
    if (!d) d = find(0);
    if (d) {
        dav_close(d);
        uint8_t *in = d->ev_in, *out = d->ev_out, *fmt = d->ev_fmt;
        memset(d, 0, sizeof *d);
        d->guest = guest, d->video = video;
        d->ev_in = in, d->ev_out = out, d->ev_fmt = fmt;     // events of an earlier life stay valid
        if (video) {
            // OOT_MOVIE_SKIP=<n> (testing): the n-th video decoder opened and every later one end at once, so a run
            // gets past the opening cinematic to the game (the title loop is the first, the opening the second).
            static int opened;
            const char *e = getenv("OOT_MOVIE_SKIP");
            int from = e ? atoi(e) : 0;
            d->skip = from > 0 && ++opened >= from;
            if (d->skip) rt_log("movie22: video decoder %d skipped (OOT_MOVIE_SKIP=%d): no pictures, its end as soon as "
                                "the game has sent all samples", opened, from);
        }
    }
    plat_mutex_unlock(&lock);
    if (!d) { rt_log("movie22: more than %d decoders alive", MAX_DECODERS); return NULL; }
    if (!d->ev_in) d->ev_in = event_new(c), d->ev_out = event_new(c), d->ev_fmt = event_new(c);
    if (video) {
        Dav1dSettings s;
        dav1d_default_settings(&s);
        s.n_threads = 4;
        s.max_frame_delay = 1;           // every sample's picture comes out at once: the game paces by time stamps
        // Tuning (Linux test runs): OOT_MOVIE_THREADS, OOT_MOVIE_DELAY, OOT_MOVIE_GRAIN=0 (no film grain synthesis).
        const char *e;
        if ((e = cfg_get("OOT_MOVIE_THREADS")) && atoi(e) > 0) s.n_threads = atoi(e);
        if ((e = cfg_get("OOT_MOVIE_DELAY")) && atoi(e) > 0) s.max_frame_delay = atoi(e);
        if ((e = cfg_get("OOT_MOVIE_GRAIN")) && *e) s.apply_grain = atoi(e) != 0;
        rt_log("movie22: dav1d %d threads, frame delay %d, film grain %d", s.n_threads, s.max_frame_delay, s.apply_grain);
        int r = dav1d_open(&d->dav, &s);
        if (r) rt_log("movie22: dav1d_open failed (%d): the cinematic shows no picture", r);
        plat_cond_init(&d->wake);
        plat_mutex_init(&d->dav_lock);
        if (d->dav && !(d->worker = plat_thread_new(decode_thread, d, "movie22")))
            rt_fatal("movie22: the decoding thread did not start");
    }
    return d;
}

static void ov_ctor(Cpu *c) { (void)c; }                       // the object's fields are never read here
static void ov_ok(Cpu *c) { ret_x(c, 0); }

static void init_common(Cpu *c, int video) {
    static atomic_long said;
    uint64_t self = arg_x(c, 0), work = arg_x(c, 1), size = arg_x(c, 2), mime = arg_x(c, 3), fmt = arg_x(c, 4), mode = arg_x(c, 5);
    Dec *d = open_dec(c, self, video);   // its rt_calls clobber x0-x7: the arguments were read above
    if ((atomic_fetch_add(&said, 1) + 1) <= 4)
        rt_log("movie22: %s decoder %#llx (work memory %#llx, %llu bytes; mime %llu, output format %llu, codec mode %llu): %s",
               video ? "video" : "audio", (unsigned long long)self, (unsigned long long)work, (unsigned long long)size,
               (unsigned long long)mime, (unsigned long long)fmt, (unsigned long long)mode,
               video ? (d && d->dav ? "dav1d" : "no decoder") : "a null decoder stands in");
    ret_x(c, d ? 0 : MOVIE_R_NO_FRAME);
}
static void ov_video_init(Cpu *c) { init_common(c, 1); }
static void ov_audio_init(Cpu *c) { init_common(c, 0); }

// Configure(MediaProperty, codec data, size): for AV1 the codec data is an av1C record (4 bytes of marker, version,
// profile, level and flags, then the configuration OBUs: the sequence header), which dav1d needs before the first
// sample when the samples do not repeat it.
static void ov_configure(Cpu *c) {
    Dec *d = find(arg_x(c, 0));
    const uint8_t *data = arg_p(c, 2);
    size_t n = (size_t)arg_x(c, 3);
    if (d && d->video && data && n > 4) {
        free(d->config);
        d->config = malloc(n - 4);
        if (d->config) memcpy(d->config, data + 4, n - 4), d->config_len = n - 4;
        d->config_sent = 0;
        char hex[3 * 32 + 1] = "";
        for (size_t i = 0; i < n && i < 32; i++) snprintf(hex + 3 * i, 4, "%02x ", data[i]);
        rt_log("movie22: codec data (%zu bytes): %s", n, hex);
    }
    if (d && !d->warned++)
        rt_log("movie22: Configure(%#llx, property %#llx, codec data %#llx, %llu bytes)", (unsigned long long)arg_x(c, 0),
               (unsigned long long)arg_x(c, 1), (unsigned long long)arg_x(c, 2), (unsigned long long)arg_x(c, 3));
    ret_x(c, 0);
}

static void ov_finalize(Cpu *c) {
    Dec *d = find(arg_x(c, 0));
    if (d) {
        rt_log("movie22: %s decoder %#llx finalized: %llu buffers taken, %llu frames handed back, %llu pictures dropped",
               d->video ? "video" : "audio", (unsigned long long)d->guest, (unsigned long long)d->decoded,
               (unsigned long long)d->acquired, (unsigned long long)d->dropped);
        plat_mutex_lock(&lock);
        dav_close(d);
        d->guest = 0;
        plat_mutex_unlock(&lock);
    }
    ret_x(c, 0);
}
static void ov_dtor(Cpu *c) {
    Dec *d = find(arg_x(c, 0));
    if (d) { plat_mutex_lock(&lock); dav_close(d); d->guest = 0; plat_mutex_unlock(&lock); }
}

// The event getters: SystemEventType **out.
static void event_get(Cpu *c, int which) {
    Dec *d = find(arg_x(c, 0));
    uint64_t *out = arg_p(c, 1);
    if (!d || !out) { ret_x(c, MOVIE_R_NO_FRAME); return; }
    *out = U(which == 0 ? d->ev_in : which == 1 ? d->ev_out : d->ev_fmt);
    ret_x(c, 0);
}
static void ov_event_in(Cpu *c) { event_get(c, 0); }
static void ov_event_out(Cpu *c) { event_get(c, 1); }
static void ov_event_fmt(Cpu *c) { event_get(c, 2); }

// Pacing (2026-11-06): the game's movie worker feeds samples whenever it has them and does not look at the input
// event first; decoding each sample as it came ran the whole stream through dav1d in a few seconds, the ready ring
// overflowed, the oldest pictures were dropped and the game waited forever for the frame of a sample it had sent
// (the main thread in WaitEvent, the worker spinning); refusing a sample broke its pipeline the same way. So every
// sample is taken, as a copy on a queue, and given to dav1d only while the ring has room for its picture: at
// DecodeAsync, and again whenever an acquire frees a ring slot. Nothing is refused or dropped.
static int room(const Dec *d) { return d->rcount < MAX_READY - 2; }

static void ov_start(Cpu *c) {
    Dec *d = find(arg_x(c, 0));
    if (d) {
        d->started = 1;
        event_signal(c, d->ev_fmt);                          // the output format is known from the start
        event_signal(c, d->ev_in);                           // input is always welcome (queued)
    }
    ret_x(c, 0);
}

static void ov_stop(Cpu *c) {
    Dec *d = find(arg_x(c, 0));
    if (d) {
        rt_log("movie22: Stop/Flush of the %s decoder %#llx after %llu buffers in, %llu frames handed back, %d ready, "
               "%d queued", d->video ? "video" : "audio", (unsigned long long)d->guest, (unsigned long long)d->decoded,
               (unsigned long long)d->acquired, d->rcount, d->qcount);
        plat_mutex_lock(&lock);
        d->started = 0;
        d->gen++;                                            // a picture the worker is decoding now is discarded
        ready_clear(d);
        queue_clear(d);
        d->clock_ns = 0;                                     // the next picture restarts the playback clock
        plat_mutex_unlock(&lock);
        plat_mutex_lock(&d->dav_lock);
        if (d->dav) dav1d_flush(d->dav);
        d->config_sent = 0;
        plat_mutex_unlock(&d->dav_lock);
    }
    ret_x(c, 0);
}

static void ov_register(Cpu *c) {                            // RegisterVideoFrame / RegisterAudioFrame(frames, count)
    Dec *d = find(arg_x(c, 0));
    const uint64_t *frames = arg_p(c, 1);
    int n = (int)arg_x(c, 2);
    if (d && frames && n > 0) {
        plat_mutex_lock(&lock);
        d->nframes = 0;
        for (int i = 0; i < n && i < MAX_FRAMES; i++) if (frames[i]) d->frames[d->nframes++] = frames[i];
        d->busy = 0;
        plat_mutex_unlock(&lock);
        if (d->warned < 2) d->warned = 2, rt_log("movie22: %d %s frames registered", d->nframes, d->video ? "video" : "audio");
    }
    ret_x(c, 0);
}

static void ready_drop_oldest(Dec *d) {
    Ready *o = &d->ready[d->rhead];
    if (!o->eos) dav1d_picture_unref(&o->pic);
    d->rhead = (d->rhead + 1) % MAX_READY, d->rcount--, d->dropped++;
}

typedef struct { Dav1dPicture pic[8]; int n; } Out;      // pictures of one sample, from dav1d to the ring

// dav_lock held: the pictures dav1d has finished (max_frame_delay 1 gives one per sample; more than 8 are dropped).
static void drain(Dec *d, Out *o) {
    for (;;) {
        Dav1dPicture pic;
        memset(&pic, 0, sizeof pic);
        int r = dav1d_get_picture(d->dav, &pic);
        if (r) {
            if (r != DAV1D_ERR(EAGAIN) && d->warned < 3) d->warned = 3, rt_log("movie22: dav1d_get_picture: %d", r);
            return;
        }
        if (o->n == 8) { dav1d_picture_unref(&pic); d->dropped++; continue; }
        o->pic[o->n++] = pic;
    }
}

static void send_data(Dec *d, Out *o, const uint8_t *p, size_t n, int64_t ts) {   // dav_lock held
    Dav1dData data;
    uint8_t *buf = dav1d_data_create(&data, n);
    if (!buf) return;
    memcpy(buf, p, n);
    data.m.timestamp = ts;
    for (int tries = 0; tries < 64; tries++) {
        int r = dav1d_send_data(d->dav, &data);
        if (r == 0) break;
        if (r == DAV1D_ERR(EAGAIN)) { drain(d, o); continue; }
        if (d->warned < 3) d->warned = 3, rt_log("movie22: dav1d_send_data: %d", r);
        dav1d_data_unref(&data);
        return;
    }
    drain(d, o);
}

static Ready *ring_push(Dec *d) {                            // lock held
    if (d->rcount == MAX_READY) ready_drop_oldest(d);
    Ready *n = &d->ready[(d->rhead + d->rcount) % MAX_READY];
    memset(n, 0, sizeof *n);
    d->rcount++;
    return n;
}

// The decoding thread (2026-11-06): dav1d ran on the guest threads that called DecodeAsync and AcquireVideoFrame,
// 15 to 30 ms per picture shown once late pictures were skipped (run-197), in the game's frame. This host thread takes
// queued samples while the ring has room, decodes them outside the ring lock, puts their pictures in the ring and
// signals the output event (a kernel event, see event_new). A Stop or Flush meanwhile bumps gen: the pictures of a
// sample taken before it are discarded.
static void decode_thread(void *arg) {
    Dec *d = arg;
    plat_mutex_lock(&lock);
    while (!d->quit) {
        if (!d->qcount || !room(d)) { plat_cond_wait(&d->wake, &lock); continue; }
        Sample q = d->queue[d->qhead];
        d->qhead = (d->qhead + 1) % d->qcap, d->qcount--;
        uint32_t gen = d->gen;
        plat_mutex_unlock(&lock);
        Out o;
        o.n = 0;
        uint64_t t0 = rt_ticks();
        plat_mutex_lock(&d->dav_lock);
        if (gen == *(volatile uint32_t *)&d->gen) {
            if (q.eos) {
                drain(d, &o);
                dav1d_flush(d->dav);                         // the next samples start the stream over (the title loop)
                d->config_sent = 0;
            } else {
                if (!d->config_sent && d->config) send_data(d, &o, d->config, d->config_len, INT64_MIN), d->config_sent = 1;
                send_data(d, &o, q.data, q.size, q.ts);
            }
        }
        plat_mutex_unlock(&d->dav_lock);
        uint64_t dt = (rt_ticks() - t0) * 625 / 12;
        free(q.data);
        plat_mutex_lock(&lock);
        d->decode_ns += dt;
        int fresh = gen == d->gen;
        for (int i = 0; i < o.n; i++) {
            if (fresh) ring_push(d)->pic = o.pic[i];
            else dav1d_picture_unref(&o.pic[i]);
        }
        if (fresh && q.eos) ring_push(d)->eos = 1;
        if (fresh && d->rcount && d->ev_out) {
            const uint8_t *ev = d->ev_out;
            plat_mutex_unlock(&lock);
            kern_signal_system_event(U(ev));
            plat_mutex_lock(&lock);
        }
    }
    plat_mutex_unlock(&lock);
    return;
}

// Lock held: the decoding thread looks at the queue again (a sample queued or a ring slot freed).
static void pump(Dec *d) { if (d->worker) plat_cond_signal(&d->wake); }

static int enqueue(Dec *d, const uint8_t *p, size_t n, int64_t ts, int eos) {   // lock held
    if (d->qcount == d->qcap) {
        int cap = d->qcap ? d->qcap * 2 : 64;
        Sample *q = malloc((size_t)cap * sizeof *q);
        if (!q) return 0;
        for (int i = 0; i < d->qcount; i++) q[i] = d->queue[(d->qhead + i) % d->qcap];
        free(d->queue), d->queue = q, d->qcap = cap, d->qhead = 0;
    }
    Sample *q = &d->queue[(d->qhead + d->qcount) % d->qcap];
    q->data = n ? malloc(n) : NULL;
    if (n && !q->data) return 0;
    if (n) memcpy(q->data, p, n);
    q->size = n, q->ts = ts, q->eos = eos;
    d->qcount++;
    return 1;
}

static void ov_decode(Cpu *c) {                              // DecodeAsync(const nn::movie::Buffer *)
    Dec *d = find(arg_x(c, 0));
    uint64_t buffer = arg_x(c, 1);
    if (!d) { ret_x(c, MOVIE_R_NO_FRAME); return; }
    uint64_t data = buffer ? LD64(buffer) : 0, size = buffer ? LD64(buffer + 0x10) : 0;
    int64_t ts = buffer ? (int64_t)LD64(buffer + 0x18) : 0;
    uint32_t flags = buffer ? LD32(buffer + 0x20) : 0;
    d->decoded++;
    int eos = (flags & FLAG_END_OF_STREAM) != 0;
    if (flags || !size) {
        static atomic_long said;
        const uint8_t *p = (const uint8_t *)(uintptr_t)data;
        if ((atomic_fetch_add(&said, 1) + 1) <= 12 || !size || (flags & ~FLAG_STREAM_START))   // ends and odd flags: always
            rt_log("movie22: %s buffer %llu: flags %#x, %llu bytes, pts %lld ns%s%s", d->video ? "video" : "audio",
                   (unsigned long long)d->decoded, flags, (unsigned long long)size, (long long)ts, eos ? " (end of stream)" : "",
                   size >= 4 && p ? "" : " (empty)");
        if (size >= 8 && p && said <= 12)
            rt_log("movie22:   starts %02x %02x %02x %02x %02x %02x %02x %02x", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    }
    if (d->video && d->skip) {                               // OOT_MOVIE_SKIP: nothing decoded
        if (eos) d->skip_end = 1;
        else if (size) d->skip_pts = ts;
        event_signal(c, d->ev_in);
        if (d->skip_end) event_signal(c, d->ev_out);
    } else if (d->video) {
        plat_mutex_lock(&lock);
        if ((data && size) || eos) {
            if (!enqueue(d, (const uint8_t *)(uintptr_t)data, (size_t)(data ? size : 0), ts, eos) && d->warned < 3)
                d->warned = 3, rt_log("movie22: out of memory queueing a %llu-byte sample", (unsigned long long)size);
        }
        pump(d);
        int have = d->rcount > 0;
        plat_mutex_unlock(&lock);
        event_signal(c, d->ev_in);
        if (have) event_signal(c, d->ev_out);
    } else {
        if (eos) d->eos_pending = 1;
        event_signal(c, d->ev_in);
        if (d->eos_pending) event_signal(c, d->ev_out);
    }
    ret_x(c, 0);
}

static int free_frame(const Dec *d) {
    for (int i = 0; i < d->nframes; i++) if (!(d->busy & (1u << i))) return i;
    return -1;
}

// A decoded picture into the game's frame buffer: Y, then interleaved UV, tight rows.
static int fill_frame(Cpu *c, Dec *d, uint64_t frame, const Dav1dPicture *pic) {
    uint8_t *buf = (uint8_t *)(uintptr_t)LD64(frame + 0x8);
    uint64_t cap = LD64(frame + 0x48);
    int w = pic->p.w, h = pic->p.h, cw = (w + 1) / 2, ch = (h + 1) / 2;
    size_t ysz = (size_t)w * h, csz = (size_t)cw * ch;
    if (!buf || pic->p.layout != DAV1D_PIXEL_LAYOUT_I420 || pic->p.bpc != 8 || ysz + 2 * csz > cap) {
        if (d->warned < 4)
            d->warned = 4, rt_log("movie22: picture %dx%d layout %d bpc %d does not fit the game's frame (%llu bytes)", w, h,
                                  pic->p.layout, pic->p.bpc, (unsigned long long)cap);
        return 0;
    }
    const uint8_t *src = pic->data[0];
    for (int y = 0; y < h; y++) memcpy(buf + (size_t)y * w, src + (size_t)y * pic->stride[0], (size_t)w);
    const uint8_t *u = pic->data[1], *v = pic->data[2];
    uint8_t *uv = buf + ysz;                                 // the game reads two planes: Y, then UV interleaved (NV12)
    for (int y = 0; y < ch; y++, u += pic->stride[1], v += pic->stride[1], uv += 2 * cw)
        for (int x = 0; x < cw; x++) uv[2 * x] = u[x], uv[2 * x + 1] = v[x];
    rt_call(c, G.set_plane, 4, frame, U(buf), (uint64_t)ysz, 0ull);
    rt_call(c, G.set_plane, 4, frame, U(buf + ysz), (uint64_t)(2 * csz), 1ull);
    rt_call(c, G.set_plane, 4, frame, 0ull, 0ull, 2ull);
    int64_t ts = pic->m.timestamp == INT64_MIN ? d->last_pts : pic->m.timestamp;
    d->last_pts = ts;
    rt_call(c, G.set_ts, 2, frame, (uint64_t)ts);
    rt_call(c, G.video_eos, 2, frame, 0ull);
    return 1;
}

static void acquire(Cpu *c, int with_index) {               // Acquire{Video,Audio}Frame(Frame **out[, int *index])
    Dec *d = find(arg_x(c, 0));
    uint64_t *out = arg_p(c, 1);
    int *index = with_index ? arg_p(c, 2) : NULL;
    if (!d || !out || !d->nframes) { ret_x(c, MOVIE_R_NO_FRAME); return; }
    if (d->video && d->skip) {
        // OOT_MOVIE_SKIP: once the game has sent its end (flag 8) buffer, one black picture with the last sample's time
        // stamp, then the end of stream. The cutscene follows the time stamps of the pictures it shows: an end alone
        // (at once, or after the game's end buffer, with time stamp 0 or the last one) left it waiting (2026-11-06).
        if (!d->skip_end || d->skip_sent >= 2) { ret_x(c, MOVIE_R_NO_FRAME); return; }
        int i = free_frame(d);
        if (i < 0) { ret_x(c, MOVIE_R_NO_FRAME); return; }
        uint64_t frame = d->frames[i];
        d->busy |= 1u << i, d->acquired++;
        if (d->skip_sent == 0) {
            uint8_t *buf = (uint8_t *)(uintptr_t)LD64(frame + 0x8);
            uint64_t cap = LD64(frame + 0x48);
            size_t ysz = (size_t)cap * 2 / 3;
            if (buf && cap) memset(buf, 16, ysz), memset(buf + ysz, 128, (size_t)cap - ysz);   // black, NV12
            rt_call(c, G.set_plane, 4, frame, U(buf), (uint64_t)ysz, 0ull);
            rt_call(c, G.set_plane, 4, frame, U(buf + ysz), (uint64_t)(cap - ysz), 1ull);
            rt_call(c, G.set_plane, 4, frame, 0ull, 0ull, 2ull);
            rt_call(c, G.set_ts, 2, frame, (uint64_t)d->skip_pts);
            rt_call(c, G.video_eos, 2, frame, 0ull);
            d->skip_sent = 1, d->last_pts = d->skip_pts;
            *out = frame;
            if (index) *index = i;
            rt_log("movie22: skipped movie handed back its last picture (pts %lld ns)", (long long)d->skip_pts);
            event_signal(c, d->ev_out);
            ret_x(c, 0);
            return;
        }
        d->skip_sent = 2;
        rt_call(c, G.set_ts, 2, frame, (uint64_t)d->skip_pts);
        rt_call(c, G.video_eos, 2, frame, 1ull);
        *out = frame;
        if (index) *index = i;
        rt_log("movie22: skipped movie handed back its end of stream");
        ret_x(c, 0);
        return;
    }
    if (!d->video) {
        if (!d->eos_pending || d->busy) { ret_x(c, MOVIE_R_NO_FRAME); return; }
        uint64_t frame = d->frames[0];
        rt_call(c, G.audio_eos, 2, frame, 1ull);
        *out = frame;
        if (index) *index = 0;
        d->busy |= 1, d->acquired++;
        ret_x(c, 0);
        return;
    }
    plat_mutex_lock(&lock);
    int i = free_frame(d);
    if (!d->rcount || i < 0) { plat_mutex_unlock(&lock); ret_x(c, MOVIE_R_NO_FRAME); return; }
    // The game takes one picture per game frame and does not look at the time stamps, so below 60 fps the movie
    // played slower (a third of its speed at 20 fps, run-196). Pictures whose successor is already due by the
    // playback clock are skipped (as far as the ring reaches); end-of-stream entries are never skipped.
    uint64_t now = rt_ticks() * 625 / 12;
    while (d->clock_ns && d->rcount >= 2) {
        Ready *h = &d->ready[d->rhead], *nx = &d->ready[(d->rhead + 1) % MAX_READY];
        if (h->eos || nx->eos || nx->pic.m.timestamp == INT64_MIN || nx->pic.m.timestamp < d->clock_pts) break;
        if ((int64_t)(now - d->clock_ns) < nx->pic.m.timestamp - d->clock_pts) break;
        dav1d_picture_unref(&h->pic);
        d->rhead = (d->rhead + 1) % MAX_READY, d->rcount--, d->skipped++;
    }
    Ready r = d->ready[d->rhead];
    d->rhead = (d->rhead + 1) % MAX_READY, d->rcount--;
    d->busy |= 1u << i;
    if (!r.eos && r.pic.m.timestamp != INT64_MIN) {
        // (Re)start the clock on the first picture, when the time stamps go back (another movie) and after a stall of
        // more than half a second (a pause, the window out of focus), so the movie resumes instead of jumping ahead.
        int64_t ts = r.pic.m.timestamp;
        if (!d->clock_ns || ts < d->clock_pts || (int64_t)(now - d->clock_ns) - (ts - d->clock_pts) > 500000000)
            d->clock_pts = ts, d->clock_ns = now;
    }
    pump(d);                                                 // the slot taken makes room for a queued sample
    int have = d->rcount > 0;
    plat_mutex_unlock(&lock);
    if (have) event_signal(c, d->ev_out);                    // more pictures wait (the event clears on each wait)
    uint64_t frame = d->frames[i];
    if (r.eos) {
        rt_call(c, G.set_ts, 2, frame, (uint64_t)d->last_pts);
        rt_call(c, G.video_eos, 2, frame, 1ull);
    } else {
        int ok = fill_frame(c, d, frame, &r.pic);
        dav1d_picture_unref(&r.pic);
        if (!ok) { d->busy &= ~(1u << i); ret_x(c, MOVIE_R_NO_FRAME); return; }
    }
    *out = frame;
    if (index) *index = i;
    d->acquired++;
    if (d->acquired == 1) rt_log("movie22: first picture handed back at %.3f s (pts %lld ns)", rt_uptime_s(), (long long)d->last_pts);
    if (!r.eos && d->acquired % 300 == 0) {                  // playback speed: time stamps against host time
        now = rt_ticks() * 625 / 12;
        if (d->pace_ns && d->last_pts > d->pace_pts)
            rt_log("movie22: pacing at %.2f s: %llu pictures handed back; the last 300 cover %.2f s of video in %.2f s, %llu "
                   "skipped, %.1f ms of decoding (on the decoding thread) per picture handed back",
                   rt_uptime_s(), (unsigned long long)d->acquired, (d->last_pts - d->pace_pts) / 1e9, (now - d->pace_ns) / 1e9,
                   (unsigned long long)(d->skipped - d->pace_skipped), (d->decode_ns - d->pace_decode_ns) / 300e6);
        d->pace_pts = d->last_pts, d->pace_ns = now, d->pace_skipped = d->skipped, d->pace_decode_ns = d->decode_ns;
    }
    if (d->acquired <= 2 || r.eos || getenv("OOT_MQ_LOG")) {
        char t[96];
        rt_log("movie22: frame %d handed back (%s, pts %lld ns, %llu buffers in, %d ready) to %s", i, r.eos ? "end of stream" : "picture",
               (long long)d->last_pts, (unsigned long long)d->decoded, d->rcount, kern_thread_describe(t, sizeof t));
    }
    ret_x(c, 0);
}
static void ov_acquire(Cpu *c) { acquire(c, 0); }
static void ov_acquire_index(Cpu *c) { acquire(c, 1); }

static void ov_release(Cpu *c) {                             // Release{Video,Audio}Frame(Frame *)
    Dec *d = find(arg_x(c, 0));
    uint64_t frame = arg_x(c, 1);
    if (d) {
        plat_mutex_lock(&lock);
        for (int i = 0; i < d->nframes; i++) if (d->frames[i] == frame) d->busy &= ~(1u << i);
        int have = d->video && d->rcount > 0;
        plat_mutex_unlock(&lock);
        if (getenv("OOT_MQ_LOG")) {
            char t[96];
            rt_log("movie22: frame %#llx released by %s (%d ready, busy %#x)", (unsigned long long)frame, kern_thread_describe(t, sizeof t), d->rcount, d->busy);
        }
        if (have) event_signal(c, d->ev_out);
    }
    ret_x(c, 0);
}

// Property-returning calls the game is not seen making (the frame properties are its own); logged if it does.
static void ov_props(Cpu *c) {
    char d[128];
    rt_log("movie22: %s property object %#llx", ldr_describe(c->pc, d, sizeof d), (unsigned long long)arg_x(c, 1));
    ret_x(c, 0);
}

// DIAG: every property the game reads from the track, with the key and the answer.
static void obs_get_int32(Cpu *c) {
    uint64_t obj = arg_x(c, 0), out = arg_x(c, 1); const char *key = arg_p(c, 2);
    uint64_t r = rt_original(c);
    rt_log("movie22: GetInt32(%#llx, \"%s\") -> %llu, %d", (unsigned long long)obj, key, (unsigned long long)r, out ? *(int *)(uintptr_t)out : 0);
}
static void obs_get_size(Cpu *c) {
    uint64_t obj = arg_x(c, 0), out = arg_x(c, 1); const char *key = arg_p(c, 2);
    uint64_t r = rt_original(c);
    rt_log("movie22: GetSize(%#llx, \"%s\") -> %llu, %llu", (unsigned long long)obj, key, (unsigned long long)r, out ? (unsigned long long)*(uint64_t *)(uintptr_t)out : 0);
}
static void obs_get_float(Cpu *c) {
    uint64_t obj = arg_x(c, 0), out = arg_x(c, 1); const char *key = arg_p(c, 2);
    uint64_t r = rt_original(c);
    rt_log("movie22: GetFloat(%#llx, \"%s\") -> %llu, %g", (unsigned long long)obj, key, (unsigned long long)r, out ? *(float *)(uintptr_t)out : 0.0f);
}
static void obs_get_string(Cpu *c) {
    uint64_t obj = arg_x(c, 0), out = arg_x(c, 1); const char *key = arg_p(c, 3);
    uint64_t r = rt_original(c);
    rt_log("movie22: GetString(%#llx, \"%s\") -> %llu, \"%s\"", (unsigned long long)obj, key, (unsigned long long)r, out && r ? (const char *)(uintptr_t)out : "");
}
// DIAG: how the game reads the frames back (which planes, which storage type), the first few calls.
static void obs_plane_size(Cpu *c) {
    uint64_t frame = arg_x(c, 0), out = arg_x(c, 1), plane = arg_x(c, 2);
    uint64_t r = rt_original(c);
    static atomic_long n;
    if ((atomic_fetch_add(&n, 1) + 1) <= 6)
        rt_log("movie22: GetPixelPlaneFilledDataSize(frame %#llx, plane %llu) -> %llu, %llu bytes", (unsigned long long)frame,
               (unsigned long long)plane, (unsigned long long)r, out ? (unsigned long long)LD64(out) : 0);
}
static void obs_storage_type(Cpu *c) {
    uint64_t frame = arg_x(c, 0), type = arg_x(c, 1);
    uint64_t r = rt_original(c);
    static atomic_long n;
    if ((atomic_fetch_add(&n, 1) + 1) <= 2)
        rt_log("movie22: SetPixelStorageType(frame %#llx, %llu) -> %llu", (unsigned long long)frame, (unsigned long long)type, (unsigned long long)r);
}
static void obs_set_buffer(Cpu *c) {
    uint64_t frame = arg_x(c, 0), buf = arg_x(c, 1), size = arg_x(c, 2);
    uint64_t r = rt_original(c);
    static atomic_long n;
    if ((atomic_fetch_add(&n, 1) + 1) <= 2)
        rt_log("movie22: VideoFrame::SetBuffer(frame %#llx, %#llx, %llu bytes) -> %llu", (unsigned long long)frame,
               (unsigned long long)buf, (unsigned long long)size, (unsigned long long)r);
}
HOST_OBSERVER("_ZNK2nn5movie13MediaProperty8GetInt32EPiPKc", obs_get_int32);
HOST_OBSERVER("_ZNK2nn5movie13MediaProperty7GetSizeEPmPKc", obs_get_size);
HOST_OBSERVER("_ZNK2nn5movie13MediaProperty8GetFloatEPfPKc", obs_get_float);
HOST_OBSERVER("_ZNK2nn5movie13MediaProperty9GetStringEPciPKc", obs_get_string);
HOST_OBSERVER("_ZN5movie10VideoFrame27GetPixelPlaneFilledDataSizeEPmNS_10PixelPlaneE", obs_plane_size);
HOST_OBSERVER("_ZN5movie10VideoFrame19SetPixelStorageTypeENS_16PixelStorageTypeE", obs_storage_type);
HOST_OBSERVER("_ZN5movie10VideoFrame9SetBufferEPvm", obs_set_buffer);

HOST_OVERRIDE("_ZN5movie12VideoDecoderC1Ev", ov_ctor);
HOST_OVERRIDE("_ZN5movie12VideoDecoderD1Ev", ov_dtor);
HOST_OVERRIDE("_ZN5movie12VideoDecoder10InitializeEPvmN2nn5movie13MediaMimeTypeENS_24VideoDecoderOutputFormatENS_14VideoCodecModeE", ov_video_init);
HOST_OVERRIDE("_ZN5movie12VideoDecoder8FinalizeEv", ov_finalize);
HOST_OVERRIDE("_ZN5movie12VideoDecoder9ConfigureEPKN2nn5movie13MediaPropertyEPKvm", ov_configure);
HOST_OVERRIDE("_ZN5movie12VideoDecoder18GetInputReadyEventEPPN2nn2os15SystemEventTypeE", ov_event_in);
HOST_OVERRIDE("_ZN5movie12VideoDecoder19GetOutputReadyEventEPPN2nn2os15SystemEventTypeE", ov_event_out);
HOST_OVERRIDE("_ZN5movie12VideoDecoder21GetFormatChangedEventEPPN2nn2os15SystemEventTypeE", ov_event_fmt);
HOST_OVERRIDE("_ZN5movie12VideoDecoder5StartEv", ov_start);
HOST_OVERRIDE("_ZN5movie12VideoDecoder4StopEv", ov_stop);
HOST_OVERRIDE("_ZN5movie12VideoDecoder5FlushEv", ov_stop);
HOST_OVERRIDE("_ZN5movie12VideoDecoder11DecodeAsyncEPKN2nn5movie6BufferE", ov_decode);
HOST_OVERRIDE("_ZN5movie12VideoDecoder18RegisterVideoFrameEPPNS_10VideoFrameEi", ov_register);
HOST_OVERRIDE("_ZN5movie12VideoDecoder17AcquireVideoFrameEPPNS_10VideoFrameE", ov_acquire);
HOST_OVERRIDE("_ZN5movie12VideoDecoder17AcquireVideoFrameEPPNS_10VideoFrameEPi", ov_acquire_index);
HOST_OVERRIDE("_ZN5movie12VideoDecoder17ReleaseVideoFrameEPNS_10VideoFrameE", ov_release);
HOST_OVERRIDE("_ZN5movie12VideoDecoder25GetVideoFrameRequirementsEPN2nn5movie13MediaPropertyE", ov_props);
HOST_OVERRIDE("_ZN5movie12VideoDecoder24GetOutputFramePropertiesEPN2nn5movie13MediaPropertyE", ov_props);

HOST_OVERRIDE("_ZN5movie12AudioDecoderC1Ev", ov_ctor);
HOST_OVERRIDE("_ZN5movie12AudioDecoderD1Ev", ov_dtor);
HOST_OVERRIDE("_ZN5movie12AudioDecoder10InitializeEPvmN2nn5movie13MediaMimeTypeENS_24AudioDecoderOutputFormatENS_14AudioCodecModeE", ov_audio_init);
HOST_OVERRIDE("_ZN5movie12AudioDecoder8FinalizeEv", ov_finalize);
HOST_OVERRIDE("_ZN5movie12AudioDecoder9ConfigureEPKN2nn5movie13MediaPropertyEPKvm", ov_configure);
HOST_OVERRIDE("_ZN5movie12AudioDecoder25GetAudioFrameRequirementsEPN2nn5movie13MediaPropertyE", ov_props);
HOST_OVERRIDE("_ZN5movie12AudioDecoder18GetInputReadyEventEPPN2nn2os15SystemEventTypeE", ov_event_in);
HOST_OVERRIDE("_ZN5movie12AudioDecoder19GetOutputReadyEventEPPN2nn2os15SystemEventTypeE", ov_event_out);
HOST_OVERRIDE("_ZN5movie12AudioDecoder21GetFormatChangedEventEPPN2nn2os15SystemEventTypeE", ov_event_fmt);
HOST_OVERRIDE("_ZN5movie12AudioDecoder5StartEv", ov_start);
HOST_OVERRIDE("_ZN5movie12AudioDecoder4StopEv", ov_ok);
HOST_OVERRIDE("_ZN5movie12AudioDecoder5FlushEv", ov_ok);
HOST_OVERRIDE("_ZN5movie12AudioDecoder11DecodeAsyncEPKN2nn5movie6BufferE", ov_decode);
HOST_OVERRIDE("_ZN5movie12AudioDecoder18RegisterAudioFrameEPPNS_10AudioFrameEi", ov_register);
HOST_OVERRIDE("_ZN5movie12AudioDecoder17AcquireAudioFrameEPPNS_10AudioFrameEPi", ov_acquire_index);
HOST_OVERRIDE("_ZN5movie12AudioDecoder17ReleaseAudioFrameEPNS_10AudioFrameE", ov_release);
HOST_OVERRIDE("_ZN5movie12AudioDecoder24GetOutputFramePropertiesEPN2nn5movie13MediaPropertyE", ov_props);
