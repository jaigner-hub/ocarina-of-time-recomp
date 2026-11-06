// SDK overrides that Ocarina of Time needs and the TotK, FE3H and MP4 hosts do not have. They live here, apart from the
// files taken from metroid-prime-4-recomp (ov_system.c, ov_input.c, ...), so those stay close to their source; an
// override that turns out to be game independent can move into them later. The pattern follows totk-pc-recomp
// runtime/ov_system.c: the SDK function is replaced by symbol (host.h HOST_OVERRIDE), so the service behind it is
// never opened. metroid-prime-4-recomp runtime/ov_mp4.c has the same role for that game; none of its nine entries
// (nn::oe performance configuration and default display resolution, nn::hid gyroscope drift, six-axis rest and touch
// screen at the detail layer) is imported by this game's main or subsdk0 (checked against the NSO import tables,
// 2026-11-05), so this file starts empty, as MP4's did against FE3H's.
//
// Each entry names the call that needs it and the game version it was seen in.
#include "platform/platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "ov.h"

// ==== nn::oe (SDK 22): applet-environment calls this game makes that ov_system.c lacks. ov_system.c's Initialize never
// opens the applet manager session, so a lifted nn::oe call that reads it faults on a null pointer.
// void SetGraphicsScaleChangedNotificationEnabled(bool): asks the system to signal when the output's graphics scale
// changes (a Switch 2 notion); nnMain calls it first thing (main+0x1202d90, 1.1.1). The host never changes scale.
static void ov_void(Cpu *c) { (void)c; }

HOST_OVERRIDE("_ZN2nn2oe42SetGraphicsScaleChangedNotificationEnabledEb", ov_void);

// GraphicsScale GetCurrentGraphicsScale(): the output's graphics scale as an enum; the SDK maps the applet manager's
// reply 10/20/30 to 0/1/2 (IGraphicsScaleController command 10) and aborts on any other value. Lichii's controller
// answers 10 (its graphics_scale_controller.h, GPL-3.0, read for behaviour), so 0 here too; nnMain stores it beside the
// operation and performance modes (main+0xe376dc, 1.1.1). What the three scales mean on hardware is not known yet.
static void ov_current_graphics_scale(Cpu *c) { ret_x(c, 0); }

// void GetDisplayVersion(DisplayVersion *out): the 16-byte version string of the running application ("1.1.1").
static void ov_display_version(Cpu *c) {
    char *out = arg_p(c, 0);
    if (!out) rt_fatal("nn::oe::GetDisplayVersion: null out");
    memset(out, 0, 16);
    snprintf(out, 16, "%d.%d.%d", rt_game_version / 100, rt_game_version / 10 % 10, rt_game_version % 10);
}

HOST_OVERRIDE("_ZN2nn2oe23GetCurrentGraphicsScaleEv", ov_current_graphics_scale);
HOST_OVERRIDE("_ZN2nn2oe17GetDisplayVersionEPNS0_14DisplayVersionE", ov_display_version);

// ==== nn::vi (SDK 22): display capability queries and layer metadata setters ov_gfx.c lacks, which this game's
// graphics bring-up makes. ov_gfx.c's Initialize never opens the vi service, so the lifted calls would fault; the
// host window has one fixed-rate SDR output, so the queries answer "not supported" and the setters succeed.
// Result QueryDisplayHdrStatus(bool *enabled, HdrOutputFormat *format, uint16_t *unknown, const Display *)
static void ov_vi_hdr_status(Cpu *c) {
    uint8_t *enabled = arg_p(c, 0); int32_t *format = arg_p(c, 1); uint16_t *u = arg_p(c, 2);
    if (enabled) *enabled = 0;
    if (format) *format = 0;
    if (u) *u = 0;
    ret_x(c, 0);
}
// Result QueryDisplayVrrSupported(bool *out, const Display *): no variable refresh rate
static void ov_vi_bool_false(Cpu *c) {
    uint8_t *out = arg_p(c, 0);
    if (out) *out = 0;
    ret_x(c, 0);
}
// Result QueryDisplayHdrPaperWhite(uint16_t *nits, const Display *): SDR reference white
static void ov_vi_paper_white(Cpu *c) {
    uint16_t *out = arg_p(c, 0);
    if (out) *out = 203;
    ret_x(c, 0);
}
// Result SetDisplayVrrCapability(Display *, bool), SetDisplay120HzRefreshRateCapability(Display *, bool),
// SetLayerExtendedSwapInterval(Layer *, int, int), ClearLayerExtendedSwapInterval(Layer *), and the four
// SetLayerNextHdr*Metadata(Layer *, ...): accepted and ignored; the present (gfx_window.c) owns the output's rate and
// format.
static void ov_vi_result_ok(Cpu *c) { ret_x(c, 0); }

HOST_OVERRIDE("_ZN2nn2vi21QueryDisplayHdrStatusEPbPNS0_15HdrOutputFormatEPtPKNS0_7DisplayE", ov_vi_hdr_status);
HOST_OVERRIDE("_ZN2nn2vi24QueryDisplayVrrSupportedEPbPKNS0_7DisplayE", ov_vi_bool_false);
HOST_OVERRIDE("_ZN2nn2vi25QueryDisplayHdrPaperWhiteEPtPKNS0_7DisplayE", ov_vi_paper_white);
HOST_OVERRIDE("_ZN2nn2vi23SetDisplayVrrCapabilityEPNS0_7DisplayEb", ov_vi_result_ok);
HOST_OVERRIDE("_ZN2nn2vi36SetDisplay120HzRefreshRateCapabilityEPNS0_7DisplayEb", ov_vi_result_ok);
HOST_OVERRIDE("_ZN2nn2vi28SetLayerExtendedSwapIntervalEPNS0_5LayerEii", ov_vi_result_ok);
HOST_OVERRIDE("_ZN2nn2vi30ClearLayerExtendedSwapIntervalEPNS0_5LayerE", ov_vi_result_ok);
HOST_OVERRIDE("_ZN2nn2vi35SetLayerNextHdrContentGamutMetadataEPNS0_5LayerENS0_15HdrContentGamutE", ov_vi_result_ok);
HOST_OVERRIDE("_ZN2nn2vi39SetLayerNextHdrToSdrToneMappingMetadataEPNS0_5LayerENS0_14HdrToneMappingE", ov_vi_result_ok);
HOST_OVERRIDE("_ZN2nn2vi52SetLayerNextHdrExtendedContentLuminanceRangeMetadataEPNS0_5LayerEtt", ov_vi_result_ok);
HOST_OVERRIDE("_ZN2nn2vi52SetLayerNextHdrStandardContentLuminanceRangeMetadataEPNS0_5LayerEtt", ov_vi_result_ok);

// Result QueryDisplayCurrentRefreshRate(RefreshRate *hz, bool *flag, const Display *): the SDK's RefreshRate is a u32
// in Hz and it accepts 60 or 120 (nn::vi::detail::DefaultDisplay::QueryRefreshRate, sdk+0x5de520, 1.1.1); the bool's
// meaning is not known (kept false). The host presents at the game's 60 Hz pace whatever the monitor does.
static void ov_vi_refresh_rate(Cpu *c) {
    uint32_t *hz = arg_p(c, 0); uint8_t *flag = arg_p(c, 1);
    if (hz) *hz = 60;
    if (flag) *flag = 0;
    ret_x(c, 0);
}
HOST_OVERRIDE("_ZN2nn2vi30QueryDisplayCurrentRefreshRateEPNS0_11RefreshRateEPbPKNS0_7DisplayE", ov_vi_refresh_rate);

// ==== Offline and without add-on content. The stubs for nn::aoc, nn::nifm and nn::account below are from
// fe3h-pc-recomp runtime/ov_fe3h.c @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner),
// which this game imports again (MP4 imported none of them); nn::socket and nn::ssl follow the same pattern for the
// forms this game uses. OoT's framework init (main+0x3372a38, 1.1.1) counts the add-on content, and its NPLN client
// brings up nifm, socket and ssl.
// int nn::aoc::CountAddOnContent(): none (ov_system.c's ListAddOnContent lists none)
static void ov_aoc_count(Cpu *c) { ret_x(c, 0); }
HOST_OVERRIDE("_ZN2nn3aoc17CountAddOnContentEv", ov_aoc_count);

// nn::nifm: every network request ends without a connection; the failure is a nifm-module result (module 110).
#define NIFM_R_NO_CONNECTION (110u | (1u << 9))
static void ov_nifm_ok(Cpu *c) { ret_x(c, 0); }                       // Result Initialize()
static void ov_nifm_none(Cpu *c) { (void)c; }                         // void SubmitNetworkRequestAndWait(), CancelNetworkRequest()
static void ov_nifm_unavailable(Cpu *c) { ret_x(c, 0); }              // bool IsNetworkAvailable()
static void ov_nifm_fail(Cpu *c) { ret_x(c, NIFM_R_NO_CONNECTION); }  // Result GetCurrentProxySetting(out), HandleNetworkRequestErrorResult()
HOST_OVERRIDE("_ZN2nn4nifm10InitializeEv", ov_nifm_ok);
HOST_OVERRIDE("_ZN2nn4nifm18IsNetworkAvailableEv", ov_nifm_unavailable);
HOST_OVERRIDE("_ZN2nn4nifm27SubmitNetworkRequestAndWaitEv", ov_nifm_none);
HOST_OVERRIDE("_ZN2nn4nifm20CancelNetworkRequestEv", ov_nifm_none);
HOST_OVERRIDE("_ZN2nn4nifm31HandleNetworkRequestErrorResultEv", ov_nifm_fail);
HOST_OVERRIDE("_ZN2nn4nifm22GetCurrentProxySettingEPNS0_12ProxySettingE", ov_nifm_fail);

// nn::socket: Initialize succeeds without bsd:u; every call that would reach a socket fails with ENETDOWN.
#define SOCKET_ENETDOWN 50
static void ov_socket_ok(Cpu *c) { ret_x(c, 0); }                          // Result Initialize(config), Finalize()
static void ov_socket_initialized(Cpu *c) { ret_x(c, 1); }                 // bool IsInitialized()
static void ov_socket_fail(Cpu *c) { ret_x(c, (uint64_t)(int64_t)-1); }    // int / ssize_t calls: -1
static void ov_socket_errno(Cpu *c) { ret_x(c, SOCKET_ENETDOWN); }         // int GetLastError()
// Local descriptors work offline on hardware, and the game's gRPC (NPLN) aborts at start-up when none of its polling
// engines can make a wakeup descriptor ("No event engine could be initialized", main+0x21f0c7c). EventFd hands out
// fake descriptors from 0x1000; reads on them report one event, writes, fcntl, close and socket options succeed; Poll
// and Select sleep for the timeout (100 ms at most) and report nothing ready. Real sockets (Socket, Connect) stay down.
static volatile int32_t fake_fd_next = 0x1000;
static void ov_eventfd_new(Cpu *c) { ret_x(c, (uint32_t)__atomic_add_fetch(&fake_fd_next, 1, __ATOMIC_SEQ_CST)); }
static void ov_eventfd_read(Cpu *c) {   // int EventFdRead(int fd, uint64_t *value)
    uint64_t *v = arg_p(c, 1);
    if (v) *v = 1;
    ret_x(c, 0);
}
// The wait gives its virtual core up: the NPLN event-engine thread polls in a loop, and holding the core through
// each sleep starved the main thread, pinned to the same core at the same priority (2026-11-05: frames 10 to 50 s
// apart during loading, the main thread waiting for core 2 in every thread dump).
static void ov_poll_sleep(Cpu *c, int ms) {
    if (ms < 0 || ms > 100) ms = 100;
    if (ms) {
        int away = kern_core_away();
        plat_sleep_ns((uint64_t)ms * 1000000u);
        kern_core_back(&away);
    }
    ret_x(c, 0);
}
static void ov_socket_poll(Cpu *c) { ov_poll_sleep(c, (int)arg_x(c, 2)); }   // int Poll(pollfd *, nfds, int timeout_ms)
static void ov_socket_select(Cpu *c) {   // int Select(nfds, fd_set *r, *w, *e, TimeVal *tv)
    const int64_t *tv = arg_p(c, 4);
    ov_poll_sleep(c, tv ? (int)(tv[0] * 1000 + tv[1] / 1000) : -1);
}
HOST_OVERRIDE("_ZN2nn6socket10InitializeERKNS0_6ConfigE", ov_socket_ok);
HOST_OVERRIDE("_ZN2nn6socket8FinalizeEv", ov_socket_ok);
HOST_OVERRIDE("_ZN2nn6socket13IsInitializedEv", ov_socket_initialized);
HOST_OVERRIDE("_ZN2nn6socket12GetLastErrorEv", ov_socket_errno);
HOST_OVERRIDE("_ZN2nn6socket5CloseEi", ov_socket_ok);
HOST_OVERRIDE("_ZN2nn6socket10SetSockOptEiNS0_5LevelENS0_6OptionEPKvj", ov_socket_ok);
HOST_OVERRIDE("_ZN2nn6socket6SelectEiPNS0_5FdSetES2_S2_PNS0_7TimeValE", ov_socket_select);
HOST_OVERRIDE("_ZN2nn6socket7EventFdEmNS0_12EventFdFlagsE", ov_eventfd_new);
HOST_OVERRIDE("_ZN2nn6socket11EventFdReadEiPm", ov_eventfd_read);
HOST_OVERRIDE("_ZN2nn6socket12EventFdWriteEim", ov_socket_ok);


// nn::diag::detail::AbortImpl(condition, function, file, line): the SDK's assertion, seen once in three runs right
// after the first frame from nn::os::detail::MultiWaitImpl::InternalWaitAnyImpl (2026-11-06). Logged before the SDK
// goes on to its Break, which the host's fatal path does not survive on that thread yet.
static void obs_abort_impl(Cpu *c) {
    const char *cond = arg_p(c, 0), *func = arg_p(c, 1), *file = arg_p(c, 2);
    char lr[128];
    rt_log("nn::diag abort: \"%s\" in %s (%s:%d), from %s", cond ? cond : "?", func ? func : "?", file ? file : "?",
           (int)arg_x(c, 3), ldr_describe(c->x[30] - 4, lr, sizeof lr));
    fflush(stderr);
    rt_original(c);
}
HOST_OBSERVER("_ZN2nn4diag6detail9AbortImplEPKcS3_S3_i", obs_abort_impl);

// nn::ssl: Initialize succeeds without ssl:s; making a context or reading the built-in certificates fails with an
// ssl-module result (module 123). Context's constructor and destructor are local SDK code and stay lifted.
#define SSL_R_UNAVAILABLE (123u | (1u << 9))
static void ov_ssl_ok(Cpu *c) { ret_x(c, 0); }
static void ov_ssl_fail(Cpu *c) { ret_x(c, SSL_R_UNAVAILABLE); }
HOST_OVERRIDE("_ZN2nn3ssl10InitializeEv", ov_ssl_ok);
HOST_OVERRIDE("_ZN2nn3ssl8FinalizeEv", ov_ssl_ok);
HOST_OVERRIDE("_ZN2nn3ssl7Context6CreateENS1_10SslVersionE", ov_ssl_fail);
HOST_OVERRIDE("_ZN2nn3ssl7Context7DestroyEv", ov_ssl_ok);
HOST_OVERRIDE("_ZN2nn3ssl7Context9ImportCrlEPmPKcj", ov_ssl_fail);
HOST_OVERRIDE("_ZN2nn3ssl7Context15ImportClientPkiEPmPKcS4_jj", ov_ssl_fail);
HOST_OVERRIDE("_ZN2nn3ssl7Context15ImportServerPkiEPmPKcjNS0_17CertificateFormatE", ov_ssl_fail);
HOST_OVERRIDE("_ZN2nn3ssl14BuiltInManager28GetBuiltInCertificateBufSizeEPjPNS0_15CaCertificateIdEj", ov_ssl_fail);
HOST_OVERRIDE("_ZN2nn3ssl14BuiltInManager22GetBuiltInCertificatesEPPNS0_26BuiltInCertificateInfoBaseEPhjPNS0_15CaCertificateIdEj", ov_ssl_fail);

// nn::account network service account: none; the calls that would make or fetch an online identity fail with an
// account-module result (module 124). The local user comes from ov_system.c.
#define ACCOUNT_R_NSA_UNAVAILABLE (124u | (1u << 9))
static void ov_nsa_fail(Cpu *c) { ret_x(c, ACCOUNT_R_NSA_UNAVAILABLE); }
HOST_OVERRIDE("_ZN2nn7account36EnsureNetworkServiceAccountAvailableERKNS0_10UserHandleE", ov_nsa_fail);
HOST_OVERRIDE("_ZN2nn7account26GetNetworkServiceAccountIdEPNS0_23NetworkServiceAccountIdERKNS0_10UserHandleE", ov_nsa_fail);
HOST_OVERRIDE("_ZN2nn7account37LoadNetworkServiceAccountIdTokenCacheEPmPcmRKNS0_10UserHandleE", ov_nsa_fail);
HOST_OVERRIDE("_ZN2nn7account44EnsureNetworkServiceAccountIdTokenCacheAsyncEPNS0_12AsyncContextERKNS0_10UserHandleE", ov_nsa_fail);

// Result nn::pctl::CheckAgcPermissionUnderMinimumAge(const account::Uid &): no parental controls (ov_system.c), permitted
static void ov_pctl_ok(Cpu *c) { ret_x(c, 0); }
HOST_OVERRIDE("_ZN2nn4pctl33CheckAgcPermissionUnderMinimumAgeERKNS_7account3UidE", ov_pctl_ok);

// Result nn::prepo::PlayReport::Send(const account::Uid &): play reports stay local (ov_system.c); building one is
// local SDK code and runs lifted
static void ov_prepo_send(Cpu *c) { ret_x(c, 0); }
HOST_OVERRIDE("_ZN2nn5prepo10PlayReport4SendERKNS_7account3UidE", ov_prepo_send);

// ==== nn::hid, Switch 2 additions: the Joy-Con mouse sensor (Jcms), the mouse devices (Nmouse) and sensor fusion. The
// host has none of these devices: handle lookups give handle 0 and no devices, start and stop succeed, finalizers are
// no-ops. The game's input module queries them at start-up (1.1.1).
static void ov_hid_none(Cpu *c) { ret_x(c, 0); }   // handles and counts of 0, Result 0
static void ov_hid_nothing(Cpu *c) { (void)c; }
static void ov_nmouse_type_set(Cpu *c) {   // void GetNmouseTypeSet(BitFlagSet<32, NmouseType> *out, NmouseHandle)
    uint32_t *out = arg_p(c, 0);
    if (out) *out = 0;
}
HOST_OVERRIDE("_ZN2nn3hid13GetJcmsHandleERKjRKNS0_8JcmsTypeE", ov_hid_none);
HOST_OVERRIDE("_ZN2nn3hid9StartJcmsERKNS0_10JcmsHandleE", ov_hid_none);
HOST_OVERRIDE("_ZN2nn3hid8StopJcmsERKNS0_10JcmsHandleE", ov_hid_none);
HOST_OVERRIDE("_ZN2nn3hid12FinalizeJcmsEv", ov_hid_nothing);
HOST_OVERRIDE("_ZN2nn3hid16GetNmouseHandlesEPNS0_12NmouseHandleEij", ov_hid_none);
HOST_OVERRIDE("_ZN2nn3hid16GetNmouseTypeSetEPNS_4util10BitFlagSetILi32ENS0_10NmouseTypeEEENS0_12NmouseHandleE", ov_nmouse_type_set);
HOST_OVERRIDE("_ZN2nn3hid14FinalizeNmouseEv", ov_hid_nothing);
HOST_OVERRIDE("_ZN2nn3hid22GetSensorFusionHandlesEPNS0_18SensorFusionHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE", ov_hid_none);
HOST_OVERRIDE("_ZN2nn3hid20FinalizeSensorFusionEv", ov_hid_nothing);

// ==== The C APIs of nn::socket and nn::ssl (nnsocket*, nnssl*), which this game's NPLN client uses beside the C++
// forms above; the same offline answers. The byte-order helpers (nnsocketInet*), nnsocketGAIStrError and nnosXxx are
// local SDK code and run lifted.
static void ov_c_socket_fail(Cpu *c) { ret_x(c, (uint64_t)(int64_t)-1); }        // int calls: -1, errno ENETDOWN
static void ov_c_socket_errno(Cpu *c) { ret_x(c, SOCKET_ENETDOWN); }               // int nnsocketGetLastErrno()
static void ov_c_socket_void(Cpu *c) { (void)c; }                                  // nnsocketSetLastErrno, nnsocketFreeAddrInfo
static void ov_c_socket_gai_fail(Cpu *c) { ret_x(c, 4); }                          // nnsocketGetAddrInfo*: EAI_FAIL
static void ov_c_ssl_ok(Cpu *c) { ret_x(c, 0); }
static void ov_c_ssl_fail(Cpu *c) { ret_x(c, SSL_R_UNAVAILABLE); }
HOST_OVERRIDE("nnsocketAccept", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketBind", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketClose", ov_socket_ok);
HOST_OVERRIDE("nnsocketConnect", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketFcntl", ov_socket_ok);
HOST_OVERRIDE("nnsocketGetPeerName", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketGetSockName", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketGetSockOpt", ov_socket_ok);
HOST_OVERRIDE("nnsocketListen", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketPoll", ov_socket_poll);
HOST_OVERRIDE("nnsocketRecv", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketRecvMsg", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketResolverGetOption", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketResolverSetOption", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketSend", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketSendMsg", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketSetSockOpt", ov_socket_ok);
HOST_OVERRIDE("nnsocketShutdown", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketSocket", ov_c_socket_fail);
HOST_OVERRIDE("nnsocketGetLastErrno", ov_c_socket_errno);
HOST_OVERRIDE("nnsocketSetLastErrno", ov_c_socket_void);
HOST_OVERRIDE("nnsocketFreeAddrInfo", ov_c_socket_void);
HOST_OVERRIDE("nnsocketGetAddrInfo", ov_c_socket_gai_fail);
HOST_OVERRIDE("nnsocketGetAddrInfoWithOptions", ov_c_socket_gai_fail);
HOST_OVERRIDE("nnsslInitialize", ov_c_ssl_ok);
HOST_OVERRIDE("nnsslFinalize", ov_c_ssl_ok);
HOST_OVERRIDE("nnsslContextDestroy", ov_c_ssl_ok);
HOST_OVERRIDE("nnsslConnectionDestroy", ov_c_ssl_ok);
HOST_OVERRIDE("nnsslConnectionCreate", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionDoHandshakeWithCertBuffer", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionGetConnectionId", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionGetNextAlpnProto", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionGetServerCertDetail", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionGetVerifyCertError", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionPeek", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionPending", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionRead", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionSetHostName", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionSetIoMode", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionSetNextAlpnProto", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionSetOption", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionSetServerCertBuffer", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionSetSessionCacheMode", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionSetSocketDescriptor", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionSetVerifyOption", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslConnectionWrite", ov_c_ssl_fail);
HOST_OVERRIDE("nnsslContextGetContextId", ov_c_ssl_fail);

// ==== nn::nsd: server environment discovery, which would open nsd:a. Result GetEnvironmentIdentifier(
// EnvironmentIdentifier *out): the retail environment "lp1" (8 bytes, as yuzu and Lichii answer); Result ResolveEx(Fqdn
// *out, const Fqdn &in): no server-name rewriting, the name stays (Fqdn is 0x100 bytes). The NPLN client asks for both
// at start-up (main+0x241f494, 1.1.1); offline, nothing resolves afterwards anyway.
static void ov_nsd_environment(Cpu *c) {
    char *out = arg_p(c, 0);
    if (out) { memset(out, 0, 8); memcpy(out, "lp1", 3); }
    ret_x(c, 0);
}
static void ov_nsd_resolve(Cpu *c) {
    char *out = arg_p(c, 0); const char *in = arg_p(c, 1);
    if (out && in && out != in) memmove(out, in, 0x100);
    ret_x(c, 0);
}
HOST_OVERRIDE("_ZN2nn3nsd24GetEnvironmentIdentifierEPNS0_21EnvironmentIdentifierE", ov_nsd_environment);
HOST_OVERRIDE("_ZN2nn3nsd9ResolveExEPNS0_4FqdnERKS1_", ov_nsd_resolve);

// OOT_MQ_LOG=1 (diagnostic, 2026-11-06): the traffic of nn::os::LightMessageQueue, to see what a thread that polls one
// with TryReceive in a sleep loop (the main thread after the first video frame, main+0xdd2680) waits for and who
// should post it. Sends log before the call; receives after, with the message when one arrived.
static int mq_log_on(void) {
    static int on = -1;
    if (on < 0) on = getenv("OOT_MQ_LOG") != NULL;
    return on;
}
static void mq_send(Cpu *c, const char *what) {
    if (mq_log_on()) {
        char t[96], lr[96];
        rt_log("mq: %s(%#llx, %#llx) by %s from %s", what, (unsigned long long)arg_x(c, 0), (unsigned long long)arg_x(c, 1),
               kern_thread_describe(t, sizeof t), ldr_describe(c->x[30] - 4, lr, sizeof lr));
    }
    rt_original(c);
}
static void mq_receive(Cpu *c, const char *what, int try) {
    uint64_t *out = arg_p(c, 0), q = arg_x(c, 1), lr = c->x[30] - 4;
    rt_original(c);
    if (mq_log_on() && (!try || (c->x[0] & 1))) {
        char t[96], l[96];
        rt_log("mq: %s(%#llx) -> %#llx by %s from %s", what, (unsigned long long)q, out ? (unsigned long long)*out : 0,
               kern_thread_describe(t, sizeof t), ldr_describe(lr, l, sizeof l));
    }
}
static void obs_mq_send(Cpu *c) { mq_send(c, "Send"); }
static void obs_mq_trysend(Cpu *c) { mq_send(c, "TrySend"); }
static void obs_mq_jam(Cpu *c) { mq_send(c, "Jam"); }
static void obs_mq_tryjam(Cpu *c) { mq_send(c, "TryJam"); }
static void obs_mq_receive(Cpu *c) { mq_receive(c, "Receive", 0); }
static void obs_mq_tryreceive(Cpu *c) { mq_receive(c, "TryReceive", 1); }
HOST_OBSERVER("_ZN2nn2os21SendLightMessageQueueEPNS0_21LightMessageQueueTypeEm", obs_mq_send);
HOST_OBSERVER("_ZN2nn2os24TrySendLightMessageQueueEPNS0_21LightMessageQueueTypeEm", obs_mq_trysend);
HOST_OBSERVER("_ZN2nn2os20JamLightMessageQueueEPNS0_21LightMessageQueueTypeEm", obs_mq_jam);
HOST_OBSERVER("_ZN2nn2os23TryJamLightMessageQueueEPNS0_21LightMessageQueueTypeEm", obs_mq_tryjam);
HOST_OBSERVER("_ZN2nn2os24ReceiveLightMessageQueueEPmPNS0_21LightMessageQueueTypeE", obs_mq_receive);
HOST_OBSERVER("_ZN2nn2os27TryReceiveLightMessageQueueEPmPNS0_21LightMessageQueueTypeE", obs_mq_tryreceive);

// OOT_MQ_LOG=1 also traces nn::os::EventType: the main thread waits on one after the cinematic's second frame pair
// (WaitEvent from main+0x11e7a6c) and nobody signals it.
static void ev_log(Cpu *c, const char *what, uint64_t obj, uint64_t lr) {
    char t[96], l[96];
    rt_log("ev: %s(%#llx) by %s from %s", what, (unsigned long long)obj, kern_thread_describe(t, sizeof t), ldr_describe(lr, l, sizeof l));
}
static void obs_ev_signal(Cpu *c) { if (mq_log_on()) ev_log(c, "Signal", arg_x(c, 0), c->x[30] - 4); rt_original(c); }
static void obs_ev_clear(Cpu *c) { if (mq_log_on()) ev_log(c, "Clear", arg_x(c, 0), c->x[30] - 4); rt_original(c); }
static void obs_ev_wait(Cpu *c) {
    uint64_t obj = arg_x(c, 0), lr = c->x[30] - 4;
    if (mq_log_on()) ev_log(c, "Wait", obj, lr);
    rt_original(c);
    if (mq_log_on()) ev_log(c, "Waited", obj, lr);
}
static void obs_ev_timedwait(Cpu *c) {
    uint64_t obj = arg_x(c, 0), lr = c->x[30] - 4;
    rt_original(c);
    if (mq_log_on() && (c->x[0] & 1)) ev_log(c, "TimedWait ok", obj, lr);
}
static void obs_ev_trywait(Cpu *c) {
    uint64_t obj = arg_x(c, 0), lr = c->x[30] - 4;
    rt_original(c);
    if (mq_log_on() && (c->x[0] & 1)) ev_log(c, "TryWait ok", obj, lr);
}
HOST_OBSERVER("_ZN2nn2os11SignalEventEPNS0_9EventTypeE", obs_ev_signal);
HOST_OBSERVER("_ZN2nn2os10ClearEventEPNS0_9EventTypeE", obs_ev_clear);
HOST_OBSERVER("_ZN2nn2os9WaitEventEPNS0_9EventTypeE", obs_ev_wait);
HOST_OBSERVER("_ZN2nn2os14TimedWaitEventEPNS0_9EventTypeENS_8TimeSpanE", obs_ev_timedwait);
HOST_OBSERVER("_ZN2nn2os12TryWaitEventEPNS0_9EventTypeE", obs_ev_trywait);
