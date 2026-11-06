// Written for ocarina-of-time-recomp (MIT, Copyright (c) 2026 Jeff Aigner).
//
// SM86 (Ampere) to Maxwell lowering. The Switch 2's NVN2 ships shader code for the T239's Ampere SM86: 128-bit
// instructions, the opcode in the low 12 bits of word 0, no scheduling words, a uniform datapath (UR0-UR63, UP0-UP6).
// The vendored Ryujinx translator decodes Maxwell (SM5x): 64-bit instructions, a scheduling word every three, and
// the forms its InstTable.cs lists. This file does not reinterpret the Ampere bytes: for each SM86 instruction it
// emits new Maxwell instructions that compute the same thing, keeps an address map so branches and the BRX jump
// tables (constant bank 1) land on the moved code, and maps the uniform registers onto general registers the
// program leaves free. Unknown forms fail with a message naming them, never guess: tools/shader_census.py --lower
// runs it over every program the game ships and groups the failures, which is the work list.
//
// Provenance of the encodings. SM86 fields: Mesa's NAK compiler, src/nouveau/compiler/nak/sm70_encode.rs (MIT,
// Copyright 2022 Collabora, Ltd.), its SM70Encoder (encode_alu_base: dst 16-23, src0 24-31, the second and third
// sources at 32-39 and 64-71 or, by the form in bits 9-11, a 32-bit immediate, a constant-buffer word (bytes 38-53,
// bank 54-58) or a uniform register (32-37, bit 91) in the wide slot; predicate 12-15) and its per-op encoders.
// Maxwell fields: the vendored Ryujinx Decoders/InstDecoders.cs and InstTable.cs (MIT). Lichii's Ampere front end
// (src/shader_recompiler/frontend/ampere/lower_to_maxwell.h, GPL-3.0) was read for behaviour only: which forms
// Nintendo's compiler emits, that BSSY/BSYNC/BREAK/WARPSYNC and DEPBAR need nothing, that a BRA's low offset bits
// are its .DIV/.CONV mode, that BRX tables hold absolute native offsets, that CALL.REL.NOINC pushes no return
// address, that SM70+ MUFU.SIN/COS take turns, and that IPA's attribute field counts dwords.
//
// Layout of the output (what Ryujinx's Decoder expects): graphics programs start with the 0x50-byte Maxwell program
// header, which is the first 0x50 bytes of the 0x80-byte Ampere one (the common words and the input/output maps are
// laid out alike); instructions follow in groups of three behind a scheduling word (an address that is a multiple
// of 32 from the first group is a scheduling word; they are zero, Ryujinx ignores them). Compute programs start with
// the first group; graphics programs' first group follows the header (addr_of).
// Branch targets are addresses of instructions; a relative branch adds its 24-bit offset to its own address + 8.
#include "sm86_lower.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t w[4]; } Insn;

static uint32_t bits(const Insn *i, unsigned off, unsigned n) {
    uint64_t v = i->w[off / 32];
    if (off / 32 + 1 < 4) v |= (uint64_t)i->w[off / 32 + 1] << 32;
    return (uint32_t)((v >> (off % 32)) & ((1ull << n) - 1));
}
static int64_t sext(uint64_t v, unsigned n) { return (int64_t)(v << (64 - n)) >> (64 - n); }

int sm86_control_parse(const uint8_t *control, uint32_t control_size, Sm86Control *c) {
    uint32_t magic;
    if (!control || control_size < SM86_CONTROL_MIN) return 0;
    memcpy(&magic, control, 4);
    if (magic != SM86_CONTROL_MAGIC) return 0;
    memcpy(&c->size, control + 0xd4, 4);
    memcpy(&c->code_end, control + 0x38, 4);
    memcpy(&c->c1_len, control + 0x3c, 4);
    memcpy(&c->c1_off, control + 0x40, 4);
    memcpy(&c->code_size, control + 0x44, 4);
    c->stage = control[0x56];
    memcpy(c->local, control + 0x58, 12);
    if (c->size < SM86_CONTROL_MIN || c->size > control_size) c->size = SM86_CONTROL_MIN;
    uint32_t header = c->stage == 5 ? 0 : SM86_HEADER;
    if (c->stage > 7 || c->code_end <= header || c->code_end % 16 || c->c1_off < c->code_end ||
        c->c1_off + c->c1_len < c->c1_off || c->c1_off + c->c1_len > c->code_size)
        return 0;
    return 1;
}

// ---- the lowering context

#define NSCRATCH 12
#define RZ 255u
#define PT 7u
#define NOREG 0xffffffffu

typedef struct {
    const Insn *in;
    uint32_t n;
    int graphics;                                  // output starts with the 0x50-byte header
    uint32_t stage;
    uint64_t *out;
    uint32_t nout, cap;
    uint32_t *first_out;                           // [n + 1]: output index of a native instruction's first word
    struct { uint32_t out, target; } *reloc;       // 24-bit relative branches to patch (target: native index)
    uint32_t nreloc, reloc_cap;
    uint32_t imm_reloc_cap, nimm_reloc;            // MOV32I immediates that are native return addresses
    struct { uint32_t out, target; } *imm_reloc;
    uint32_t max_gpr;                              // highest general register the program touches
    uint32_t ur_next, scratch_base;                // the pool above max_gpr: uniform registers from ur_next up
                                                   // to scratch_base, scratch registers from there (allocated
                                                   // on first use, ascending so pairs and runs stay adjacent)
    int nscratch;
    uint8_t ur_map[64];                            // uniform register -> general register (0 = unmapped)
    struct { uint8_t kind, bank; uint32_t value; } ur_def[64];   // 0 none, 1 a constant word (bank, offset),
                                                   // 2 an immediate, 3 several definitions or a computed one
    const uint8_t *control, *c1;                   // the control section and jump tables (BRX successors)
    uint32_t ctl_size, c1_len;
    uint8_t *leader;                               // [n]: the instruction starts a basic block
    uint64_t (*live_out)[4];                       // [n]: registers live after the instruction (liveness mode)
    int liveness;                                  // scratch comes from the registers dead at each instruction
    uint32_t scratch_len;                          // registers in the scratch block (liveness mode: per instruction)
    uint64_t reserved[4];                          // registers the lowering holds for the program's length
    struct { uint8_t kind, bank; uint32_t value; } cur_def[64];   // local forwarding: the constant uniform
                                                   // register n holds at this point of its block (kind as ur_def)
    uint8_t needs_reg[64];                         // uniform register n has a use no forwarding covers
    uint8_t ur_pair[64];                           // n is half of a 64-bit pair (UIMAD.WIDE, ULDC.64): adjacent registers
    uint32_t ur_first[64], ur_last[64];            // the instructions a uniform register is touched at, first and last
    uint8_t ur_whole[64];                          // a use before its first definition: live over the whole program
    int scanned;                                   // needs_reg is final: a read of another register is a scan bug
    uint32_t used_preds;                           // predicates the program touches (bit n = Pn)
    uint8_t up_map[8];                             // uniform predicate -> predicate (0xff unmapped, 7 = UPT)
    int force_swap;                                // up_swap although predicates are free (a retry)
    int up_swap;                                   // no predicate is free: UPn lives in bit n of up_reg and is
                                                   // swapped into Pn around each instruction naming it
    uint32_t up_reg;
    int up_reg_set;                                // up_reg is allocated (register 0 is a valid choice)
    uint64_t pred;                                 // the current instruction's predicate, in Maxwell's bits 16-19
    int tensor;                                    // HMMA to the MMA marker (sm86_lower_tensor_cores, compute)
    int hmma_count;                                // HMMAs lowered so far (OOT_TENSOR_HMMA_MASK)
    uint32_t index;                                // the current native instruction
    int failed;
    char error[96];
    uint32_t error_index;
} L;

static int fail(L *l, const char *msg) {
    if (!l->failed) {
        l->failed = 1;
        l->error_index = l->index;
        snprintf(l->error, sizeof l->error, "%s", msg);
    }
    return 0;
}

// Address of output instruction k (Maxwell's view: the scheduling words at every 32 bytes from the first
// instruction group, which follows the 0x50-byte header of a graphics program directly: Ryujinx Decoder.FillBlock
// counts from Translator's address + HeaderSize. 2026-11-06: counted from the block start instead, every third
// instruction of every vertex and fragment shader sat where the decoder skips a scheduling word.)
static uint32_t addr_of(const L *l, uint32_t k) {
    return (l->graphics ? 0x50 : 0) + 8 + 8 * k + 8 * (k / 3);
}

static void emit(L *l, uint64_t word) {
    if (l->nout == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 4096;
        l->out = realloc(l->out, l->cap * sizeof *l->out);
        if (!l->out) { fail(l, "out of memory"); l->cap = 0; return; }
    }
    if (l->out) l->out[l->nout++] = word;
}

// A branch at the next emitted word to native instruction target (patched at the end).
static void reloc(L *l, uint32_t target) {
    if (l->nreloc == l->reloc_cap) {
        l->reloc_cap = l->reloc_cap ? l->reloc_cap * 2 : 256;
        l->reloc = realloc(l->reloc, l->reloc_cap * sizeof *l->reloc);
        if (!l->reloc) { fail(l, "out of memory"); l->reloc_cap = 0; return; }
    }
    if (l->reloc) l->reloc[l->nreloc].out = l->nout, l->reloc[l->nreloc].target = target, l->nreloc++;
}

static uint32_t alloc_gpr(L *l) {
#ifdef SM86_TOOL
    if (getenv("SM86_POOL")) {
        fprintf(stderr, "alloc: ur_next %u scratch_base %u max_gpr %u up_reg %u; needs:", l->ur_next, l->scratch_base, l->max_gpr, l->up_reg);
        for (int n = 0; n < 63; n++) if (l->needs_reg[n]) fprintf(stderr, " UR%d(k%d%s)", n, l->ur_def[n].kind, l->ur_map[n] ? "=R" : "");
        fprintf(stderr, "\n");
    }
#endif
    if (l->ur_next >= l->scratch_base) return fail(l, "no general register free for a uniform register"), NOREG;
    l->reserved[l->ur_next / 64] |= 1ull << (l->ur_next % 64);
    return l->ur_next++;
}

// Scratch register k of the current expansion: scratch_base + k, the block growing on first use.
static uint32_t S(L *l, int k) {
    if (k >= NSCRATCH || l->scratch_base + (uint32_t)k > 254 || (l->liveness && (uint32_t)k >= l->scratch_len))
        return fail(l, "no general register free for scratch"), RZ;
    if (l->nscratch <= k) l->nscratch = k + 1;
    return l->scratch_base + (uint32_t)k;
}

typedef struct Src Src;
static void to_reg_into(L *l, Src s, uint32_t r);
static void mov_imm(L *l, uint32_t rd, uint32_t v);
struct Src {
    int kind;                 // 0 register, 1 immediate, 2 constant buffer
    uint32_t reg, imm, cb_off, cb_bank;
    int neg, abs_;
};

static Src reg_src(uint32_t r) { Src s = {0}; s.reg = r; return s; }

// The general register standing in for uniform register n (URZ = RZ).
// Liveness mode: a register neither reserved nor live anywhere in instructions a..b (NOREG when there is none).
static uint32_t alloc_dead(L *l, uint32_t a, uint32_t b) {
    uint64_t busy[4];
    memcpy(busy, l->reserved, sizeof busy);
    for (uint32_t i = a; i <= b && i < l->n; i++)
        for (int w = 0; w < 4; w++) busy[w] |= l->live_out[i][w];
    for (uint32_t c = 0; c < 255; c++)
        if (!(busy[c / 64] >> (c % 64) & 1)) { l->reserved[c / 64] |= 1ull << (c % 64); return c; }
    return NOREG;
}

static uint32_t ur(L *l, uint32_t n) {
    if (n == 63) return RZ;
    if (n > 63) return fail(l, "uniform register number"), RZ;
    if (l->scanned && !l->needs_reg[n]) return fail(l, "uniform register read the scan did not see"), RZ;
    if (!l->ur_map[n]) {
        uint32_t r = NOREG;
        if (l->ur_pair[n] && l->needs_reg[n ^ 1] && !l->ur_map[n ^ 1] && l->ur_next + 1 < l->scratch_base) {
            // Both halves of a pair at once, so they land in adjacent registers whatever the order of first use.
            uint32_t e = n & ~1u, r0 = alloc_gpr(l), r1 = alloc_gpr(l);
            if (r0 == NOREG || r1 == NOREG) return RZ;
            l->ur_map[e] = (uint8_t)r0, l->ur_map[e + 1] = (uint8_t)r1;
            return l->ur_map[n];
        }
        if (l->ur_next < l->scratch_base) r = alloc_gpr(l);
        else if (l->liveness) {
            // No room above the program's registers: one dead over the instructions the uniform register spans
            // (the whole program when it is read before any definition); a pair takes two adjacent ones dead over
            // both halves' spans.
            uint32_t a = l->ur_whole[n] || !l->ur_first[n] ? 0 : l->ur_first[n] - 1;
            uint32_t b = l->ur_whole[n] || !l->ur_last[n] ? l->n - 1 : l->ur_last[n] - 1;
            if (l->ur_pair[n] && l->needs_reg[n ^ 1] && !l->ur_map[n ^ 1]) {
                uint32_t m = n ^ 1, e = n & ~1u;
                uint32_t a2 = l->ur_whole[m] || !l->ur_first[m] ? 0 : l->ur_first[m] - 1;
                uint32_t b2 = l->ur_whole[m] || !l->ur_last[m] ? l->n - 1 : l->ur_last[m] - 1;
                if (a2 < a) a = a2;
                if (b2 > b) b = b2;
                uint64_t busy[4];
                memcpy(busy, l->reserved, sizeof busy);
                for (uint32_t i = a; i <= b && i < l->n; i++)
                    for (int w = 0; w < 4; w++) busy[w] |= l->live_out[i][w];
                for (uint32_t c = 0; c + 1 < 255; c++)
                    if (!(busy[c / 64] >> (c % 64) & 1) && !(busy[(c + 1) / 64] >> ((c + 1) % 64) & 1)) {
                        l->reserved[c / 64] |= 1ull << (c % 64), l->reserved[(c + 1) / 64] |= 1ull << ((c + 1) % 64);
                        l->ur_map[e] = (uint8_t)c, l->ur_map[e + 1] = (uint8_t)(c + 1);
                        return l->ur_map[n];
                    }
                return fail(l, "no register pair dead over a uniform pair's span"), RZ;
            }
            r = alloc_dead(l, a, b);
            if (r == NOREG) return fail(l, "no general register dead over a uniform register's span"), RZ;
        } else {
            return fail(l, "no general register free for a uniform register"), RZ;
        }
        l->ur_map[n] = (uint8_t)r;
    }
    return l->ur_map[n];
}

// Uniform register n as an operand: a constant word or an immediate when its only definition loads one (no
// register needed), else its register.
static Src ur_src(L *l, uint32_t n) {
    Src s = {0};
    if (n < 63 && l->ur_def[n].kind == 1) { s.kind = 2, s.cb_bank = l->ur_def[n].bank, s.cb_off = l->ur_def[n].value; return s; }
    if (n < 63 && l->ur_def[n].kind == 2) { s.kind = 1, s.imm = l->ur_def[n].value; return s; }
    if (n < 63 && l->cur_def[n].kind == 1) { s.kind = 2, s.cb_bank = l->cur_def[n].bank, s.cb_off = l->cur_def[n].value; return s; }
    if (n < 63 && l->cur_def[n].kind == 2) { s.kind = 1, s.imm = l->cur_def[n].value; return s; }
    return reg_src(ur(l, n));
}

// Does uniform register n have a register at all (false: every use forwards a constant, or there is no use)?
static int ur_has_reg(const L *l, uint32_t n) { return n < 63 && l->ur_def[n].kind == 3 && l->needs_reg[n]; }

// The register a definition of uniform register n writes: RZ when it has none (the value is never read as such).
static uint32_t ur_dst(L *l, uint32_t n) { return ur_has_reg(l, n) ? ur(l, n) : RZ; }

// Uniform registers n and n + 1 as an adjacent register pair, in scratch k, k + 1 when they are forwarded
// constants or not adjacent in the mapping.
static uint32_t ur_pair_in(L *l, uint32_t n, int k, uint32_t hint, int even) {
    if (n >= 63) return fail(l, "uniform register pair"), NOREG;
    Src lo = ur_src(l, n), hi = ur_src(l, n + 1);
    if (lo.kind == 0 && hi.kind == 0 && hi.reg == lo.reg + 1 && !(even && (lo.reg & 1))) return lo.reg;
    // hint: two adjacent registers the caller can spare for the pair (a load's destination), used instead of scratch.
    // even: the pair must start on an even register (RED's address); the scratch pair moves up by one when it does not.
    uint32_t a = hint != NOREG ? hint : S(l, k);
    if (hint == NOREG && even && (a & 1)) a = S(l, ++k);
    uint32_t b = hint != NOREG ? hint + 1 : S(l, k + 1);
    if (b != a + 1) return fail(l, "scratch pair"), NOREG;
    if (lo.kind == 0 && lo.reg == RZ) mov_imm(l, a, 0); else to_reg_into(l, lo, a);
    if (hi.kind == 0 && hi.reg == RZ) mov_imm(l, b, 0); else to_reg_into(l, hi, b);
    return a;
}
static uint32_t ur_pair(L *l, uint32_t n, int k) { return ur_pair_in(l, n, k, NOREG, 0); }

// Uniform register n read by a uniform-datapath op: its register, or a forwarded constant brought into scratch k.
static uint32_t ur_in(L *l, uint32_t n, int k) {
    Src s = ur_src(l, n);
    if (s.kind == 0) return s.reg;
    uint32_t r = S(l, k);
    to_reg_into(l, s, r);
    return r;
}

// The predicate standing in for uniform predicate n (UPT = PT): one the program never touches.
static uint32_t up(L *l, uint32_t n) {
    if (n >= 7) return PT;
    if (l->up_swap) return n;
    if (l->up_map[n] == 0xff) {
        uint32_t k = 0;
        while (k < 7 && (l->used_preds >> k & 1)) k++;
        if (k == 7) return fail(l, "no predicate free for a uniform predicate"), PT;
        l->used_preds |= 1u << k;
        l->up_map[n] = (uint8_t)k;
    }
    return l->up_map[n];
}

// ---- register scan: which general registers the program uses (the pool for the lowering lies above them)

static int uniform_op(uint32_t op);
static uint32_t up_fields(const Insn *x);
static int successors(const L *l, uint32_t i, uint32_t *out, uint32_t cap);
static void scratch_at(L *l, uint32_t i);

static int uniform_op(uint32_t op) {
    uint32_t base = op & 0x1ff;
    return (base >= 0x080 && base < 0x100) || base == 0x1c2 || base == 0x1c3 || base == 0x1c4 || op == 0x958;
}

static void scan_registers(L *l) {
    uint32_t m = 0;
    for (uint32_t i = 0; i < l->n; i++) {
        const Insn *x = &l->in[i];
        uint32_t op = x->w[0] & 0xfff, base = op & 0x1ff, form = op >> 9;
        if (uniform_op(op)) continue;                              // every register field is a uniform register
        if (base >= 0x140 && base < 0x160 && base != 0x149) continue;   // branches, EXIT, NOP...: immediates
        if (bits(x, 12, 4) == 15) continue;                        // @!PT: never executes
        uint32_t f[4] = {bits(x, 16, 8), bits(x, 24, 8), bits(x, 32, 8), bits(x, 64, 8)};
        int b_is_reg = base < 0x100 ? form == 1 || form == 2 || form == 3 || form == 7 : 1;
        if (base >= 0x100 && (form == 4 || form == 5 || form == 6 || op == 0xf60)) b_is_reg = 0;
        if (base == 0x182 || base == 0x181 || base == 0x183 || base == 0x184) b_is_reg = base != 0x182 && !bits(x, 91, 1);
        if (base == 0x19b) b_is_reg = 1;                           // SUST: the data register at 32-37 beside the handle's constant
        for (int k = 0; k < 4; k++) {
            if (k == 2 && !b_is_reg) continue;
            if (f[k] != RZ && f[k] > m) m = f[k];
        }
    }
    l->max_gpr = m;
    l->ur_next = m + 4 > 255 ? 255 : m + 4;                        // + 3 for 128-bit operands behind a field
}

// ---- control flow: basic blocks and successors

static int64_t branch_target_of(uint32_t index, const Insn *x, int bra) {
    uint64_t raw = (bra ? (x->w[1] & ~3u) : x->w[1]) | ((uint64_t)(x->w[2] & 0x3ffff) << 32);
    return (int64_t)index * 16 + 16 + sext(raw, 50);
}

// The jump-table record of the BRX at native index i (the records name it by its offset from the first
// instruction): its entries as native indices (count returned, 0 unknown).
static uint32_t brx_targets(const L *l, uint32_t i, uint32_t *out, uint32_t cap) {
    uint32_t count = 0;
    if (!l->control || l->ctl_size < SM86_CONTROL_MIN + 4) return 0;
    memcpy(&count, l->control + SM86_CONTROL_MIN, 4);
    if (count > 64 || SM86_CONTROL_MIN + 4 + count * 16 > l->ctl_size) return 0;
    for (uint32_t r = 0; r < count; r++) {
        uint32_t rec[4];
        memcpy(rec, l->control + SM86_CONTROL_MIN + 4 + r * 16, 16);
        if (rec[0] != i * 16) continue;                       // the offset counts from the first instruction
        uint32_t n = 0;
        for (uint32_t e = 0; e < rec[1] && n < cap; e++) {
            uint32_t v;
            if (rec[3] + e * 4 + 4 > l->c1_len) return 0;
            memcpy(&v, l->c1 + rec[3] + e * 4, 4);
            if (v % 16 || v / 16 >= l->n) return 0;
            out[n++] = v / 16;
        }
        return n;
    }
    return 0;
}

// Successors of native instruction i (count; -1 when they cannot be known: a jump table without its record).
static int successors(const L *l, uint32_t i, uint32_t *out, uint32_t cap) {
    const Insn *x = &l->in[i];
    uint32_t op = x->w[0] & 0xfff, g = bits(x, 12, 4);
    int n = 0;
    if (g == 15) { if (i + 1 < l->n) out[n++] = i + 1; return n; }
    switch (op) {
    case 0x947: case 0x944: {
        int64_t t = branch_target_of(i, x, op == 0x947);
        if (t < 0 || t % 16 || (uint64_t)t / 16 > l->n) return -1;
        uint32_t mode = op == 0x947 ? bits(x, 32, 2) : 0;
        if (mode != 2 && (uint64_t)t / 16 < l->n) out[n++] = (uint32_t)(t / 16);
        if ((g != 7 || op == 0x944 || mode == 2) && i + 1 < l->n) out[n++] = i + 1;
        return n;
    }
    case 0x94d:
        if (g != 7 && i + 1 < l->n) out[n++] = i + 1;
        return n;
    case 0x949: case 0x958: {
        uint32_t t = brx_targets(l, i, out, cap);
        if (!t) return -1;
        n = (int)t;
        if (g != 7 && i + 1 < l->n && n < (int)cap) out[n++] = i + 1;
        return n;
    }
    case 0x950:
        // RET.REL.NODEC returns to the instruction after some CALL: every call site's successor.
        for (uint32_t k = 0; k < l->n && n < (int)cap; k++)
            if ((l->in[k].w[0] & 0xfff) == 0x944 && k + 1 < l->n) out[n++] = k + 1;
        return n;
    default:
        if (i + 1 < l->n) out[n++] = i + 1;
        return n;
    }
}

// Block leaders: the entry, every branch target, every instruction after a branch, call, return or exit.
// Returns 0 when a jump table's targets are unknown.
static int build_blocks(L *l) {
    l->leader = calloc(l->n, 1);
    if (!l->leader) return fail(l, "out of memory"), 0;
    l->leader[0] = 1;
    uint32_t t[72];
    for (uint32_t i = 0; i < l->n; i++) {
        uint32_t op = l->in[i].w[0] & 0xfff;
        if (!(op == 0x947 || op == 0x944 || op == 0x94d || op == 0x949 || op == 0x958 || op == 0x950)) continue;
        int n = successors(l, i, t, 72);
        if (n < 0) return 0;
        for (int k = 0; k < n; k++) if (t[k] != i + 1) l->leader[t[k]] = 1;
        if (i + 1 < l->n) l->leader[i + 1] = 1;
    }
    return 1;
}

// ---- the registers an instruction reads and writes (for liveness; reads over-approximated by three registers,
// the width a 128-bit operand reaches; the write is the destination's first register, none when predicated)

static void set_bit(uint64_t m[4], uint32_t r) { if (r < 255) m[r / 64] |= 1ull << (r % 64); }
static void set_run(uint64_t m[4], uint32_t r, uint32_t w) { for (uint32_t k = 0; k < w; k++) set_bit(m, r + k); }
static int get_bit(const uint64_t m[4], uint32_t r) { return r < 255 && (m[r / 64] >> (r % 64) & 1); }

static void insn_regs(const Insn *x, uint64_t use[4], uint64_t def[4]) {
    uint32_t op = x->w[0] & 0xfff, base = op & 0x1ff, form = op >> 9, g = bits(x, 12, 4);
    uint32_t f16 = bits(x, 16, 8), f24 = bits(x, 24, 8), f32 = bits(x, 32, 8), f64 = bits(x, 64, 8);
    int predicated = g != 7;
    if (g == 15) return;
    if (uniform_op(op)) {
        if (op == 0x3c2 || op == 0x3c4) set_run(use, f24, 1);      // R2UR, REDUX read a general register
        return;
    }
    if (base >= 0x140 && base < 0x160) {
        if (op == 0x949) set_run(use, f24, 1);
        return;
    }
    // A register at 32-39: ALU and conversion ops in form 1 (the others carry an immediate, a constant or a uniform
    // register there); texture ops always (Rb); stores and atomics (the data); loads and LDC never.
    int b_reg = base < 0x140 ? (form == 1) : base >= 0x160 && base < 0x180;
    if (base == 0x186 || base == 0x187 || base == 0x188 || base == 0x122 || base == 0x18e || base == 0x18c || base == 0x18d ||
        base == 0x1a8 || base == 0x1a9 || base == 0x189 || base == 0x127 || base >= 0x19b)
        b_reg = 1;
    int c_reg = base < 0x100 ? 1 : 0;                              // ALU: a register at 64 in every form
    int store = base == 0x186 || base == 0x187 || base == 0x188 || base == 0x122 || base == 0x18e || base == 0x127 ||
                base == 0x19b || base == 0x19c || base == 0x19e || base == 0x19f;
    int no_dst = store || base == 0x118 || base == 0x11a || base == 0x192 || base == 0x11d || base == 0x004 || base == 0x01c ||
                 (base == 0x00b) || (base == 0x00c) || (base == 0x034);   // FSETP, ISETP, HSETP2: predicates only
    // The destination's width: loads by their size (73-75: 6 = 128 bits, 5 = 64), texture ops by their mask (the
    // first two components to Rd, the rest to the second destination at 64-71 when given), IMAD.WIDE and a 64-bit
    // CS2R two registers; else one. Over-stating a write is unsafe (it frees a live register), so packed-half
    // texture results count as fewer. Reads are exact where the width is known and generous (four) where it is not
    // (texture coordinates); an over-stated read only keeps a register live longer.
    uint32_t w = 1, size = bits(x, 73, 3), by_size = size == 6 ? 4 : size == 5 ? 2 : 1;
    int load = base == 0x181 || base == 0x183 || base == 0x184 || base == 0x182;
    int tex = base >= 0x160 && base < 0x180, tex2 = 0;
    if (load) w = by_size;
    else if (tex) {
        uint32_t count = (uint32_t)__builtin_popcount(bits(x, 72, 4)), half = bits(x, 79, 1) && base != 0x16f && base != 0x170;
        if (half) count = (count + 1) / 2;
        if (f64 != RZ && count > 2) tex2 = (int)count - 2, count = 2;
        w = count;
    }
    else if (base == 0x025 || (base == 0x005 && bits(x, 80, 1))) w = 2;
    if (!no_dst && !predicated) {
        set_run(def, f16, w);
        if (tex2) set_run(def, f64, (uint32_t)tex2);
    }
    if (predicated && !no_dst) set_run(use, f16, w);               // a partial write keeps the old value live
    // Reads.
    int no_a = base == 0x002 || (base >= 0x100 && base <= 0x109) || base == 0x119 || base == 0x005 || base == 0x11a || base == 0x118 ||
               base == 0x126 || base == 0x121 || base == 0x01c || base == 0x003 || base == 0x006;
    uint32_t a_w = tex ? 4 : (load || store || base == 0x18c || base == 0x18d || base == 0x1a8 || base == 0x1a9 || base == 0x18e) ?
                   (bits(x, 72, 1) || base == 0x1a8 || base == 0x3a8 || base == 0x18e ? 2 : 1) : (base == 0x025 ? 1 : 1);
    if (!no_a) set_run(use, f24, a_w);
    if (b_reg) set_run(use, f32, tex ? 4 : store ? by_size : 1);
    if (c_reg || base == 0x122 || base == 0x189 || tex) set_run(use, f64, base == 0x025 ? 2 : tex ? 1 : 1);
    if (base == 0x122) set_run(use, f32, bits(x, 74, 2) + 1);     // AST data
    if (base == 0x121) set_run(use, f32, 1);                      // ALD vertex register
    if (base == 0x03c) {                                          // HMMA: A 4, B 2, C 2 read; D 2 written (f16 accumulators)
        set_run(use, f24, 4), set_run(use, f32, 2), set_run(use, f64, 2);
        if (!predicated) set_run(def, f16, 2);
        else set_run(use, f16, 2);
    }
    if (base == 0x03b && !predicated) set_run(def, f16, 1u << bits(x, 72, 2));   // LDSM: count registers
    if (base == 0x19f) {                                          // SURED: coordinates by the dimension, one data register
        static const uint8_t coords[8] = {1, 1, 2, 2, 3, 3, 1, 1};
        set_run(use, f24, coords[bits(x, 61, 3)]);
    }
    if (base == 0x19b) {                                          // SUST: coordinates by the dimension, data by the mask
        static const uint8_t coords[8] = {1, 1, 2, 2, 3, 3, 1, 1};
        set_run(use, f24, coords[bits(x, 61, 3)]);
        set_run(use, f32 & 0x3f, (uint32_t)__builtin_popcount(bits(x, 72, 4)));
    }
    if (base == 0x197) {                                          // SULD: coordinates by the dimension, a result by mask
        static const uint8_t coords[8] = {1, 1, 2, 2, 3, 3, 1, 1};
        uint32_t n = (uint32_t)__builtin_popcount(bits(x, 72, 4));
        set_run(use, f24, coords[bits(x, 61, 3)]);
        if (!predicated) set_run(def, f16, n ? n : 1);
        else set_run(use, f16, n ? n : 1);
    }
}

// ---- uniform registers: definitions, forwarding and the registers they need

// The uniform registers an instruction reads: list of (number, count) pairs; returns how many.
static int ur_reads(const Insn *x, uint32_t out[8][2]) {
    uint32_t op = x->w[0] & 0xfff, base = op & 0x1ff, form = op >> 9;
    int n = 0;
    if (bits(x, 12, 4) == 15) return 0;
    if (base >= 0x080 && base < 0x100) {
        int one_source = base == 0x0bd || base == 0x0be || base == 0x0bf || base == 0x082 || base == 0x086 || base == 0x0b9;
        if (!one_source) out[n][0] = bits(x, 24, 8), out[n++][1] = 1;
        if (form == 1 || (base == 0x082 && op == 0xc82)) out[n][0] = bits(x, 32, 8) & 63, out[n++][1] = 1;
        if (base != 0x08c && base != 0x087 && base != 0x082 && base != 0x086 && base != 0x0b9 && base != 0x0bd && base != 0x0be &&
            base != 0x0bf && base != 0x085 && base != 0x09a && (form == 1 || form == 4 || form == 5 || form == 6))
            out[n][0] = bits(x, 64, 8), out[n++][1] = 1;
        if (base == 0x0b9 && op == 0xabb) out[n][0] = bits(x, 24, 8), out[n++][1] = 1;
    } else if ((form == 6 || form == 7) && base < 0x120) {
        out[n][0] = bits(x, 32, 6), out[n++][1] = 1;
    } else if (op == 0xf60 || op == 0xf63 || op == 0xf66 || op == 0xf69) {
        out[n][0] = bits(x, 40, 6), out[n++][1] = 1;
    } else if (op == 0x981 && bits(x, 91, 1) && !bits(x, 38, 2)) {
        out[n][0] = bits(x, 32, 6), out[n++][1] = 2;
    } else if ((op == 0x986 || op == 0x9a8 || op == 0x98e || op == 0x987 || op == 0x988) && bits(x, 91, 1)) {
        out[n][0] = bits(x, 64, 6), out[n++][1] = op == 0x987 || op == 0x988 ? 1 : 2;
    } else if ((op == 0x983 || op == 0x984) && bits(x, 91, 1)) {
        out[n][0] = bits(x, 32, 6), out[n++][1] = 1;
    }
    return n;
}

// What an instruction defines for the local forwarding: the uniform register (63 none), its pair, and the
// constant it loads (kind 1 constant word, 2 immediate, 3 computed).
static uint32_t ur_write(const Insn *x, int *pair, uint8_t *kind, uint8_t *bank, uint32_t *value) {
    uint32_t op = x->w[0] & 0xfff, base = op & 0x1ff, dst = bits(x, 16, 8), g = bits(x, 12, 4);
    *pair = 0, *kind = 3, *bank = 0, *value = 0;
    if (g == 15) return 63;
    if (base >= 0x080 && base < 0x100 && base != 0x08c && base != 0x09c) {
        if (op == 0x882 && g == 7) *kind = 2, *value = x->w[1];
        if (op == 0xab9 && g == 7 && bits(x, 76, 2) == 0 && bits(x, 80, 11) == 0 && bits(x, 54, 5) < 18 &&
            (bits(x, 73, 3) == 4 || bits(x, 73, 3) == 5) && bits(x, 38, 16) % 4 == 0 && bits(x, 38, 16) / 4 < 0x4000) {
            *kind = 1, *bank = (uint8_t)bits(x, 54, 5), *value = bits(x, 38, 16), *pair = bits(x, 73, 3) == 5;
        } else if (op == 0xab9 || op == 0xabb) {
            *pair = bits(x, 73, 3) == 5;
        }
        if (base == 0x0a5) *pair = 1;
        return dst & 63;
    }
    if (op == 0x3c2 || op == 0x9c3 || op == 0x3c4) return dst & 63;
    return 63;
}

// Uniform register definitions: a register with one definition that is a direct 32-bit ULDC (or a pair, or a
// UMOV immediate) is forwarded as that constant; the others get general registers. Predicates: every field
// that can name one is counted as used (over-marking only shrinks the pool for the uniform predicates).
static void scan_uniform(L *l) {
    uint8_t count[64] = {0};
    for (uint32_t i = 0; i < l->n; i++) {
        const Insn *x = &l->in[i];
        uint32_t op = x->w[0] & 0xfff, base = op & 0x1ff, dst = bits(x, 16, 8), g = bits(x, 12, 4);
        if (g == 15) continue;
        int defines = 0, pair = 0, kind = 3;
        if (base >= 0x080 && base < 0x100 && base != 0x08c && base != 0x09c) {
            defines = 1;
            if (op == 0x882 && g == 7) kind = 2;
            if (op == 0x882 && g == 7) l->ur_def[dst & 63].value = x->w[1];
            if (op == 0xab9 && g == 7 && bits(x, 76, 2) == 0 && bits(x, 80, 11) == 0 && bits(x, 54, 5) < 18 &&
                (bits(x, 73, 3) == 4 || bits(x, 73, 3) == 5)) {
                kind = 1, pair = bits(x, 73, 3) == 5;
                l->ur_def[dst & 63].bank = (uint8_t)bits(x, 54, 5), l->ur_def[dst & 63].value = bits(x, 38, 16);
                if (pair) l->ur_def[(dst + 1) & 63].bank = (uint8_t)bits(x, 54, 5), l->ur_def[(dst + 1) & 63].value = bits(x, 38, 16) + 4;
            }
            if (base == 0x0a5) pair = 1;
        } else if (op == 0x3c2 || op == 0x9c3 || op == 0x3c4) {
            defines = 1;
        } else if (op == 0xf60 || op == 0xf63) {
            // reads only
        }
        if (!defines || dst == 63) continue;
        for (int k = 0; k <= pair; k++) {
            uint32_t n = (dst + k) & 63;
            if (count[n] < 255) count[n]++;
            l->ur_def[n].kind = (uint8_t)(count[n] == 1 ? kind : 3);
        }
    }
    for (uint32_t n = 0; n < 64; n++)
        if (count[n] > 1 || (l->ur_def[n].kind == 1 && (l->ur_def[n].value % 4 || l->ur_def[n].value / 4 >= 0x4000)))
            l->ur_def[n].kind = count[n] ? 3 : 0;
    // Local forwarding: within a basic block, a use after a ULDC / UMOV of the same uniform register takes the
    // constant. A uniform register whose every use forwards (globally or locally) needs no register; the others
    // get one each, above the program's registers, and the scratch block starts above them.
    {
        struct { uint8_t kind, bank; uint32_t value; } cur[64];
        memset(cur, 0, sizeof cur);
        for (uint32_t i = 0; i < l->n; i++) {
            const Insn *x = &l->in[i];
            if (l->leader && l->leader[i]) memset(cur, 0, sizeof cur);
            uint32_t reads[8][2];
            int nr = ur_reads(x, reads);
            for (int k = 0; k < nr; k++)
                for (uint32_t c = 0; c < reads[k][1]; c++) {
                    uint32_t n = reads[k][0] + c;
                    if (n >= 63) continue;
                    int forwards = l->ur_def[n].kind == 1 || l->ur_def[n].kind == 2 || cur[n].kind == 1 || cur[n].kind == 2;
                    if (!forwards) l->needs_reg[n] = 1;
                    if (!l->ur_first[n] && !l->ur_last[n]) l->ur_whole[n] = 1;   // read before any definition
                    if (!l->ur_first[n]) l->ur_first[n] = i + 1;
                    l->ur_last[n] = i + 1;
                }
            int pair;
            uint8_t kind, bank;
            uint32_t value;
            uint32_t w = ur_write(x, &pair, &kind, &bank, &value);
            for (int k = 0; w < 63 && k <= pair && w + k < 63; k++) {
                if (!l->ur_first[w + k]) l->ur_first[w + k] = i + 1;
                l->ur_last[w + k] = i + 1;
            }
            if (w < 63) {
                cur[w].kind = kind, cur[w].bank = bank, cur[w].value = value;
                if (pair && w + 1 < 63) cur[w + 1].kind = kind == 1 ? 1 : 3, cur[w + 1].bank = bank, cur[w + 1].value = value + 4;
                if (pair && w + 1 < 63 && !(w & 1)) l->ur_pair[w] = l->ur_pair[w + 1] = 1;
            }
        }
    }
    for (uint32_t i = 0; i < l->n; i++) {
        const Insn *x = &l->in[i];
        uint32_t op = x->w[0] & 0xfff, base = op & 0x1ff, g = bits(x, 12, 4);
        int uniform = base >= 0x080 && base < 0x100;
        if (g != 15 && (g & 7) != 7 && !uniform) l->used_preds |= 1u << (g & 7);
        if (uniform) continue;
        // The predicate fields of each op (NAK's encoders): a mask of the 3-bit fields at 68, 77, 81, 84, 87.
        unsigned fields = 0;
        switch (base) {
        case 0x00b: fields = 0x1c; break;                             // FSETP: 81, 84, 87
        case 0x00c: fields = 0x1d; break;                             // ISETP: 68, 81, 84, 87
        case 0x007: case 0x008: case 0x009: case 0x00a: case 0x017: case 0x033: fields = 0x10; break;   // 87
        case 0x010: fields = 0x1e; break;                             // IADD3: 77, 81, 84, 87
        case 0x011: case 0x024: case 0x025: fields = 0x14; break;     // LEA, IMAD: 81, 87
        case 0x012: fields = 0x14; break;                             // LOP3: 81, 87
        case 0x01c: fields = 0x1f; break;                             // PLOP3
        case 0x006: fields = 0x14; break;                             // VOTE: 81, 87
        case 0x034: fields = 0x1c; break;                             // HSETP2
        case 0x189: case 0x1a1: fields = 0x04; break;                 // SHFL, MATCH: 81
        case 0x147: case 0x14d: case 0x15b: fields = 0x10; break;     // BRA, EXIT, KILL: 87
        default: break;
        }
        static const unsigned at[5] = {68, 77, 81, 84, 87};
        for (unsigned k = 0; k < 5; k++) {
            if (!(fields >> k & 1)) continue;
            uint32_t v = bits(x, at[k], 3);
            if (v != 7) l->used_preds |= 1u << v;
        }
    }
    memset(l->up_map, 0xff, sizeof l->up_map);
    // The uniform predicates are swapped through a register when no real predicate is free, or when a first attempt
    // ran out of free ones (force_swap: three compute kernels of the Great Deku Tree scene name four uniform
    // predicates and leave three free).
    l->up_swap = l->force_swap || (l->used_preds & 0x7f) == 0x7f;
    // The registers the lowering needs for the program's length: one per uniform register no forwarding covers,
    // one for the uniform predicates when they are swapped through a register. The scratch block starts above them.
    uint32_t nur = 0;
    for (uint32_t n = 0; n < 63; n++)
        if (l->needs_reg[n]) nur++;                               // defined or not (a read before any definition)
    if (l->up_swap)                                               // the uniform predicates' register (up_swap)
        for (uint32_t i = 0; i < l->n; i++) if (up_fields(&l->in[i])) { nur++; break; }
    l->scratch_base = l->ur_next + nur > 255 ? 255 : l->ur_next + nur;
    for (uint32_t r = l->ur_next; r < l->scratch_base; r++) l->reserved[r / 64] |= 1ull << (r % 64);
    l->scanned = 1;
}

// The uniform predicates an instruction names (bit n = UPn), for the swap around it.
static uint32_t up_fields(const Insn *x) {
    uint32_t op = x->w[0] & 0xfff, base = op & 0x1ff, m = 0, g = bits(x, 12, 4);
    unsigned fields = 0;
    if (base >= 0x080 && base < 0x100) {
        if ((g & 7) != 7) m |= 1u << (g & 7);
        switch (base) {
        case 0x090: case 0x091: case 0x0a4: case 0x0a5: fields = 0x1e; break;
        case 0x08c: fields = 0x1d; break;
        case 0x087: fields = 0x10; break;
        case 0x086: fields = 0x04; break;
        case 0x09c: fields = 0x1f; break;
        default: break;
        }
    } else if (op == 0x81c && bits(x, 67, 1)) {
        fields = 0x01;
    }
    static const unsigned at[5] = {68, 77, 81, 84, 87};
    for (unsigned k = 0; k < 5; k++) {
        if (!(fields >> k & 1)) continue;
        uint32_t v = bits(x, at[k], 3);
        if (v != 7) m |= 1u << v;
    }
    return m;
}

// ---- liveness (only for programs whose registers leave no room for the static pools)

static int compute_liveness(L *l) {
    uint32_t n = l->n;
    uint64_t (*use)[4] = calloc(n, sizeof *use), (*def)[4] = calloc(n, sizeof *def);
    uint64_t (*in)[4] = calloc(n, sizeof *in);
    l->live_out = calloc(n, sizeof *l->live_out);
    if (!use || !def || !in || !l->live_out) { free(use), free(def), free(in); return fail(l, "out of memory"), 0; }
    for (uint32_t i = 0; i < n; i++) insn_regs(&l->in[i], use[i], def[i]);
    // Backward dataflow to a fixed point: out(i) = union of in(s) over successors; in(i) = use | (out & ~def).
    uint32_t succ[72];
    int changed = 1, rounds = 0;
    while (changed && rounds++ < 200) {
        changed = 0;
        for (uint32_t i = n; i-- > 0;) {
            uint64_t out[4] = {0, 0, 0, 0};
            int ns = successors(l, i, succ, 72);
            if (ns < 0) { free(use), free(def), free(in); return 0; }
            for (int k = 0; k < ns; k++)
                for (int w = 0; w < 4; w++) out[w] |= in[succ[k]][w];
            uint64_t ni[4];
            for (int w = 0; w < 4; w++) ni[w] = use[i][w] | (out[w] & ~def[i][w]);
            if (memcmp(ni, in[i], sizeof ni)) changed = 1, memcpy(in[i], ni, sizeof ni);
            memcpy(l->live_out[i], out, sizeof out);
        }
    }
    // The scratch block of an instruction must also leave its own operands and destination alone: fold them in.
    for (uint32_t i = 0; i < n; i++) {
        for (int w = 0; w < 4; w++) l->live_out[i][w] |= use[i][w];
        for (uint32_t r = 0; r < 255; r++) if (get_bit(def[i], r)) set_run(l->live_out[i], r, 4);
    }
    free(use), free(def), free(in);
    return rounds < 200;
}

// The longest run of registers free at instruction i, as the scratch block.
static void scratch_at(L *l, uint32_t i) {
#ifdef SM86_TOOL
    if (getenv("SM86_LIVE")) {
        int live = 0;
        for (uint32_t r = 0; r < 255; r++) live += get_bit(l->live_out[i], r);
        fprintf(stderr, "live %u: %d registers live after instruction %u (op %03x):", i, live, i, l->in[i].w[0] & 0xfff);
        if (i < 2) for (uint32_t r = 0; r < 255; r++) if (get_bit(l->live_out[i], r)) fprintf(stderr, " %u", r);
        fprintf(stderr, "\n");
    }
#endif
    uint32_t best = 0, best_len = 0, run = 0;
    for (uint32_t r = 0; r < 255; r++) {
        int busy = get_bit(l->live_out[i], r) || get_bit(l->reserved, r);
        if (busy) run = 0;
        else if (++run > best_len) best_len = run, best = r + 1 - run;
        if (best_len >= NSCRATCH) break;
    }
    l->scratch_base = best, l->scratch_len = best_len > NSCRATCH ? NSCRATCH : best_len, l->nscratch = 0;
}

// ---- operands

static Src src0(const Insn *x, int mods) {
    Src s = reg_src(bits(x, 24, 8));
    if (mods) s.neg = bits(x, 72, 1), s.abs_ = bits(x, 73, 1);
    return s;
}

// The second (which 1) or third (which 2) source by the form in bits 9-11 (NAK encode_alu_base).
static Src srcn(L *l, const Insn *x, int which, int mods) {
    Src s = {0};
    uint32_t form = bits(x, 9, 3);
    int wide = which == 1 ? (form == 4 || form == 5 || form == 6) : (form == 2 || form == 3 || form == 7);
    if (form == 0) { fail(l, "ALU source form 0"); return reg_src(RZ); }
    if (wide) {
        if (form == 4 || form == 2) s.kind = 1, s.imm = x->w[1];
        else if (form == 5 || form == 3) {
            s.kind = 2, s.cb_off = bits(x, 38, 16), s.cb_bank = bits(x, 54, 5);
            if (mods) s.abs_ = bits(x, 62, 1), s.neg = bits(x, 63, 1);
        } else {
            s = ur_src(l, bits(x, 32, 6));
            if (mods) s.abs_ = bits(x, 62, 1), s.neg = bits(x, 63, 1);
        }
    } else {
        int upper = which == 1 ? (form == 2 || form == 3 || form == 7) : 1;
        s.reg = bits(x, upper ? 64 : 32, 8);
        if (mods) s.abs_ = bits(x, upper ? 74 : 62, 1), s.neg = bits(x, upper ? 75 : 63, 1);
    }
    return s;
}

// Maxwell operand fields. A constant-buffer word: dword offset at 20-33, bank at 34-38.
static int cbuf_ok(const Src *s) { return s->cb_bank < 18 && s->cb_off % 4 == 0 && s->cb_off / 4 < 0x4000; }
static uint64_t cbuf_bits(const Src *s) { return ((uint64_t)(s->cb_off / 4) << 20) | ((uint64_t)s->cb_bank << 34); }
// A 20-bit immediate: 19 bits at 20-38 and the sign at 56; floats carry their high 20 bits, integers sign-extend.
static int fimm20_ok(uint32_t v) { return (v & 0xfff) == 0; }
static uint64_t fimm20(uint32_t v) { return ((uint64_t)((v >> 12) & 0x7ffff) << 20) | ((uint64_t)(v >> 31) << 56); }
static int iimm20_ok(uint32_t v) { int32_t s = (int32_t)v; return s >= -(1 << 19) && s < (1 << 19); }
static uint64_t iimm20(uint32_t v) { return ((uint64_t)(v & 0x7ffff) << 20) | ((uint64_t)((v >> 19) & 1) << 56); }

#define MOV_R   0x5c98000000000000ull
#define MOV_C   0x4c98000000000000ull
#define MOV32I  0x0100000000000000ull
#define LOP32I  0x0400000000000000ull
#define QUAD    ((uint64_t)0xf << 39)

static void mov_imm(L *l, uint32_t rd, uint32_t v) { emit(l, MOV32I | l->pred | rd | ((uint64_t)0xf << 12) | ((uint64_t)v << 20)); }
static void mov_reg(L *l, uint32_t rd, uint32_t ra) { emit(l, MOV_R | l->pred | rd | QUAD | ((uint64_t)ra << 20)); }
static void mov_cbuf(L *l, uint32_t rd, const Src *s) { emit(l, MOV_C | l->pred | rd | QUAD | cbuf_bits(s)); }
// rd = ra op imm32 (LOP32I: 0 and, 1 or, 2 xor)
static void lop_imm(L *l, uint32_t rd, uint32_t ra, uint32_t v, int lop) {
    emit(l, LOP32I | l->pred | rd | ((uint64_t)ra << 8) | ((uint64_t)v << 20) | ((uint64_t)lop << 53));
}

// Copies a source into register r.
static void to_reg_into(L *l, Src s, uint32_t r) {
    if (s.kind == 1) mov_imm(l, r, s.imm);
    else if (s.kind == 2) { if (!cbuf_ok(&s)) { fail(l, "constant buffer operand"); return; } mov_cbuf(l, r, &s); }
    else mov_reg(l, r, s.reg);
}

// Brings a source into a register (scratch k for an immediate or constant), applying the float modifiers the
// target form cannot carry (want_neg / want_abs say which it can).
static uint32_t to_reg(L *l, Src s, int k, int keep_neg, int keep_abs) {
    uint32_t r = s.reg;
    if (s.kind == 1) {
        if (s.imm == 0 && !s.neg && !s.abs_) return RZ;
        r = S(l, k), mov_imm(l, r, s.imm);
    } else if (s.kind == 2) {
        if (!cbuf_ok(&s)) return fail(l, "constant buffer operand"), RZ;
        r = S(l, k), mov_cbuf(l, r, &s);
    }
    int neg = s.neg && !keep_neg, abs_ = s.abs_ && !keep_abs;
    if (abs_ || neg) {
        uint32_t t = S(l, k);
        if (abs_) lop_imm(l, t, r, 0x7fffffff, 0), r = t;
        if (neg) lop_imm(l, t, r, 0x80000000, 2), r = t;
    }
    return r;
}

// The b slot of a Maxwell R / C / I20 / I32 instruction family. Returns the opcode base chosen with the operand
// bits set: the register at 20-27, the constant at 20-38, or an immediate; an immediate that fits no form is
// materialised in scratch k. is_float selects the float immediate rule; op32 is 0 when the family has no 32-bit
// immediate form (then the scratch register is used).
static uint64_t b_slot(L *l, Src s, uint64_t op_r, uint64_t op_c, uint64_t op_i, uint64_t op32, int is_float, int k,
                       int *is_32) {
    *is_32 = 0;
    if (s.kind == 2 && cbuf_ok(&s)) return op_c | cbuf_bits(&s);
    if (s.kind == 1) {
        if (is_float ? fimm20_ok(s.imm) : iimm20_ok(s.imm)) return op_i | (is_float ? fimm20(s.imm) : iimm20(s.imm));
        if (op32) { *is_32 = 1; return op32 | ((uint64_t)s.imm << 20); }
    }
    uint32_t r = to_reg(l, s, k, 1, 1);
    return op_r | ((uint64_t)r << 20);
}

static uint64_t base(L *l, uint64_t op, uint32_t rd, uint32_t ra) { return op | l->pred | rd | ((uint64_t)ra << 8); }
// Maxwell HFMA2 R (Ryujinx InstHfma2R): Rd = a(swizzle) * b + c on half pairs; a swizzle 2 = H0_H0, 3 = H1_H1.
#define HFMA2_R(rd, ra, aswz, rb, rc) (0x5d00000000000000ull | (rd) | ((uint64_t)(ra) << 8) | ((uint64_t)(rb) << 20) | \
                                       ((uint64_t)(rc) << 39) | ((uint64_t)(aswz) << 47))
static void h2_relu(L *l, uint32_t rd);

// ---- the per-opcode lowerings

static void lower_mov(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8);
    if (bits(x, 72, 4) != 15) return (void)fail(l, "MOV quad mask");
    if (op == 0x802) return mov_imm(l, rd, x->w[1]);
    Src b = srcn(l, x, 1, 0);
    if (b.kind == 2) {
        if (!cbuf_ok(&b)) return (void)fail(l, "MOV constant buffer address");
        return mov_cbuf(l, rd, &b);
    }
    mov_reg(l, rd, b.reg);
}

// FMUL: dnz 76, sat 77, rnd 78-79, ftz 80, scale 84-86 (4 = none).
static void lower_fmul(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), scale = bits(x, 84, 3);
    static const uint32_t scale_map[8] = {0, 3, 2, 1, 0, 6, 5, 4};
    if (bits(x, 76, 1) || scale == 0 || bits(x, 81, 3) || bits(x, 87, 4)) return (void)fail(l, "FMUL mode");
    Src a = src0(x, 1), b = srcn(l, x, 1, 1);
    int neg = a.neg ^ b.neg;
    uint32_t ra = to_reg(l, a, 0, 1, 0);
    b.neg = 0;
    int is32;
    uint64_t w = b_slot(l, b, 0x5c68000000000000ull, 0x4c68000000000000ull, 0x3868000000000000ull,
                        0x1e00000000000000ull, 1, 1, &is32);
    w |= l->pred | rd | ((uint64_t)ra << 8);
    if (is32) {
        if (neg || scale != 4) {               // FMUL32I has no negation or scale: fold them through a register
            Src t = b; t.neg = 0;
            uint32_t rb = to_reg(l, t, 1, 0, 0);
            w = base(l, 0x5c68000000000000ull, rd, ra) | ((uint64_t)rb << 20);
            is32 = 0;
        } else {
            w |= ((uint64_t)bits(x, 80, 1) << 53) | ((uint64_t)bits(x, 77, 1) << 55);
            return emit(l, w);
        }
    }
    w |= ((uint64_t)bits(x, 78, 2) << 39) | ((uint64_t)scale_map[scale] << 41) | ((uint64_t)bits(x, 80, 1) << 44) |
         ((uint64_t)neg << 48) | ((uint64_t)bits(x, 77, 1) << 50);
    emit(l, w);
}

// FADD: the second source is in the b slot (form 1) or the c slot (forms 2, 3, 7); sat 77, rnd 78-79, ftz 80.
static void lower_fadd(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), form = bits(x, 9, 3);
    if (bits(x, 81, 10)) return (void)fail(l, "FADD mode");
    Src a = src0(x, 1), b = srcn(l, x, form == 1 ? 1 : 2, 1);
    uint32_t ra = a.reg;
    int is32;
    uint64_t w = b_slot(l, b, 0x5c58000000000000ull, 0x4c58000000000000ull, 0x3858000000000000ull,
                        0x0800000000000000ull, 1, 1, &is32);
    w |= l->pred | rd | ((uint64_t)ra << 8);
    if (is32 && (bits(x, 77, 1) || bits(x, 78, 2))) {
        uint32_t rb = to_reg(l, b, 1, 1, 1);
        w = base(l, 0x5c58000000000000ull, rd, ra) | ((uint64_t)rb << 20);
        is32 = 0;
    }
    if (is32) {
        w |= ((uint64_t)b.neg << 53) | ((uint64_t)a.abs_ << 54) | ((uint64_t)bits(x, 80, 1) << 55) |
             ((uint64_t)a.neg << 56) | ((uint64_t)b.abs_ << 57);
        return emit(l, w);
    }
    w |= ((uint64_t)bits(x, 78, 2) << 39) | ((uint64_t)bits(x, 80, 1) << 44) | ((uint64_t)b.neg << 45) |
         ((uint64_t)a.abs_ << 46) | ((uint64_t)a.neg << 48) | ((uint64_t)b.abs_ << 49) | ((uint64_t)bits(x, 77, 1) << 50);
    emit(l, w);
}

// FFMA: a * b + c; dnz 76, sat 77, rnd 78-79, ftz 80. Maxwell: RRR, R I20 R, R C R and R R C.
static void lower_ffma(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8);
    if (bits(x, 76, 1) || bits(x, 81, 10)) return (void)fail(l, "FFMA mode");
    Src a = src0(x, 1), b = srcn(l, x, 1, 1), c = srcn(l, x, 2, 1);
    int neg_a = a.neg ^ b.neg, neg_c = c.neg;
    uint32_t ra = to_reg(l, a, 0, 1, 0);
    uint64_t mods = ((uint64_t)neg_a << 48) | ((uint64_t)neg_c << 49) | ((uint64_t)bits(x, 77, 1) << 50) |
                    ((uint64_t)bits(x, 78, 2) << 51) | ((uint64_t)bits(x, 80, 1) << 53);
    b.neg = 0, c.neg = 0;
    if (b.abs_) b.reg = to_reg(l, b, 1, 1, 0), b.kind = 0, b.abs_ = 0;
    if (c.abs_) c.reg = to_reg(l, c, 2, 1, 0), c.kind = 0, c.abs_ = 0;
    uint64_t w;
    if (c.kind == 2 && cbuf_ok(&c)) {                   // R R C: the register b at 39-46
        uint32_t rb = to_reg(l, b, 1, 1, 1);
        w = base(l, 0x5180000000000000ull, rd, ra) | cbuf_bits(&c) | ((uint64_t)rb << 39);
    } else {
        uint32_t rc = to_reg(l, c, 2, 1, 1);
        int is32;
        w = b_slot(l, b, 0x5980000000000000ull, 0x4980000000000000ull, 0x3280000000000000ull, 0, 1, 1, &is32);
        w |= l->pred | rd | ((uint64_t)ra << 8) | ((uint64_t)rc << 39);
    }
    emit(l, w | mods);
}

// FMNMX: min when the predicate at 87-89 (negated by 90) holds; ftz 80; .NAN (81) is not lowered.
static void lower_fmnmx(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8);
    if (bits(x, 74, 6) || bits(x, 81, 1) || bits(x, 82, 5)) return (void)fail(l, "FMNMX mode");
    Src a = src0(x, 1), b = srcn(l, x, 1, 1);
    int is32;
    uint64_t w = b_slot(l, b, 0x5c60000000000000ull, 0x4c60000000000000ull, 0x3860000000000000ull, 0, 1, 1, &is32);
    w |= l->pred | rd | ((uint64_t)a.reg << 8) | ((uint64_t)bits(x, 87, 3) << 39) | ((uint64_t)bits(x, 90, 1) << 42) |
         ((uint64_t)bits(x, 80, 1) << 44) | ((uint64_t)b.neg << 45) | ((uint64_t)a.abs_ << 46) | ((uint64_t)a.neg << 48) |
         ((uint64_t)b.abs_ << 49);
    emit(l, w);
}

// FSETP: set op 74-75, compare 76-79, ftz 80, outputs 81-83 and 84-86, accumulator predicate 87-89 (negated 90).
static void lower_fsetp(L *l, const Insn *x) {
    Src a = src0(x, 1), b = srcn(l, x, 1, 1);
    if (bits(x, 74, 2) == 3) return (void)fail(l, "FSETP combination");
    int is32;
    uint64_t w = b_slot(l, b, 0x5bb0000000000000ull, 0x4bb0000000000000ull, 0x36b0000000000000ull, 0, 1, 1, &is32);
    w |= l->pred | bits(x, 84, 3) | ((uint64_t)bits(x, 81, 3) << 3) | ((uint64_t)b.neg << 6) | ((uint64_t)a.abs_ << 7) |
         ((uint64_t)a.reg << 8) | ((uint64_t)bits(x, 87, 3) << 39) | ((uint64_t)bits(x, 90, 1) << 42) |
         ((uint64_t)a.neg << 43) | ((uint64_t)b.abs_ << 44) | ((uint64_t)bits(x, 74, 2) << 45) |
         ((uint64_t)bits(x, 80, 1) << 47) | ((uint64_t)bits(x, 76, 4) << 48);
    emit(l, w);
}

// FSET.BF: 1.0f or 0.0f (or ~0 / 0 without .BF) from the comparison; the fields are FSETP's.
static void lower_fset(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8);
    Src a = src0(x, 1), b = srcn(l, x, 1, 1);
    if (bits(x, 74, 2) == 3 || bits(x, 81, 6)) return (void)fail(l, "FSET mode");
    int is32;
    uint64_t w = b_slot(l, b, 0x5800000000000000ull, 0x4800000000000000ull, 0x3000000000000000ull, 0, 1, 1, &is32);
    w |= l->pred | rd | ((uint64_t)a.reg << 8) | ((uint64_t)bits(x, 87, 3) << 39) | ((uint64_t)bits(x, 90, 1) << 42) |
         ((uint64_t)a.neg << 43) | ((uint64_t)b.abs_ << 44) | ((uint64_t)bits(x, 74, 2) << 45) |
         ((uint64_t)bits(x, 76, 4) << 48) | ((uint64_t)1 << 52) | ((uint64_t)b.neg << 53) | ((uint64_t)a.abs_ << 54) |
         ((uint64_t)bits(x, 80, 1) << 55);
    emit(l, w);
}

// ISETP: .EX 72, signed 73, set op 74-75, compare 76-78, outputs 81-83 / 84-86, accumulator 87-89 (90).
static void lower_isetp(L *l, const Insn *x) {
    Src a = src0(x, 0), b = srcn(l, x, 1, 0);
    if (bits(x, 74, 2) == 3) return (void)fail(l, "ISETP combination");
    if (bits(x, 72, 1)) {
        // .EX: the high half of a 64-bit compare, the low half's result in the predicate at 68-70 (negated by
        // 71): EQ = high equal AND low; NE = high different OR low; an ordering = high strictly so OR (high equal
        // AND low). Computed in a predicate the instruction does not name, saved around it in scratch.
        uint32_t cmp = bits(x, 76, 3), pd = bits(x, 81, 3), pd2 = bits(x, 84, 3), low = bits(x, 68, 4), acc = bits(x, 87, 4);
        uint32_t combine = bits(x, 74, 2), guard = bits(x, 12, 3), sgn = bits(x, 73, 1);
        if (cmp == 0 || cmp == 7) return (void)fail(l, "ISETP.EX compare");
        uint32_t t = 0;
        while (t == pd || t == pd2 || t == (low & 7) || t == (acc & 7) || t == guard) t++;
        if (t > 6) return (void)fail(l, "ISETP.EX temporary");
        uint32_t ra = a.reg, rb = to_reg(l, b, 1, 0, 0), save = S(l, 2);
        emit(l, base(l, 0x38e8000000000000ull, save, RZ) | ((uint64_t)(1u << t) << 20));       // P2R save, mask t
        uint64_t common = 0x5b60000000000000ull | l->pred | 7 | ((uint64_t)t << 3) | ((uint64_t)ra << 8) | ((uint64_t)rb << 20) |
                          ((uint64_t)sgn << 48);
        uint64_t with_low = ((uint64_t)(low & 7) << 39) | ((uint64_t)(low >> 3) << 42);
        if (cmp == 2) emit(l, common | with_low | ((uint64_t)0 << 45) | ((uint64_t)2 << 49));
        else if (cmp == 5) emit(l, common | with_low | ((uint64_t)1 << 45) | ((uint64_t)5 << 49));
        else {
            emit(l, common | with_low | ((uint64_t)0 << 45) | ((uint64_t)2 << 49));
            emit(l, common | ((uint64_t)t << 39) | ((uint64_t)1 << 45) | ((uint64_t)(cmp <= 3 ? 1 : 4) << 49));
        }
        emit(l, 0x5090000000000000ull | l->pred | pd2 | ((uint64_t)pd << 3) | ((uint64_t)t << 12) | ((uint64_t)7 << 29) |
                    ((uint64_t)(acc & 7) << 39) | ((uint64_t)(acc >> 3) << 42) | ((uint64_t)combine << 45));
        emit(l, 0x38f0000000000000ull | l->pred | ((uint64_t)save << 8) | ((uint64_t)(1u << t) << 20));   // R2P restore
        return;
    }
    if (bits(x, 68, 4) != 7) return (void)fail(l, "ISETP low predicate without .EX");
    int is32;
    uint64_t w = b_slot(l, b, 0x5b60000000000000ull, 0x4b60000000000000ull, 0x3660000000000000ull, 0, 0, 1, &is32);
    w |= l->pred | bits(x, 84, 3) | ((uint64_t)bits(x, 81, 3) << 3) | ((uint64_t)a.reg << 8) |
         ((uint64_t)bits(x, 87, 3) << 39) | ((uint64_t)bits(x, 90, 1) << 42) | ((uint64_t)bits(x, 74, 2) << 45) |
         ((uint64_t)bits(x, 73, 1) << 48) | ((uint64_t)bits(x, 76, 3) << 49);
    emit(l, w);
}

// SEL (0x007) and FSEL (0x008): a when the predicate at 87-89 (90) holds, else b.
static void lower_sel(L *l, const Insn *x, int floating) {
    uint32_t rd = bits(x, 16, 8);
    Src a = src0(x, floating), b = srcn(l, x, 1, floating);
    uint32_t ra = to_reg(l, a, 0, 0, 0);
    int is32;
    uint64_t w;
    if (b.neg || b.abs_) w = 0x5ca0000000000000ull | ((uint64_t)to_reg(l, b, 1, 0, 0) << 20);
    else w = b_slot(l, b, 0x5ca0000000000000ull, 0x4ca0000000000000ull, 0x38a0000000000000ull, 0, floating, 1, &is32);
    w |= l->pred | rd | ((uint64_t)ra << 8) | ((uint64_t)bits(x, 87, 3) << 39) | ((uint64_t)bits(x, 90, 1) << 42);
    emit(l, w);
}

// IMNMX: signed 73; min when the predicate at 87-89 (90) holds.
static void lower_imnmx(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8);
    Src a = src0(x, 0), b = srcn(l, x, 1, 0);
    if (bits(x, 72, 1) || bits(x, 74, 6) || bits(x, 80, 7)) return (void)fail(l, "IMNMX mode");
    int is32;
    uint64_t w = b_slot(l, b, 0x5c20000000000000ull, 0x4c20000000000000ull, 0x3820000000000000ull, 0, 0, 1, &is32);
    w |= l->pred | rd | ((uint64_t)a.reg << 8) | ((uint64_t)bits(x, 87, 3) << 39) | ((uint64_t)bits(x, 90, 1) << 42) |
         ((uint64_t)bits(x, 73, 1) << 48);
    emit(l, w);
}

// LOP3.LUT: table 72-79, .PAND 80, predicate output 81-83 (result != 0), predicate input 87-90.
static void lower_lop3(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), lut = bits(x, 72, 8), pout = bits(x, 81, 3);
    if (bits(x, 80, 1) || bits(x, 87, 4) != 15) return (void)fail(l, "LOP3 predicate mode");
    Src a = src0(x, 0), b = srcn(l, x, 1, 0), c = srcn(l, x, 2, 0);
    uint32_t rb = to_reg(l, b, 1, 0, 0), rc = to_reg(l, c, 2, 0, 0);
    emit(l, base(l, 0x5be0000000000000ull, rd, a.reg) | ((uint64_t)rb << 20) | ((uint64_t)lut << 28) |
                ((uint64_t)(pout == 7 ? 0 : 3) << 36) | ((uint64_t)rc << 39) | ((uint64_t)pout << 48));
}

// The carry in of the .X forms: t += 1 when predicate p (inverted by inv) holds. Nothing for !PT.
static void add_carry_in(L *l, uint32_t t, uint32_t p, int inv) {
    if (p == 7 && inv) return;
    uint32_t one = S(l, 6);
    mov_imm(l, one, 1);
    if (p != 7) emit(l, base(l, 0x5ca0000000000000ull, one, one) | ((uint64_t)RZ << 20) | ((uint64_t)p << 39) | ((uint64_t)inv << 42));
    emit(l, base(l, 0x5c10000000000000ull, t, t) | ((uint64_t)one << 20));
}

// IADD3: a + b + c; negations 72 (a), 63 (b, register and constant forms), 75 (c); .X 74; carry outs 81-83 and
// 84-86; carry ins 87-89 (90) and 77-79 (80).
static void lower_iadd3(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8);
    Src a = src0(x, 0), b = srcn(l, x, 1, 0), c = srcn(l, x, 2, 0);
    int neg_a = bits(x, 72, 1), neg_b = (op == 0x810) ? 0 : bits(x, 63, 1), neg_c = bits(x, 75, 1);
    uint32_t cout = bits(x, 81, 3), cout2 = bits(x, 84, 3), ext = bits(x, 74, 1);
    if (bits(x, 73, 1) || bits(x, 76, 1)) return (void)fail(l, "IADD3 mode");
    if (ext) {
        // IADD3.X: a + b + c + carry predicates (87-89 negated by 90, 77-79 negated by 80); no carry out here.
        if (cout != 7 || cout2 != 7 || neg_a || neg_b || neg_c) return (void)fail(l, "IADD3.X carry out");
        uint32_t rb = to_reg(l, b, 1, 0, 0), rc = to_reg(l, c, 2, 0, 0);
        uint32_t t = S(l, 3);
        emit(l, base(l, 0x5cc0000000000000ull, t, a.reg) | ((uint64_t)rb << 20) | ((uint64_t)rc << 39));
        uint32_t one = S(l, 4);
        for (int k = 0; k < 2; k++) {
            uint32_t p = bits(x, k ? 77 : 87, 3), inv = bits(x, k ? 80 : 90, 1);
            if (p == 7 && inv) continue;                              // !PT: no carry
            mov_imm(l, one, 1);
            if (p != 7) emit(l, base(l, 0x5ca0000000000000ull, one, one) | ((uint64_t)RZ << 20) | ((uint64_t)p << 39) | ((uint64_t)inv << 42));
            emit(l, base(l, 0x5c10000000000000ull, t, t) | ((uint64_t)one << 20));
        }
        mov_reg(l, rd, t);
        return;
    }
    if (bits(x, 77, 4) != 15 || bits(x, 87, 4) != 15) return (void)fail(l, "IADD3 carry in without .X");
    if (cout == 7 && cout2 == 7) {
        if (c.kind == 0 && c.reg == RZ) {
            // Two sources: IADD R / C / I20 / I32 with the negations (AvgMode: 1 NegB, 2 NegA).
            if (neg_a && neg_b) return (void)fail(l, "IADD3 both negated");
            int is32;
            uint64_t w = b_slot(l, b, 0x5c10000000000000ull, 0x4c10000000000000ull, 0x3810000000000000ull,
                                0x1c00000000000000ull, 0, 1, &is32);
            uint32_t avg = neg_b ? 1 : neg_a ? 2 : 0;
            w |= l->pred | rd | ((uint64_t)a.reg << 8) | ((uint64_t)avg << (is32 ? 55 : 48));
            return emit(l, w);
        }
        uint32_t rb = to_reg(l, b, 1, 0, 0), rc = to_reg(l, c, 2, 0, 0);
        emit(l, base(l, 0x5cc0000000000000ull, rd, a.reg) | ((uint64_t)rb << 20) | ((uint64_t)rc << 39) |
                    ((uint64_t)neg_c << 49) | ((uint64_t)neg_b << 50) | ((uint64_t)neg_a << 51));
        return;
    }
    // A carry out: the unsigned sum a + b (+ c) overflowed. Rd may be one of the sources, so sum in scratch:
    // carry1 = sum < b, and with a third source carry2 = (sum + c) < c; the carry out is carry1 | carry2
    // (at most one of them can be set when both carries go to one predicate... no: both may be set only if the
    // sum wrapped twice, impossible for three 32-bit values summing below 3 * 2^32, as Pu and Pv count them).
    if (neg_a || neg_b || neg_c) return (void)fail(l, "IADD3 carry out with negation");
    uint32_t rb = to_reg(l, b, 1, 0, 0), rc = to_reg(l, c, 2, 0, 0);
    uint32_t t = S(l, 3);
    emit(l, base(l, 0x5c10000000000000ull, t, a.reg) | ((uint64_t)rb << 20));
    if (rc == RZ) {
        if (cout2 != 7) return (void)fail(l, "IADD3 second carry out of two sources");
        // Pu = t < b (unsigned): ISETP.LT.U32 Pu, PT, t, rb, PT
        emit(l, 0x5b60000000000000ull | l->pred | 7 | ((uint64_t)cout << 3) | ((uint64_t)t << 8) | ((uint64_t)rb << 20) |
                    ((uint64_t)7 << 39) | ((uint64_t)1 << 49));
        mov_reg(l, rd, t);
        return;
    }
    // Three sources: Pu (81-83) holds when the sum wrapped at least once, Pv (84-86) when twice. The wraps are
    // counted in u from the two unsigned comparisons (sum < addend), through the first output predicate.
    uint32_t pu = cout != 7 ? cout : cout2;
    uint32_t u = S(l, 4), one = S(l, 5);
    emit(l, 0x5b60000000000000ull | l->pred | 7 | ((uint64_t)pu << 3) | ((uint64_t)t << 8) | ((uint64_t)rb << 20) |
                ((uint64_t)7 << 39) | ((uint64_t)1 << 49));
    mov_imm(l, one, 1);
    emit(l, base(l, 0x5ca0000000000000ull, u, one) | ((uint64_t)RZ << 20) | ((uint64_t)pu << 39));
    emit(l, base(l, 0x5c10000000000000ull, t, t) | ((uint64_t)rc << 20));
    emit(l, 0x5b60000000000000ull | l->pred | 7 | ((uint64_t)pu << 3) | ((uint64_t)t << 8) | ((uint64_t)rc << 20) |
                ((uint64_t)7 << 39) | ((uint64_t)1 << 49));
    emit(l, base(l, 0x5ca0000000000000ull, one, one) | ((uint64_t)RZ << 20) | ((uint64_t)pu << 39));
    emit(l, base(l, 0x5c10000000000000ull, u, u) | ((uint64_t)one << 20));
    if (cout != 7)      // Pu = u != 0
        emit(l, 0x5b60000000000000ull | l->pred | 7 | ((uint64_t)cout << 3) | ((uint64_t)u << 8) | ((uint64_t)RZ << 20) |
                    ((uint64_t)7 << 39) | ((uint64_t)5 << 49));
    if (cout2 != 7)     // Pv = u == 2
        emit(l, 0x3660000000000000ull | l->pred | 7 | ((uint64_t)cout2 << 3) | ((uint64_t)u << 8) | ((uint64_t)2 << 20) |
                    ((uint64_t)7 << 39) | ((uint64_t)2 << 49));
    mov_reg(l, rd, t);
}

// LEA: Rd = (Ra << shift) + Rb (shift 75-79); 72 negates the shifted a; .HI 80; carry out 81-83; .X 74 with the
// carry in at 87-90; 63 negates Rb in the register and constant forms. Maxwell LEA: a 8, b 20 (register, constant
// or 20-bit immediate), shift 39-43, NegA 45, X 46, carry out predicate 48-50; the translator reads neither X nor the
// carry predicate (Ryujinx InstEmitIntegerArithmetic.cs EmitLea / EmitLeaHi), so carries go through comparisons.
static void lower_lea(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8), shift = bits(x, 75, 5), cout = bits(x, 81, 3);
    Src a = src0(x, 0), b = srcn(l, x, 1, 0);
    int neg_b = op == 0x811 ? 0 : bits(x, 63, 1), neg_a = bits(x, 72, 1);
    int ext = bits(x, 74, 1);
    uint32_t cin = bits(x, 87, 3), cin_inv = bits(x, 90, 1);
    if (!ext && bits(x, 87, 4) != 15) return (void)fail(l, "LEA carry in without .X");
    if (neg_b && (ext || cout != 7 || neg_a)) return (void)fail(l, "LEA negated addend");
    if (neg_b && !bits(x, 80, 1)) {
        // Rd = (a << shift) - b: the shift (when there is one), then IADD with the second operand negated (bit 48).
        uint32_t t = a.reg, rb = to_reg(l, b, 1, 0, 0);
        if (shift) {
            t = S(l, 2);
            emit(l, base(l, 0x3848000000000000ull, t, a.reg) | ((uint64_t)shift << 20));
        }
        emit(l, base(l, 0x5c10000000000000ull, rd, t) | ((uint64_t)rb << 20) | ((uint64_t)1 << 48));
        return;
    }
    if (bits(x, 80, 1)) {
        // LEA.HI: Rd = Rb + the high word of (Rc:Ra << shift): SHF.L Rt = (Rc:Ra) << shift high, or a right shift
        // of Ra by 32 - shift when Rc is RZ (sign-extended by 73); then the add, with a carry out when asked.
        uint32_t rc = bits(x, 64, 8), t = S(l, 2);
        if (neg_a || bits(x, 72, 1)) return (void)fail(l, "LEA.HI negation");
        if (rc == RZ) {
            emit(l, base(l, 0x3828000000000000ull, t, a.reg) | ((uint64_t)(32 - shift) << 20) | ((uint64_t)bits(x, 73, 1) << 48));
        } else {
            if (bits(x, 73, 1)) return (void)fail(l, "LEA.HI signed with a high word");
            emit(l, base(l, 0x36f8000000000000ull, t, a.reg) | ((uint64_t)shift << 20) | ((uint64_t)rc << 39));
        }
        uint32_t rb = to_reg(l, b, 1, 0, 0);
        if (cout == 7 && !ext) {
            emit(l, base(l, 0x5c10000000000000ull, rd, t) | ((uint64_t)rb << 20) | ((uint64_t)neg_b << 48));   // 48: -b
            return;
        }
        emit(l, base(l, 0x5c10000000000000ull, t, t) | ((uint64_t)rb << 20));
        if (cout != 7)       // Pc = t < b (unsigned): the sum wrapped
            emit(l, 0x5b60000000000000ull | l->pred | 7 | ((uint64_t)cout << 3) | ((uint64_t)t << 8) | ((uint64_t)rb << 20) |
                        ((uint64_t)7 << 39) | ((uint64_t)1 << 49));
        if (ext) add_carry_in(l, t, cin, cin_inv);
        mov_reg(l, rd, t);
        return;
    }
    if (cout == 7 && !ext) {
        int is32;
        uint64_t w = b_slot(l, b, 0x5bd0000000000000ull, 0x4bd0000000000000ull, 0x36d0000000000000ull, 0, 0, 1, &is32);
        emit(l, w | l->pred | rd | ((uint64_t)a.reg << 8) | ((uint64_t)shift << 39) | ((uint64_t)neg_a << 45));
        return;
    }
    // With a carry out or in: Ryujinx ignores Maxwell LEA's carry predicate and X, so the sum is made by hand:
    // t = (a << shift) + b, the carry out Pc = t < b (unsigned), the carry in added from its predicate.
    if (neg_a) return (void)fail(l, "LEA carry with a negated source");
    uint32_t t = S(l, 2), rb = to_reg(l, b, 1, 0, 0);
    emit(l, base(l, 0x3848000000000000ull, t, a.reg) | ((uint64_t)shift << 20));   // SHL imm
    emit(l, base(l, 0x5c10000000000000ull, t, t) | ((uint64_t)rb << 20));
    if (cout != 7)
        emit(l, 0x5b60000000000000ull | l->pred | 7 | ((uint64_t)cout << 3) | ((uint64_t)t << 8) | ((uint64_t)rb << 20) |
                    ((uint64_t)7 << 39) | ((uint64_t)1 << 49));
    if (ext) add_carry_in(l, t, cin, cin_inv);
    mov_reg(l, rd, t);
}

// IMAD (0x024 forms): a * b + c, low 32 bits (signedness 73 does not change them). Also the compiler's moves
// (IMAD.MOV.U32 Rd, RZ, RZ, x), shifts (IMAD.SHL by a power of two) and adds (a * 1 + c) in that form.
// IMAD.X (bit 75 with an immediate 1 multiplier... ) is not lowered.
static void lower_imad(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8);
    Src a = src0(x, 0), b = srcn(l, x, 1, 0), c = srcn(l, x, 2, 0);
    int ext = bits(x, 74, 1), neg_c = bits(x, 75, 1);
    if (bits(x, 76, 5) || bits(x, 81, 3) != 7 || (bits(x, 84, 3) != 0 && bits(x, 84, 3) != 7) || (!ext && bits(x, 87, 4) != 15))
        return (void)fail(l, "IMAD mode");
    if (ext) {
        // IMAD.X: a * b + c + the carry predicate (87-89, inverted by 90): the product and addend by the plain
        // lowering into Rd, then the carry; the plain lowering reads Rd's sources before it writes Rd.
        if (neg_c) return (void)fail(l, "IMAD.X negated addend");
        Insn y = *x;
        y.w[2] &= ~(1u << (74 - 64));                        // the plain instruction
        y.w[2] |= 0xfu << (87 - 64);                         // with no carry in
        lower_imad(l, &y, op);
        if (l->failed) return;
        add_carry_in(l, rd, bits(x, 87, 3), bits(x, 90, 1));
        return;
    }
    if (a.reg == RZ && b.kind == 0 && b.reg == RZ) {                   // IMAD.MOV.U32 Rd, RZ, RZ, c
        if (neg_c && c.kind == 0) {
            emit(l, base(l, 0x5c10000000000000ull, rd, RZ) | ((uint64_t)c.reg << 20) | ((uint64_t)1 << 48));
            return;
        }
        if (neg_c) return (void)fail(l, "IMAD.MOV negated constant");
        if (c.kind == 1) return mov_imm(l, rd, c.imm);
        if (c.kind == 2) { if (!cbuf_ok(&c)) return (void)fail(l, "IMAD.MOV constant buffer"); return mov_cbuf(l, rd, &c); }
        return mov_reg(l, rd, c.reg);
    }
    if (b.kind == 1 && b.imm != 0 && (b.imm & (b.imm - 1)) == 0 && c.kind == 0 && c.reg == RZ && !neg_c && b.imm > 1) {
        uint32_t sh = 0;
        while ((1u << sh) != b.imm) sh++;
        emit(l, base(l, 0x3848000000000000ull, rd, a.reg) | ((uint64_t)sh << 20));   // SHL imm
        return;
    }
    if (b.kind == 1 && b.imm == 1) {                                        // a + c, c negated by 75
        uint32_t rc = to_reg(l, c, 2, 0, 0);
        emit(l, base(l, 0x5c10000000000000ull, rd, a.reg) | ((uint64_t)rc << 20) | ((uint64_t)(neg_c ? 1 : 0) << 48));
        return;
    }
    if (neg_c) return (void)fail(l, "IMAD negated addend");
    if (op == 0x224 && c.kind == 0 && c.reg != RZ && b.kind == 0 && b.reg == RZ) {   // a * 0 + c: a move
        return mov_reg(l, rd, c.reg);
    }
    // Maxwell IMAD R / I20 / C / RC: unsigned low product plus c.
    if (c.kind == 2 && cbuf_ok(&c)) {
        uint32_t rb = to_reg(l, b, 1, 0, 0);
        emit(l, base(l, 0x5200000000000000ull, rd, a.reg) | cbuf_bits(&c) | ((uint64_t)rb << 39));
        return;
    }
    uint32_t rc = to_reg(l, c, 2, 0, 0);
    int is32;
    uint64_t w = b_slot(l, b, 0x5a00000000000000ull, 0x4a00000000000000ull, 0x3400000000000000ull, 0, 0, 1, &is32);
    emit(l, w | l->pred | rd | ((uint64_t)a.reg << 8) | ((uint64_t)rc << 39));
}

// IMAD.WIDE (0x025): Rd:Rd+1 = a * b + c:c+1, 64-bit; .U32 or signed (73). Low word by IMAD, high by IMAD.HI,
// then the 64-bit add of c.
static void lower_imad_wide(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), c = bits(x, 64, 8);
    int is_signed = bits(x, 73, 1);
    Src a = src0(x, 0), b = srcn(l, x, 1, 0);
    if (bits(x, 72, 1) || bits(x, 74, 7) || bits(x, 81, 3) != 7 || (bits(x, 84, 3) != 0 && bits(x, 84, 3) != 7) ||
        bits(x, 87, 4) != 15 || rd >= 254 || (c != RZ && c >= 254))
        return (void)fail(l, "IMAD.WIDE mode");
    uint32_t rb = to_reg(l, b, 0, 0, 0);
    // The result goes straight to Rd, Rd+1 when no source lies in them but a, which is read by the second word
    // only (Rd+1 may be a source: it is read before the write); otherwise through two scratch registers. The
    // direct form needs one scratch register at most, where a vertex shader with 247 live registers has few.
    int direct = a.reg != rd && rb != rd && (c == RZ || (c != rd && c != rd + 1 && c + 1 != rd && c + 1 != rd + 1));
    // Through scratch the multiplier's own scratch register (when b was an immediate or a constant) takes the high
    // word, which is written by the instruction that last reads it: two scratch registers in all, not four.
    int rb_scratch = b.kind != 0 && rb != RZ;
    uint32_t lo = direct ? rd : S(l, rb_scratch ? 1 : 0), hi = direct ? rd + 1 : rb_scratch ? rb : S(l, 1);
    uint64_t sign = is_signed ? ((uint64_t)1 << 48) | ((uint64_t)1 << 53) : 0;
    emit(l, base(l, 0x5a00000000000000ull, lo, a.reg) | ((uint64_t)rb << 20) | ((uint64_t)RZ << 39));
    emit(l, base(l, 0x5a00000000000000ull, hi, a.reg) | ((uint64_t)rb << 20) | ((uint64_t)RZ << 39) | sign | ((uint64_t)1 << 54));
    if (c != RZ) {                                        // + c:c+1 with the carry through the condition code
        emit(l, base(l, 0x5c10000000000000ull, lo, lo) | ((uint64_t)c << 20) | ((uint64_t)1 << 47));
        emit(l, base(l, 0x5c10000000000000ull, hi, hi) | ((uint64_t)(c + 1) << 20) | ((uint64_t)1 << 43));
    }
    if (direct) return;
    mov_reg(l, rd, lo);
    mov_reg(l, rd + 1, hi);
}

// SHF: funnel shifts. low 24-31, shift b (register or immediate), high 64-71; type 73-74 (0 S64, 1 U64, 2 S32,
// 3 U32), wrap 75, right 76, high word of the result 80.
static void lower_shf(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8), low = bits(x, 24, 8), high = bits(x, 64, 8);
    uint32_t type = bits(x, 73, 2), wrap = bits(x, 75, 1), right = bits(x, 76, 1), hi_res = bits(x, 80, 1);
    Src b = srcn(l, x, 1, 0);
    if (bits(x, 77, 3) || bits(x, 81, 11)) return (void)fail(l, "SHF mode");
    if (op == 0x419) {
        // SHF.L.U32.HI Rd, RZ, Rb, imm: the immediate is the high word, shifted left by Rb: imm << Rb.
        uint32_t rs = bits(x, 64, 8), t = S(l, 2);
        if (low != RZ || !hi_res || right || type != 3) return (void)fail(l, "SHF immediate high word mode");
        mov_imm(l, t, x->w[1]);
        emit(l, base(l, 0x5c48000000000000ull, rd, t) | ((uint64_t)rs << 20) | ((uint64_t)wrap << 39));
        return;
    }
    if (op != 0x819 && op != 0x219 && op != 0xa19 && op != 0xc19) return (void)fail(l, "SHF form");
    int imm = b.kind == 1;
    uint32_t amount = b.imm;
    if (imm && amount > 32 && !wrap) return (void)fail(l, "SHF immediate above 32");
    uint32_t rb = imm ? 0 : to_reg(l, b, 1, 0, 0);
    // Plain 32-bit shifts: a zero high word (left) or a zero low word with the high result (right) select them.
    if (!right && !hi_res && high == RZ && (type == 3 || type == 1)) {           // SHL low
        if (imm) emit(l, base(l, 0x3848000000000000ull, rd, low) | ((uint64_t)amount << 20) | ((uint64_t)wrap << 39));
        else emit(l, base(l, 0x5c48000000000000ull, rd, low) | ((uint64_t)rb << 20) | ((uint64_t)wrap << 39));
        return;
    }
    if (right && hi_res && low == RZ && (type == 3 || type == 2)) {              // SHR high (U32 or S32)
        uint64_t sgn = (uint64_t)(type == 2) << 48;
        if (imm) emit(l, base(l, 0x3828000000000000ull, rd, high) | ((uint64_t)amount << 20) | ((uint64_t)wrap << 39) | sgn);
        else emit(l, base(l, 0x5c28000000000000ull, rd, high) | ((uint64_t)rb << 20) | ((uint64_t)wrap << 39) | sgn);
        return;
    }
    if (right && !hi_res && high == RZ && (type == 3 || type == 1)) {            // SHR low, zero high: logical
        if (imm) emit(l, base(l, 0x3828000000000000ull, rd, low) | ((uint64_t)amount << 20) | ((uint64_t)wrap << 39));
        else emit(l, base(l, 0x5c28000000000000ull, rd, low) | ((uint64_t)rb << 20) | ((uint64_t)wrap << 39));
        return;
    }
    if (!right && hi_res && low == RZ && type == 3) {                            // SHL of the high word alone
        if (imm) emit(l, base(l, 0x3848000000000000ull, rd, high) | ((uint64_t)amount << 20) | ((uint64_t)wrap << 39));
        else emit(l, base(l, 0x5c48000000000000ull, rd, high) | ((uint64_t)rb << 20) | ((uint64_t)wrap << 39));
        return;
    }
    // True funnel shifts: Maxwell SHF.L Rd = high word of (high:low << s) when .HI, SHF.R Rd = low word of
    // (high:low >> s). MaxShift 37-38: 0 U64 (clamp 64), 2 U32 (clamp 32); M (50) wraps.
    if (!hi_res && !right) return (void)fail(l, "SHF.L low result");
    if (hi_res && right) return (void)fail(l, "SHF.R high result");
    if (type == 0 || type == 2) return (void)fail(l, "signed funnel shift");
    uint64_t max = (uint64_t)(type == 3 ? 2 : 0) << 37;
    if (imm) {
        if (amount > 63) return (void)fail(l, "SHF immediate");
        emit(l, base(l, right ? 0x38f8000000000000ull : 0x36f8000000000000ull, rd, low) | ((uint64_t)amount << 20) |
                    ((uint64_t)high << 39) | max | ((uint64_t)wrap << 50));
    } else {
        emit(l, base(l, right ? 0x5cf8000000000000ull : 0x5bf8000000000000ull, rd, low) | ((uint64_t)rb << 20) |
                    ((uint64_t)high << 39) | max | ((uint64_t)wrap << 50));
    }
}

// I2FP (0x045): float from a 32-bit integer; signed 74, rounding 78-79. Maxwell I2F.F32.{U,S}32.
static void lower_i2fp(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8);
    if (bits(x, 75, 2) != 2 || bits(x, 77, 1) || bits(x, 80, 4) || bits(x, 84, 2) != 2 || bits(x, 86, 6) || bits(x, 72, 2) ||
        bits(x, 24, 8) || bits(x, 60, 2))
        return (void)fail(l, "I2FP mode");
    Src b = srcn(l, x, 1, 0);
    int is32;
    uint64_t w = b_slot(l, b, 0x5cb8000000000000ull, 0x4cb8000000000000ull, 0x38b8000000000000ull, 0, 0, 1, &is32);
    w |= l->pred | rd | ((uint64_t)2 << 8) | ((uint64_t)2 << 10) | ((uint64_t)bits(x, 74, 1) << 13) | ((uint64_t)bits(x, 78, 2) << 39);
    emit(l, w);
}

// I2F (0x106): from a narrower integer; source size 84-85 (0 8, 1 16, 2 32), signed 74, byte/half select 60-61,
// rounding 78-79, destination size 75-76 (2 = F32).
static void lower_i2f(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), ssize = bits(x, 84, 2), dsize = bits(x, 75, 2), sel = bits(x, 60, 2);
    if (dsize != 2 || ssize > 2 || bits(x, 24, 8) || bits(x, 77, 1) || bits(x, 80, 4) || bits(x, 86, 6))
        return (void)fail(l, "I2F mode");
    Src b = srcn(l, x, 1, 0);
    uint32_t rb = to_reg(l, b, 1, 0, 0);
    emit(l, base(l, 0x5cb8000000000000ull, rd, 0) | ((uint64_t)2 << 8) | ((uint64_t)ssize << 10) | ((uint64_t)bits(x, 74, 1) << 13) |
                ((uint64_t)rb << 20) | ((uint64_t)bits(x, 78, 2) << 39) | ((uint64_t)(ssize == 1 ? sel * 2 : sel) << 41));
}

// F2I (0x105): signed 72, destination size 75-76, NTZ 77, rounding 78-79 (0 round, 1 floor, 2 ceil, 3 trunc),
// ftz 80, source size 84-85 (2 = F32). The source may be a half (size 1) selected by bit 60.
static void lower_f2i(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), ssize = bits(x, 84, 2), dsize = bits(x, 75, 2);
    if (bits(x, 24, 8) || bits(x, 73, 2) || bits(x, 81, 3) || bits(x, 86, 6)) return (void)fail(l, "F2I mode");
    if (ssize != 2) return (void)fail(l, "F2I half source");
    if (dsize == 3) return (void)fail(l, "F2I 64-bit destination");
    Src b = srcn(l, x, 1, 1);
    uint32_t rb = to_reg(l, b, 1, 1, 1);
    emit(l, base(l, 0x5cb0000000000000ull, rd, 0) | ((uint64_t)dsize << 8) | ((uint64_t)2 << 10) | ((uint64_t)bits(x, 72, 1) << 12) |
                ((uint64_t)rb << 20) | ((uint64_t)bits(x, 78, 2) << 39) | ((uint64_t)bits(x, 80, 1) << 44) |
                ((uint64_t)b.neg << 45) | ((uint64_t)b.abs_ << 49));
}

// FRND (0x107): to an integral float; rounding 78-79, ftz 80, sizes 75-76 / 84-85 (2 = F32). Maxwell F2F.F32.F32
// with an integer rounding mode (bit 42 set, the mode in 39-40).
static void lower_frnd(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8);
    if (bits(x, 75, 2) != 2 || bits(x, 84, 2) != 2 || bits(x, 24, 8) || bits(x, 77, 1) || bits(x, 81, 3) || bits(x, 86, 6))
        return (void)fail(l, "FRND mode");
    Src b = srcn(l, x, 1, 1);
    uint32_t rb = to_reg(l, b, 1, 1, 1);
    emit(l, base(l, 0x5ca8000000000000ull, rd, 0) | ((uint64_t)2 << 8) | ((uint64_t)2 << 10) | ((uint64_t)rb << 20) |
                ((uint64_t)(8 + bits(x, 78, 2)) << 39) | ((uint64_t)bits(x, 80, 1) << 44) | ((uint64_t)b.neg << 45) |
                ((uint64_t)b.abs_ << 49));
}

// MUFU: operation 74-79 (0 cos, 1 sin, 2 ex2, 3 lg2, 4 rcp, 5 rsq, 6 rcp64h, 7 rsq64h, 8 sqrt, 9 tanh); the
// source in the b slot. SM70+ sine and cosine take turns; Maxwell's take radians: scale by 2 pi first.
static void lower_mufu(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), fn = bits(x, 74, 6);
    if (fn > 8 || bits(x, 73, 1) || bits(x, 24, 8) || bits(x, 80, 12)) return (void)fail(l, "MUFU mode");
    Src b = srcn(l, x, 1, 1);
    uint32_t rb = to_reg(l, b, 1, 1, 1);
    if (fn <= 1) {
        uint32_t t = S(l, 2);
        emit(l, base(l, 0x1e00000000000000ull, t, rb) | ((uint64_t)0x40c90fdb << 20));   // FMUL32I by 2 pi
        rb = t;
    }
    emit(l, base(l, 0x5080000000000000ull, rd, rb) | ((uint64_t)fn << 20) | ((uint64_t)b.abs_ << 46) | ((uint64_t)b.neg << 48));
}

// IPA: the attribute field at 64-73 counts dwords (28 = position.x at a[0x70], 32 = a[0x80]); sample mode 76-77,
// constant (.PASS) 78, the interpolation weight register in the b slot (RZ: the usual one). Ra (an offset for the
// offset mode, which is refused) is ignored, so IPA 0x526, whose Ra is a uniform register (bit 91; UR4 holding 0 in
// the two fragment shaders that use it), lowers as 0x326 does.
static void lower_ipa(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), attr = bits(x, 64, 10);
    if (bits(x, 32, 8) != RZ || bits(x, 76, 2) > 1 || bits(x, 79, 1) || bits(x, 81, 3) != 7 || attr * 4 > 0x3ff)
        return (void)fail(l, "IPA mode");
    emit(l, base(l, 0xe000000000000000ull, rd, RZ) | ((uint64_t)RZ << 20) | ((uint64_t)(attr * 4) << 28) |
                ((uint64_t)7 << 47) | ((uint64_t)bits(x, 76, 2) << 52) | ((uint64_t)bits(x, 78, 1) << 55));
}

// ALD / AST: address 40-49 (bytes), components 74-75 (+ 1), patch 76, output 79; the vertex index register
// at 32-39 (ALD) or 64-71 (AST); AST's data at 32-39. Maxwell ALD/AST with the same fields.
static void lower_ald_ast(L *l, const Insn *x, int store) {
    uint32_t address = bits(x, 40, 10), count = bits(x, 74, 2) + 1;
    uint32_t patch = bits(x, 76, 1), output = bits(x, 79, 1);
    if (bits(x, 77, 1) || patch || bits(x, 24, 8) != RZ || address % 4 || address + count * 4 > 0x400)
        return (void)fail(l, "attribute access mode");
    if (output && (store || !patch)) return (void)fail(l, "attribute output read");
    uint32_t value = store ? bits(x, 32, 8) : bits(x, 16, 8);
    uint32_t vertex = store ? bits(x, 64, 8) : bits(x, 32, 8);
    if (value == RZ) {
        if (!store) return;
        if (count > 4) return (void)fail(l, "AST of RZ");
        for (uint32_t i = 0; i < count; i++) mov_imm(l, S(l, i), 0);
        value = S(l, 0);
    }
    emit(l, (store ? 0xeff0000000000000ull : 0xefd8000000000000ull) | l->pred | value | ((uint64_t)RZ << 8) |
                ((uint64_t)address << 20) | ((uint64_t)patch << 31) | ((uint64_t)output << 32) | ((uint64_t)vertex << 39) |
                ((uint64_t)(count - 1) << 47));
}

// LDC (0x182, indexed): index register 24-31, byte offset 38-53, bank 54-58, width 73-75 (Maxwell's codes),
// mode 78-79 (0, or 2 for .IS).
static void lower_ldc(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), width = bits(x, 73, 3), mode = bits(x, 78, 2), bank = bits(x, 54, 5), off = bits(x, 38, 16);
    if (width > 5 || (mode != 0 && mode != 2) || bank >= 18 || bits(x, 76, 2) || bits(x, 80, 12) || bits(x, 32, 6))
        return (void)fail(l, "LDC mode");
    emit(l, base(l, 0xef90000000000000ull, rd, bits(x, 24, 8)) | ((uint64_t)off << 20) | ((uint64_t)bank << 36) |
                ((uint64_t)mode << 44) | ((uint64_t)width << 48));
}

// ULDC (0x0b9 direct, indexed by a uniform register with 0xabb): into the register standing for the uniform.
static void lower_uldc(L *l, const Insn *x, uint32_t op) {
    uint32_t dst = bits(x, 16, 8), width = bits(x, 73, 3), bank = bits(x, 54, 5), off = bits(x, 38, 16);
    if (bits(x, 76, 2) || bits(x, 80, 11) || bits(x, 91, 1) != (op == 0xabb) || bank >= 18 || width > 5) return (void)fail(l, "ULDC mode");
    if (dst < 63 && !ur_has_reg(l, dst) && (width == 4 || width == 5) && (dst + 1 >= 63 || width == 4 || !ur_has_reg(l, dst + 1)))
        return;                                               // every use forwards the constant
    uint32_t target = ur_dst(l, dst);
    uint32_t index = op == 0xabb ? ur_in(l, bits(x, 24, 8), 5) : RZ;
    if (target == RZ) return;
    if (width == 4 && index == RZ) {
        Src s = {0}; s.kind = 2, s.cb_off = off, s.cb_bank = bank;
        if (!cbuf_ok(&s)) return (void)fail(l, "ULDC address");
        return mov_cbuf(l, target, &s);
    }
    if (width == 5) {
        uint32_t t2 = ur_dst(l, dst + 1);
        if (t2 == RZ && target == RZ) return;
        if ((dst & 1) || t2 != target + 1) {
            // A pair that is not adjacent in the mapping: load both words separately.
            emit(l, base(l, 0xef90000000000000ull, target, index) | ((uint64_t)off << 20) | ((uint64_t)bank << 36) | ((uint64_t)4 << 48));
            emit(l, base(l, 0xef90000000000000ull, t2, index) | ((uint64_t)(off + 4) << 20) | ((uint64_t)bank << 36) | ((uint64_t)4 << 48));
            return;
        }
    }
    emit(l, base(l, 0xef90000000000000ull, target, index) | ((uint64_t)off << 20) | ((uint64_t)bank << 36) | ((uint64_t)width << 48));
}

// S2R / S2UR: special register 72-79, same numbering as Maxwell for the ones the translator knows.
static void lower_s2r(L *l, const Insn *x, uint32_t op) {
    uint32_t sr = bits(x, 72, 8), rd = bits(x, 16, 8);
    if (x->w[1] || (x->w[2] & 0xffff00ffu)) return (void)fail(l, "S2R mode");
    uint32_t target = op == 0x9c3 ? ur_dst(l, rd) : rd;
    if (sr == 132) return mov_imm(l, target, 0x00010100);           // SR_VARIABLE_RATE: 1x1 shading
    if (!(sr == 0 || (sr >= 17 && sr <= 19) || (sr >= 29 && sr <= 35) || (sr >= 37 && sr <= 39) || (sr >= 56 && sr <= 60)))
        return (void)fail(l, "special register");
    emit(l, 0xf0c8000000000000ull | l->pred | target | ((uint64_t)sr << 20));
}

// CS2R: a 64-bit special register pair; SRZ (255) gives zeros.
static void lower_cs2r(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), sr = bits(x, 72, 8);
    if (sr != 255 || rd == RZ) return (void)fail(l, "CS2R register");
    mov_imm(l, rd, 0);
    if (bits(x, 80, 1)) mov_imm(l, rd + 1, 0);
}

// LDG (0x181 register address, 0x981 with a uniform base at 32-37 or none): size 73-75 (Maxwell's), 24-bit
// offset at 40-63, address register pair at 24-31 (.E).
static void lower_ldg(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8), size = bits(x, 73, 3), address = bits(x, 24, 8);
    uint32_t offset = bits(x, 40, 24);
    if (size > 6 || (size == 5 && (rd & 1)) || (size == 6 && (rd & 3))) return (void)fail(l, "LDG size");
    if (op == 0x381) {
        if ((x->w[2] & ~0x2e00u) != 0x001e8100u || (address & 1)) return (void)fail(l, "LDG register mode");
    } else {
        // A wide load's destination (its first two registers, aligned) holds the uniform base while the address is
        // formed when the pair is not already in registers: the address is read before the data is written, and a
        // vertex shader with 247 live registers has no scratch run for the pair.
        uint32_t spare = size >= 5 && rd != RZ && rd < 253 ? rd : NOREG;
        if (spare != NOREG && address != RZ && address + 1 >= spare && address <= spare + 1) spare = NOREG;   // the address is in it
        uint32_t ubase = bits(x, 38, 2) ? NOREG : ur_pair_in(l, bits(x, 32, 6), 2, spare, 0);
        int descriptor = (x->w[2] & ~(0x3fu | 0xe00u)) == 0x0c1e1100u || x->w[2] == 0x0c1ebd00u;
        if ((!descriptor && (x->w[2] & ~(0x04000000u | 0x3fu | 0xe00u | 0x2000u)) != 0x081e8100u) || ubase == NOREG)
            return (void)fail(l, "LDG uniform mode");
        if (descriptor) {
            if (address & 1) return (void)fail(l, "LDG descriptor address");
        } else if (address == RZ) {
            address = ubase;
        } else {
            // 64-bit add of the uniform pair and the register (pair with bit 90, else zero-extended).
            // In place over the base when it was brought into the spare registers.
            int inplace = spare != NOREG && ubase == spare;
            uint32_t lo = inplace ? ubase : S(l, 0), hi = inplace ? ubase + 1 : S(l, 1);
            if (ubase == RZ) return (void)fail(l, "LDG uniform zero base");
            emit(l, base(l, 0x5c10000000000000ull, lo, ubase) | ((uint64_t)address << 20) | ((uint64_t)1 << 47));
            emit(l, base(l, 0x5c10000000000000ull, hi, ubase + 1) | ((uint64_t)(bits(x, 90, 1) ? address + 1 : RZ) << 20) | ((uint64_t)1 << 43));
            address = lo;
            if (hi != lo + 1) return (void)fail(l, "LDG scratch pair");
        }
    }
    emit(l, base(l, 0xeed0000000000000ull, rd, address) | ((uint64_t)offset << 20) | ((uint64_t)1 << 45) | ((uint64_t)size << 48));
}

// STG (0x186 register address, 0x986 uniform base at 64-69): data at 32-39.
static void lower_stg(L *l, const Insn *x, uint32_t op) {
    uint32_t data = bits(x, 32, 8), size = bits(x, 73, 3), address = bits(x, 24, 8), offset = bits(x, 40, 24);
    if (size > 6 || bits(x, 16, 8)) return (void)fail(l, "STG mode");
    if (data == RZ && size > 4) {
        // A wide store of zeros needs an aligned run of zeroed registers: scratch from index 4 on (0 to 3 serve the
        // address), moved up to the alignment.
        uint32_t n = size == 6 ? 4 : 2, first = S(l, 4);
        int k = 4 + (int)((n - first % n) % n);
        data = S(l, k);
        for (uint32_t i = 0; i < n; i++) mov_imm(l, S(l, k + (int)i), 0);
        if (l->failed) return;
    }
    if (op == 0x986) {
        uint32_t ubase = ur_pair(l, bits(x, 64, 6), 2);
        int descriptor = (x->w[2] & ~(0x3fu | 0xe00u)) == 0x0c101100u;
        if ((!descriptor && (x->w[2] & ~(0x04000000u | 0x3fu | 0xe00u)) != 0x0810a100u) || !bits(x, 72, 1))
            return (void)fail(l, "STG uniform mode");
        if (descriptor) {
            if (address & 1) return (void)fail(l, "STG descriptor address");
        } else if (address == RZ) {
            address = ubase;
        } else {
            uint32_t lo = S(l, 0), hi = S(l, 1);
            if (ubase == RZ) return (void)fail(l, "STG uniform zero base");
            emit(l, base(l, 0x5c10000000000000ull, lo, ubase) | ((uint64_t)address << 20) | ((uint64_t)1 << 47));
            emit(l, base(l, 0x5c10000000000000ull, hi, ubase + 1) | ((uint64_t)(bits(x, 90, 1) ? address + 1 : RZ) << 20) | ((uint64_t)1 << 43));
            address = lo;
            if (hi != lo + 1) return (void)fail(l, "STG scratch pair");
        }
    } else if ((x->w[2] & ~0xe00u) != 0x0010a100u || (address & 1)) {
        return (void)fail(l, "STG register mode");
    }
    emit(l, 0xeed8000000000000ull | l->pred | data | ((uint64_t)address << 8) | ((uint64_t)offset << 20) | ((uint64_t)1 << 45) |
                ((uint64_t)size << 48));
}

// SUST.P (0xb9b: the image handle from a constant-buffer word, its dword index at 40-53 and bank at 54-58 as TEX's,
// as the game's compute shaders bind their images): coordinates at 24-31 (their count by the dimension at 61-63,
// NVN's order 1D, buffer, 1D array, 2D, 2D array, 3D), data at 32-39, channel mask 72-75. Maxwell SUST.B (bindless, Ryujinx InstSustB):
// data 0-7, coordinates 8-15, mask 20-23, dimension 33-35, handle register 39-46; the handle is loaded from the
// constant word as TEX.B's is, and the translator's bindless elimination turns it back into the constant's slot.
static void lower_sust(L *l, const Insn *x) {
    uint32_t coord = bits(x, 24, 8), data = bits(x, 32, 8), dim = bits(x, 61, 3), mask = bits(x, 72, 4);
    if (bits(x, 76, 5) || dim > 5 || bits(x, 81, 3) != 0) return (void)fail(l, "SUST mode");
    Src h = {0};
    h.kind = 2, h.cb_off = bits(x, 40, 14) * 4, h.cb_bank = bits(x, 54, 5);
    if (!cbuf_ok(&h)) return (void)fail(l, "SUST handle");
    uint32_t hr = to_reg(l, h, 4, 0, 0);
    emit(l, 0xeb20000000000000ull | l->pred | data | ((uint64_t)coord << 8) | ((uint64_t)mask << 20) | ((uint64_t)dim << 33) |
                ((uint64_t)hr << 39));
}

// SULD.P (0xb97, 2026-11-06: the level's compute kernels, 23 of them refused for it): SUST.P's layout for the
// handle (dword index 40-53, bank 54-58), coordinates (24-31) and dimension (61-63), the destination at 16-23 (one
// register per channel of the mask at 72-75), a predicate result at 81-83 (PT throughout the corpus) and 1 at
// 84-86 in every one seen. Maxwell SULD.P.B (bindless, Ryujinx InstSuldB): dest 0-7, coordinates 8-15, channel
// mask 20-23, the predicate result 30-32 (PT: none), dimension 33-35, handle register 39-46.
static void lower_suld(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), coord = bits(x, 24, 8), dim = bits(x, 61, 3), mask = bits(x, 72, 4);
    if (bits(x, 76, 5) || dim > 5 || bits(x, 81, 3) != 7 || bits(x, 84, 3) != 1 || bits(x, 87, 5) || mask == 0)
        return (void)fail(l, "SULD mode");
    Src h = {0};
    h.kind = 2, h.cb_off = bits(x, 40, 14) * 4, h.cb_bank = bits(x, 54, 5);
    if (!cbuf_ok(&h)) return (void)fail(l, "SULD handle");
    uint32_t hr = to_reg(l, h, 4, 0, 0);
    emit(l, 0xeb00000000000000ull | l->pred | rd | ((uint64_t)coord << 8) | ((uint64_t)mask << 20) | ((uint64_t)PT << 30) |
                ((uint64_t)dim << 33) | ((uint64_t)hr << 39));
}

// SURED (0x59f, 2026-11-06: the Great Deku Tree scene's compute kernel b9141b0b, 1164 requests): a reduction of one
// 32-bit value into a surface texel. Data 32-39, coordinates 24-31, the handle's constant word as SUST's (index 40-53,
// bank 54-58), dimension 61-63 (1 buffer, 3 2D), reduction 87-89 (Add, Min, Max, Inc, Dec, And, Or, Xor on both
// sides); the rest of the word must be the one seen (U32, strong, GPU scope). Maxwell SURED.B (bindless, Ryujinx
// InstSuredB): data 0-7, coordinates 8-15, size 20-22 (U32 = 0), operation 24-26, dimension 33-35, handle 39-46.
// Fields as Lichii's Ampere front end reads them (GPL, for the field meanings only).
static void lower_sured(L *l, const Insn *x) {
    uint32_t coord = bits(x, 24, 8), data = bits(x, 32, 8), dim = bits(x, 61, 3);
    if ((x->w[2] & ~0x03800000u) != 0x0010e000u || (dim != 1 && dim != 3) || bits(x, 59, 2) || coord > 253)
        return (void)fail(l, "SURED mode");
    Src h = {0};
    h.kind = 2, h.cb_off = bits(x, 40, 14) * 4, h.cb_bank = bits(x, 54, 5);
    if (!cbuf_ok(&h)) return (void)fail(l, "SURED handle");
    uint32_t hr = to_reg(l, h, 4, 0, 0);
    emit(l, 0xeb50000000000000ull | l->pred | data | ((uint64_t)coord << 8) | ((uint64_t)bits(x, 87, 3) << 24) |
                ((uint64_t)dim << 33) | ((uint64_t)hr << 39));
}

// ---- tensor-core instructions (Turing+), emulated with shuffles and half-precision FMAs
//
// The game's neural upscaler (NDSR) runs on them. The fragment layouts are PTX's (mma.m16n8k16, ldmatrix,
// movmatrix): lane = 4 * g + t (g = lane >> 2, t = lane & 3); an 8x8 half matrix fragment gives lane the pair
// (row g, columns 2t and 2t + 1) in one register. Lane constants come from S2R SR_LANEID each time (scratch is per
// instruction). SHFL.IDX with a register lane and clamp 31 reads another lane's register; its validity predicate
// goes to PT (48-50), not P0, which the program may hold live across the emulated instruction.
#define SHFL_IDX_R(rd, ra, lane) (0xef10000000000000ull | (rd) | ((uint64_t)(ra) << 8) | ((uint64_t)(lane) << 20) | \
                                  ((uint64_t)1 << 29) | ((uint64_t)0x1f << 34) | ((uint64_t)PT << 48))
#define IADD_IMM(rd, ra, imm) (0x3810000000000000ull | (rd) | ((uint64_t)(ra) << 8) | ((uint64_t)((imm) & 0xfffff) << 20))
#define SHL_IMM(rd, ra, sh) (0x3848000000000000ull | (rd) | ((uint64_t)(ra) << 8) | ((uint64_t)(sh) << 20))
#define SHR_IMM(rd, ra, sh) (0x3828000000000000ull | (rd) | ((uint64_t)(ra) << 8) | ((uint64_t)(sh) << 20))
#define SHR_R(rd, ra, rb) (0x5c28000000000000ull | (rd) | ((uint64_t)(ra) << 8) | ((uint64_t)(rb) << 20))
#define PRMT_IMM(rd, ra, sel, rc) (0x36c0000000000000ull | (rd) | ((uint64_t)(ra) << 8) | ((uint64_t)(sel) << 20) | ((uint64_t)(rc) << 39))
#define LDS(rd, ra, off, size) (0xef48000000000000ull | (rd) | ((uint64_t)(ra) << 8) | ((uint64_t)((off) & 0xffffff) << 20) | ((uint64_t)(size) << 48))
#define S2R_LANEID(rd) (0xf0c8000000000000ull | (rd))
// HFMA2_R (defined with lower_hfma2): Rd = a(swizzle) * b + c on half pairs; a swizzle 2 = H0_H0, 3 = H1_H1.

static int stub_on(const char *what) {   // OOT_SM86_STUB=hmma|ldsm|all (diagnostic): the emulation replaced by zeros, to time the rest
    const char *e = getenv("OOT_SM86_STUB");
    return e && (!strcmp(e, what) || !strcmp(e, "all"));
}

static int overlaps(uint32_t a, uint32_t na, uint32_t b, uint32_t nb) { return a != RZ && b != RZ && a < b + nb && b < a + na; }

// LDSM (0x83b): count 72-73 (1, 2, 4 matrices), transposed 78, row address register 24-31 (+ a uniform register
// at 32-37 by 91), 24-bit offset 40-63, Rd..Rd+count-1. Matrix m's eight row addresses sit in lanes 8m..8m+7.
// Plain: lane gets the 32-bit pair at row g, column 2t of each matrix (row address from lane 8m + g, + 4t).
// Transposed: the pair is (row 2t, col g), (row 2t + 1, col g): two 16-bit loads from two rows' addresses, + 2g.
static void lower_ldsm(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), ra = bits(x, 24, 8), offset = bits(x, 40, 24), count = 1u << bits(x, 72, 2), trans = bits(x, 78, 1);
    if (bits(x, 74, 4) || bits(x, 79, 12) || count > 4 || rd == RZ || rd + count > 255) return (void)fail(l, "LDSM mode");
    if (stub_on("ldsm")) { for (uint32_t m = 0; m < count; m++) mov_imm(l, rd + m, 0); return; }
    if (bits(x, 91, 1)) {
        uint32_t u = to_reg(l, ur_src(l, bits(x, 32, 6)), 6, 0, 0);
        emit(l, base(l, 0x5c10000000000000ull, S(l, 6), ra) | ((uint64_t)u << 20));
        ra = S(l, 6);
    } else if (overlaps(rd, count, ra, 1)) {
        mov_reg(l, S(l, 6), ra);
        ra = S(l, 6);
    }
    uint32_t g = S(l, 0), t = S(l, 1), idx = S(l, 2), addr = S(l, 3);
    emit(l, l->pred | S2R_LANEID(g));
    emit(l, l->pred | (uint64_t)(LOP32I | t | ((uint64_t)g << 8) | ((uint64_t)3 << 20)));            // t = lane & 3
    emit(l, l->pred | SHR_IMM(g, g, 2));                                                              // g = lane >> 2
    if (!trans) {
        emit(l, l->pred | SHL_IMM(t, t, 2));                                                          // 4t
        for (uint32_t m = 0; m < count; m++) {
            emit(l, l->pred | IADD_IMM(idx, g, 8 * m));
            emit(l, l->pred | SHFL_IDX_R(addr, ra, idx));
            emit(l, base(l, 0x5c10000000000000ull, addr, addr) | ((uint64_t)t << 20));
            emit(l, l->pred | LDS(rd + m, addr, offset, 4));
        }
        return;
    }
    uint32_t h0 = S(l, 4), h1 = S(l, 5);
    emit(l, l->pred | SHL_IMM(t, t, 1));                                                              // 2t
    emit(l, l->pred | SHL_IMM(g, g, 1));                                                              // 2g
    for (uint32_t m = 0; m < count; m++) {
        for (uint32_t i = 0; i < 2; i++) {
            emit(l, l->pred | IADD_IMM(idx, t, 8 * m + i));
            emit(l, l->pred | SHFL_IDX_R(addr, ra, idx));
            emit(l, base(l, 0x5c10000000000000ull, addr, addr) | ((uint64_t)g << 20));
            emit(l, l->pred | LDS(i ? h1 : h0, addr, offset, 2));
        }
        emit(l, l->pred | PRMT_IMM(rd + m, h0, 0x5410, h1));
    }
}

// HMMA (0x23c): m16n8k16 (75-76 and 78-79 hold 1), f16 sources (82-84), f16 accumulators (76 clear): A in
// Ra..Ra+3 (rows g / g+8, k 2t.. and 2t+8..), B in Rb..Rb+1 (k 2t.., 2t+8..; column g), C in Rc..Rc+1 and D in
// Rd..Rd+1 (row g, columns 2t, 2t+1; row g+8). Lane computes its four outputs from the A rows of its group's
// four lanes and the B columns 2t and 2t+1 of lanes 8t..8t+3 and 8t+4..8t+7, two k per step, with the products
// fused into the f16 accumulators (the tensor core accumulates wider; the upscaler's network tolerates it).
int sm86_lower_tensor_cores;

static void lower_hmma(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), ra = bits(x, 24, 8), rb = bits(x, 32, 8), rc = bits(x, 64, 8);
    if (bits(x, 72, 8) != 8 || bits(x, 80, 12) || rd == RZ || ra == RZ || rb == RZ || rd > 253 || ra > 251 || rb > 253 || (rc != RZ && rc > 253))
        return (void)fail(l, "HMMA mode");
    if (stub_on("hmma")) { mov_imm(l, rd, 0), mov_imm(l, rd + 1, 0); return; }
    // The tensor cores (2026-11-06): the translator's MMA marker, prefix 0xf7f (no Maxwell instruction uses it), Rd
    // 0-7, Ra 8-15, Rb 20-27, Rc 39-46, which tools/xlate patch 0016 turns into a cooperative matrix multiply-add of
    // the same fragments. It reads every source before it writes, so overlaps need no copies. Only unconditional
    // ones: a matrix operation needs the whole subgroup, as HMMA itself does.
    int hmma_n = l->hmma_count++;   // OOT_TENSOR_HMMA_MASK=<hex> (diagnostic): only the HMMAs whose index bit is set
    const char *mask_env = getenv("OOT_TENSOR_HMMA_MASK");
    int masked = mask_env && *mask_env && hmma_n < 64 && !((strtoull(mask_env, NULL, 16) >> hmma_n) & 1);
    if (l->tensor && !masked && l->pred == (uint64_t)PT << 16) {
        emit(l, 0xf7f0000000000000ull | (uint64_t)rc << 39 | (uint64_t)rb << 20 | (uint64_t)PT << 16 | (uint64_t)ra << 8 | rd);
        return;
    }
    if (overlaps(rd, 2, ra, 4)) {                                 // the output reuses A's registers: A to scratch first
        for (uint32_t i = 0; i < 4; i++) mov_reg(l, S(l, 8 + (int)i), ra + i);
        ra = S(l, 8);
    }
    if (overlaps(rd, 2, rb, 2)) {
        if (ra == S(l, 8)) return (void)fail(l, "HMMA destination over both sources");
        mov_reg(l, S(l, 8), rb), mov_reg(l, S(l, 9), rb + 1);
        rb = S(l, 8);
    }
    if (rc == RZ) mov_imm(l, rd, 0), mov_imm(l, rd + 1, 0);
    else if (rd != rc) {
        if (overlaps(rd, 2, rc, 2)) return (void)fail(l, "HMMA destination over the accumulator");
        mov_reg(l, rd, rc), mov_reg(l, rd + 1, rc + 1);
    }
    uint32_t g4 = S(l, 0), t8 = S(l, 1), idx = S(l, 2), ta0 = S(l, 3), ta1 = S(l, 4), tb0 = S(l, 5), tb1 = S(l, 6), bk = S(l, 7);
    emit(l, l->pred | S2R_LANEID(g4));
    emit(l, l->pred | (uint64_t)(LOP32I | t8 | ((uint64_t)g4 << 8) | ((uint64_t)3 << 20)));
    emit(l, l->pred | SHL_IMM(t8, t8, 3));                                                            // 8t
    emit(l, l->pred | (uint64_t)(LOP32I | g4 | ((uint64_t)g4 << 8) | ((uint64_t)0x1c << 20)));       // 4g
    for (uint32_t h = 0; h < 2; h++) {
        for (uint32_t j = 0; j < 4; j++) {
            emit(l, l->pred | IADD_IMM(idx, g4, j));
            emit(l, l->pred | SHFL_IDX_R(ta0, ra + 2 * h, idx));          // A[g][k], A[g][k+1], k = 2j + 8h
            emit(l, l->pred | SHFL_IDX_R(ta1, ra + 2 * h + 1, idx));      // A[g+8][k], A[g+8][k+1]
            emit(l, l->pred | IADD_IMM(idx, t8, j));
            emit(l, l->pred | SHFL_IDX_R(tb0, rb + h, idx));              // B[k][2t], B[k+1][2t]
            emit(l, l->pred | IADD_IMM(idx, t8, 4 + j));
            emit(l, l->pred | SHFL_IDX_R(tb1, rb + h, idx));              // B[k][2t+1], B[k+1][2t+1]
            emit(l, l->pred | PRMT_IMM(bk, tb0, 0x5410, tb1));            // (B[k][2t], B[k][2t+1])
            emit(l, l->pred | PRMT_IMM(tb0, tb0, 0x7632, tb1));           // (B[k+1][2t], B[k+1][2t+1])
            emit(l, l->pred | HFMA2_R(rd, ta0, 2, bk, rd));
            emit(l, l->pred | HFMA2_R(rd, ta0, 3, tb0, rd));
            emit(l, l->pred | HFMA2_R(rd + 1, ta1, 2, bk, rd + 1));
            emit(l, l->pred | HFMA2_R(rd + 1, ta1, 3, tb0, rd + 1));
        }
    }
}

// MOVM (0x23a) .MT88: the fragment of the transposed 8x8 half matrix: lane's pair becomes (M[2t][g], M[2t+1][g]),
// held by lanes 8t + (g >> 1) and 8t + 4 + (g >> 1) in half g & 1.
static void lower_movm(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), ra = bits(x, 24, 8);
    if (bits(x, 72, 20) || rd == RZ) return (void)fail(l, "MOVM mode");
    uint32_t g = S(l, 0), t8 = S(l, 1), idx = S(l, 2), v0 = S(l, 3), v1 = S(l, 4), sh = S(l, 5);
    emit(l, l->pred | S2R_LANEID(g));
    emit(l, l->pred | (uint64_t)(LOP32I | t8 | ((uint64_t)g << 8) | ((uint64_t)3 << 20)));
    emit(l, l->pred | SHL_IMM(t8, t8, 3));                                                            // 8t
    emit(l, l->pred | SHR_IMM(g, g, 2));                                                              // g
    emit(l, l->pred | (uint64_t)(LOP32I | sh | ((uint64_t)g << 8) | ((uint64_t)1 << 20)));
    emit(l, l->pred | SHL_IMM(sh, sh, 4));                                                            // 16 * (g & 1)
    emit(l, l->pred | SHR_IMM(g, g, 1));                                                              // g >> 1
    emit(l, base(l, 0x5c10000000000000ull, idx, t8) | ((uint64_t)g << 20));                           // 8t + (g >> 1)
    emit(l, l->pred | SHFL_IDX_R(v0, ra, idx));
    emit(l, l->pred | IADD_IMM(idx, idx, 4));
    emit(l, l->pred | SHFL_IDX_R(v1, ra, idx));
    emit(l, l->pred | SHR_R(v0, v0, sh));
    emit(l, l->pred | SHR_R(v1, v1, sh));
    emit(l, l->pred | PRMT_IMM(rd, v0, 0x5410, v1));
}

// TEX (0xb60: handle from a constant-buffer word, its dword index at 40-53 and bank at 54-58; 0xf60: handle in a
// uniform register at 40-45; 0x361 TEX.B: handle in the operand list). dimension 61-63, mask 72-75, second
// destination 64-71, offsets 76, depth compare 78, fp16 results 79, lod mode 87-89, nodep 90. The operand list
// is (array index), coordinates, [handle], [lod], [offsets], [reference], packed into Ra then Rb: up to four
// values are split evenly (1+1, 2+1, 2+2), five or more as 4+n; with bit 60 clear the coordinates are Ra's and the
// rest Rb's. Maxwell TEX takes the coordinates in Ra.. and
// the rest in Rb..; TEX.B takes the handle first in Rb. Values already lying that way are used in place, the
// others go through scratch. Bank 2 is the game's texture constant buffer, which the translator knows as the
// bound descriptors (TEX with the word index); another bank loads the handle for TEX.B.
static void lower_tex(L *l, const Insn *x, uint32_t op) {
    static const uint32_t coord_count[8] = {1, 2, 3, 3, 2, 3, 0, 4};
    static const uint32_t maxwell_dim[8] = {0, 2, 4, 6, 1, 3, 0, 7};
    uint32_t rd = bits(x, 16, 8), rd2 = bits(x, 64, 8), ra = bits(x, 24, 8), rb = bits(x, 32, 8);
    uint32_t dim = bits(x, 61, 3), mask = bits(x, 72, 4), lod = bits(x, 87, 3);
    int offsets = bits(x, 76, 1), dc = bits(x, 78, 1), half = bits(x, 79, 1), nodep = bits(x, 90, 1);
    int bindless = op != 0xb60;
    uint32_t handle = RZ;
    if (op == 0xf60) {
        handle = to_reg(l, ur_src(l, bits(x, 40, 6)), 4, 0, 0);
        if (bits(x, 46, 13)) return (void)fail(l, "TEX uniform handle mode");
    } else if (op == 0x361) {
        if (bits(x, 40, 19)) return (void)fail(l, "TEX.B mode");
    }
    if (dim == 6 || bits(x, 77, 1) || bits(x, 81, 3) != 7 || bits(x, 59, 1) != (op == 0x361 ? 1u : 0u) ||
        (lod > 3 && lod != 6 && lod != 7) || (dc && mask != 1) || mask == 0)
        return (void)fail(l, "TEX mode");
    uint32_t coords = coord_count[dim];
    int has_lod = lod == 2 || lod == 3 || lod == 6 || lod == 7;
    uint32_t handle_slot = op == 0x361 ? 1 : 0;
    uint32_t count = coords + handle_slot + has_lod + offsets + dc;
    // Bit 60 clear: Ra holds the coordinates and Rb everything after them, for TEX as for TEX.B (measured on the
    // 3D fetches of one fragment shader, coordinates in Ra..Ra+2, then the LOD in Rb and the packed offsets in Rb+1;
    // other combinations of unpacked operands are assumed to follow the same rule, not seen).
    uint32_t in_a = !bits(x, 60, 1) ? coords : count <= 4 ? (count + 1) / 2 : op == 0x361 ? coords : 4;
    uint32_t tid = bits(x, 40, 14), bank = bits(x, 54, 5);
    uint32_t v[8];
    for (uint32_t i = 0; i < count && i < 8; i++)
        v[i] = i < in_a ? (ra == RZ ? RZ : ra + i) : (rb == RZ ? RZ : rb + (i - in_a));
    if (count > 8) return (void)fail(l, "TEX operand count");
    int bound = !bindless && bank == 2;
    if (!bindless && bank >= 18) return (void)fail(l, "TEX descriptor bank");
    // The meta list as Maxwell wants it: the handle (TEX.B), lod, offsets, reference.
    uint32_t meta[4], nmeta = 0, next = coords + handle_slot;
    uint32_t tex_handle = op == 0x361 ? v[coords] : handle;
    if (bindless) meta[nmeta++] = tex_handle;
    else if (!bound) {
        Src s = {0}; s.kind = 2, s.cb_off = tid * 4, s.cb_bank = bank;
        if (!cbuf_ok(&s)) return (void)fail(l, "TEX handle address");
        mov_cbuf(l, S(l, 4), &s);
        meta[nmeta++] = S(l, 4);
    }
    if (has_lod) meta[nmeta++] = v[next++];
    if (offsets) meta[nmeta++] = v[next++];
    if (dc) meta[nmeta++] = v[next++];
    // Coordinates: in place when contiguous, else copied to scratch 0...
    uint32_t c0 = v[0];
    int contiguous = c0 != RZ;
    for (uint32_t i = 1; i < coords; i++) contiguous &= v[i] == c0 + i;
    if (!contiguous) {
        for (uint32_t i = 0; i < coords; i++) mov_reg(l, S(l, (int)i), v[i]);
        c0 = S(l, 0);
    }
    uint32_t m0 = RZ;
    if (nmeta) {
        m0 = meta[0];
        contiguous = m0 != RZ;
        for (uint32_t i = 1; i < nmeta; i++) contiguous &= meta[i] == m0 + i;
        if (!contiguous) {
            for (uint32_t i = 0; i < nmeta; i++) if (meta[i] != S(l, 4 + (int)i)) mov_reg(l, S(l, 4 + (int)i), meta[i]);
            m0 = S(l, 4);
        }
    }
    uint32_t components = (uint32_t)__builtin_popcount(mask);
    int split = rd2 != RZ;
    // Results: Maxwell writes them to consecutive registers from the destination; the SM86 destinations are
    // Rd, Rd+1 and then Rd2..: direct when they line up (or when the extra components are dropped).
    int direct = !half && (components <= 2 || (split ? rd2 == rd + 2 : 1)) && rd != RZ;
    uint32_t target = direct ? rd : S(l, 0);
    if (!direct) {
        if (rd == RZ && components > 2 && split) direct = 0;
        target = S(l, 0);
    }
    if (!bound) {
        emit(l, 0xde80000000000000ull | l->pred | target | ((uint64_t)c0 << 8) | ((uint64_t)m0 << 20) |
                    ((uint64_t)maxwell_dim[dim] << 28) | ((uint64_t)mask << 31) | ((uint64_t)offsets << 36) |
                    ((uint64_t)lod << 37) | ((uint64_t)nodep << 49) | ((uint64_t)dc << 50) | ((uint64_t)7 << 51));
    } else {
        emit(l, 0xc000000000000000ull | l->pred | target | ((uint64_t)c0 << 8) | ((uint64_t)m0 << 20) |
                    ((uint64_t)maxwell_dim[dim] << 28) | ((uint64_t)mask << 31) | ((uint64_t)tid << 36) |
                    ((uint64_t)nodep << 49) | ((uint64_t)dc << 50) | ((uint64_t)7 << 51) | ((uint64_t)offsets << 54) |
                    ((uint64_t)lod << 55));
    }
    if (direct) return;
    if (half) {
        // Packed halves: (c0, c1) in Rd, (c2, c3) in Rd2 (or Rd+1); F2F.F16.F32 writes the half in the low 16 bits.
        for (uint32_t pair = 0; pair < 2 && pair * 2 < components; pair++) {
            uint32_t t = pair == 0 ? rd : split ? rd2 : (rd == RZ ? RZ : rd + 1);
            if (t == RZ) continue;
            uint32_t lo = S(l, (int)(pair * 2)), hi = pair * 2 + 1 < components ? S(l, (int)(pair * 2 + 1)) : RZ;
            emit(l, base(l, 0x5ca8000000000000ull, S(l, 6), 0) | ((uint64_t)1 << 8) | ((uint64_t)2 << 10) | ((uint64_t)lo << 20));
            emit(l, base(l, 0x5ca8000000000000ull, S(l, 7), 0) | ((uint64_t)1 << 8) | ((uint64_t)2 << 10) | ((uint64_t)hi << 20));
            emit(l, base(l, 0x5c18000000000000ull, t, S(l, 7)) | ((uint64_t)S(l, 6) << 20) | ((uint64_t)16 << 39));
        }
        return;
    }
    for (uint32_t i = 0; i < components; i++) {
        uint32_t t = (!split || i < 2) ? (rd == RZ ? RZ : rd + i) : rd2 + (i - 2);
        if (t != RZ) mov_reg(l, t, S(l, (int)i));
    }
}

// TLD4 (0xb63 bound handle word at 40-53 / bank 54-58; 0xf63 handle in a uniform register at 40-45): gather of
// component 87-88; dimension 61-63 (1 2D, 5 array 2D), mask 72-75, second destination 64-71, offsets 76-77 (1
// AOFFI, 2 per-tap), depth compare 78, fp16 79, nodep 90. Coordinates: x in Ra and y in Rb alone; with offsets or
// a reference, x and y in Ra, Ra+1 and the extra value in Rb; an array adds its layer first in Ra.
static void lower_tld4(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8), rd2 = bits(x, 64, 8), ra = bits(x, 24, 8), rb = bits(x, 32, 8);
    uint32_t mask = bits(x, 72, 4), dim = bits(x, 61, 3), tid = bits(x, 40, 14), bank = bits(x, 54, 5);
    uint32_t toff = bits(x, 76, 2), comp = bits(x, 87, 2);
    int dc = bits(x, 78, 1), half = bits(x, 79, 1), nodep = bits(x, 90, 1), array = dim == 5;
    int bindless = op == 0xf63;
    uint32_t handle = bindless ? to_reg(l, ur_src(l, bits(x, 40, 6)), 4, 0, 0) : RZ;
    // Per-tap offsets (2, PTP: two registers of four x/y byte pairs after the coordinates, Maxwell's TLD4.PTP) and
    // fp16 results (79: the four components packed in pairs into Rd and the second destination) since 2026-11-06:
    // the level's final pass 7034cdb8 gathers with both.
    if ((dim != 1 && !array) || toff == 3 || bits(x, 81, 3) != 7 || bits(x, 59, 1) || bits(x, 89, 1) ||
        mask == 0 || !bits(x, 60, 1) || (bindless && bits(x, 46, 13)) || (!bindless && bank >= 18) ||
        (dc && (toff || array)) || (half && (rd == RZ || (__builtin_popcount(mask) > 2 && rd2 == RZ))))
        return (void)fail(l, "TLD4 mode");
    int plain = !toff && !dc;
    uint32_t cx = ra, cy = plain && !array ? rb : (ra == RZ ? RZ : ra + 1);
    uint32_t c0;
    if (array) {
        // Maxwell array gather: (layer, x, y) in Ra..: layer and x from Ra, Ra+1, y from Rb.
        mov_reg(l, S(l, 0), ra), mov_reg(l, S(l, 1), ra == RZ ? RZ : ra + 1), mov_reg(l, S(l, 2), rb);
        c0 = S(l, 0);
    } else if (cy == cx + 1 && cx != RZ) {
        c0 = cx;
    } else {
        mov_reg(l, S(l, 0), cx), mov_reg(l, S(l, 1), cy);
        c0 = S(l, 0);
    }
    uint32_t extra = plain ? RZ : rb;               // offsets or the reference
    int bound = !bindless && bank == 2;
    uint32_t m0 = extra;
    if (!bound) {
        if (bindless) mov_reg(l, S(l, 4), handle);
        else {
            Src s = {0}; s.kind = 2, s.cb_off = tid * 4, s.cb_bank = bank;
            if (!cbuf_ok(&s)) return (void)fail(l, "TLD4 handle address");
            mov_cbuf(l, S(l, 4), &s);
        }
        if (extra != RZ) mov_reg(l, S(l, 5), extra);
        if (extra != RZ && toff == 2) mov_reg(l, S(l, 6), extra + 1);   // PTP: the second offsets register
        m0 = S(l, 4);
    }
    uint32_t components = (uint32_t)__builtin_popcount(mask);
    int split = rd2 != RZ;
    int direct = !half && (components <= 2 || (split ? rd2 == rd + 2 : 1)) && rd != RZ;
    uint32_t target = direct ? rd : S(l, 0);
    if (!bound) {
        emit(l, 0xdec0000000000000ull | l->pred | target | ((uint64_t)c0 << 8) | ((uint64_t)m0 << 20) |
                    ((uint64_t)(array ? 3 : 2) << 28) | ((uint64_t)mask << 31) | ((uint64_t)toff << 36) | ((uint64_t)comp << 38) |
                    ((uint64_t)nodep << 49) | ((uint64_t)dc << 50) | ((uint64_t)7 << 51));
    } else {
        emit(l, 0xc800000000000000ull | l->pred | target | ((uint64_t)c0 << 8) | ((uint64_t)m0 << 20) |
                    ((uint64_t)(array ? 3 : 2) << 28) | ((uint64_t)mask << 31) | ((uint64_t)tid << 36) | ((uint64_t)nodep << 49) |
                    ((uint64_t)dc << 50) | ((uint64_t)7 << 51) | ((uint64_t)toff << 54) | ((uint64_t)comp << 56));
    }
    if (direct) return;
    if (half) {   // pairs of components as halves: Rd = (c1 << 16) | c0, the second destination (c3 << 16) | c2
        for (uint32_t pair = 0; pair < 2 && pair * 2 < components; pair++) {
            uint32_t t = pair == 0 ? rd : split ? rd2 : rd + 1;
            uint32_t lo = S(l, (int)(pair * 2)), hi = pair * 2 + 1 < components ? S(l, (int)(pair * 2 + 1)) : RZ;
            emit(l, base(l, 0x5ca8000000000000ull, S(l, 6), 0) | ((uint64_t)1 << 8) | ((uint64_t)2 << 10) | ((uint64_t)lo << 20));
            emit(l, base(l, 0x5ca8000000000000ull, S(l, 7), 0) | ((uint64_t)1 << 8) | ((uint64_t)2 << 10) | ((uint64_t)hi << 20));
            emit(l, base(l, 0x5c18000000000000ull, t, S(l, 7)) | ((uint64_t)S(l, 6) << 20) | ((uint64_t)16 << 39));
        }
        return;
    }
    for (uint32_t i = 0; i < components; i++) {
        uint32_t t = (!split || i < 2) ? (rd == RZ ? RZ : rd + i) : rd2 + (i - 2);
        if (t != RZ) mov_reg(l, t, S(l, (int)i));
    }
}

// HADD2 (0x030): packed half add; .F32 output (78) with a = -RZ unpacks one half of b to a float (the
// compiler's half-to-float). Operand swizzles: a 74-75, b 60-61 (register) or 81-82 (upper slot); abs/neg a
// 73/72, b 62/63 or 83/84. Maxwell HADD2 R: a 8, b 20, b swizzle 28-29, abs b 30, neg b 31, sat 32, ftz 39,
// neg a 43, abs a 44, a swizzle 47-48, output format 49-50 (0 packed, 1 F32 low, 2 F32 high? Ryujinx OFmt).
static void lower_hadd2(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8), form = bits(x, 9, 3);
    if (bits(x, 76, 1) || bits(x, 79, 1) || (bits(x, 81, 10) != 0 && form == 1)) return (void)fail(l, "HADD2 mode");
    uint32_t a = bits(x, 24, 8), a_swz = bits(x, 74, 2);
    int a_abs = bits(x, 73, 1), a_neg = bits(x, 72, 1);
    uint32_t b, b_swz; int b_abs, b_neg;
    if (op == 0x230) {
        b = bits(x, 32, 8), b_swz = bits(x, 60, 2), b_abs = bits(x, 62, 1), b_neg = bits(x, 63, 1);
    } else if (op == 0x430) {                       // immediate half pair
        b = S(l, 0), b_swz = 0, b_abs = 0, b_neg = 0;
        mov_imm(l, b, x->w[1]);
    } else {
        Src s = {0}; s.kind = 2, s.cb_off = bits(x, 38, 16), s.cb_bank = bits(x, 54, 5);
        if (!cbuf_ok(&s)) return (void)fail(l, "HADD2 constant");
        b = S(l, 0), b_swz = bits(x, 60, 2), b_abs = bits(x, 62, 1), b_neg = bits(x, 63, 1);
        mov_cbuf(l, b, &s);
    }
    int f32 = bits(x, 78, 1);
    if (f32) {
        // F32 output: Rd = float(half) of the selected lane of b (+ a, which the compiler leaves -RZ).
        if (a != RZ || a_abs || b_swz < 2 || bits(x, 77, 1) || bits(x, 80, 1)) return (void)fail(l, "HADD2.F32 mode");
        // F2F.F32.F16 Rd, b.H0/H1 (Sh bit 41 selects the high half), with the sign of a's negation ignored
        // (a is -RZ = -0): neg b and abs b apply.
        emit(l, base(l, 0x5ca8000000000000ull, rd, 0) | ((uint64_t)2 << 8) | ((uint64_t)1 << 10) | ((uint64_t)b << 20) |
                    ((uint64_t)(b_swz == 3) << 41) | ((uint64_t)b_neg << 45) | ((uint64_t)b_abs << 49));
        return;
    }
    emit(l, base(l, 0x5d10000000000000ull, rd, a) | ((uint64_t)b << 20) | ((uint64_t)b_swz << 28) | ((uint64_t)b_abs << 30) |
                ((uint64_t)b_neg << 31) | ((uint64_t)bits(x, 77, 1) << 32) | ((uint64_t)bits(x, 80, 1) << 39) |
                ((uint64_t)a_neg << 43) | ((uint64_t)a_abs << 44) | ((uint64_t)a_swz << 47));
}

// HFMA2 (0x031): a * b + c on half pairs; a 24-31 (neg 72, abs 73, swizzle 74-75), b 32-39 register (neg 63,
// abs 62, swizzle 60-61) / 32-bit immediate (0x431) / constant (0x631), c 64-71 (swizzle 81-82, abs 83, neg 84);
// dnz 76, saturate 77, F32 output 78, ftz 80. Maxwell HFMA2 R (Ryujinx InstHfma2R): a 8, b 20, c 39, swizzles
// a 47-48, b 28-29, c 35-36, NegA 31, NegC 30, Sat 32; no NegB (folded into a) and no abs.
static void lower_hfma2(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8), a = bits(x, 24, 8), a_swz = bits(x, 74, 2), c = bits(x, 64, 8), c_swz = bits(x, 81, 2);
    int a_neg = bits(x, 72, 1), c_neg = bits(x, 84, 1), sat = bits(x, 77, 1);
    int relu = bits(x, 79, 1);                                   // .RELU (Ampere): max(result, 0), the network's activation
    if (bits(x, 73, 1) || bits(x, 83, 1) || bits(x, 78, 1) || bits(x, 85, 2)) return (void)fail(l, "HFMA2 mode");
    // Forms: 0x231 registers; 0x831 / 0xa31 an immediate / constant in b's slot; 0x431 / 0x631 in c's slot (then
    // the register at 32-39 is b). The wide operand goes to scratch first.
    uint32_t b, b_swz = 0; int b_neg = 0;
    uint32_t form = op >> 9;
    if (form == 1) {
        b = bits(x, 32, 8), b_swz = bits(x, 60, 2), b_neg = bits(x, 63, 1);
    } else if (form == 4 || form == 2) {
        if (form == 2) b = bits(x, 32, 8), b_swz = bits(x, 60, 2), b_neg = bits(x, 63, 1), c = S(l, 0), mov_imm(l, c, x->w[1]);
        else b = S(l, 0), mov_imm(l, b, x->w[1]);
    } else if (form == 5 || form == 3) {
        Src cb = {0}; cb.kind = 2, cb.cb_off = bits(x, 38, 16), cb.cb_bank = bits(x, 54, 5);
        if (!cbuf_ok(&cb)) return (void)fail(l, "HFMA2 constant");
        if (form == 3) b = bits(x, 32, 8), b_swz = bits(x, 60, 2), b_neg = bits(x, 63, 1), c = S(l, 0), mov_cbuf(l, c, &cb);
        else b = S(l, 0), b_swz = bits(x, 60, 2), b_neg = bits(x, 63, 1), mov_cbuf(l, b, &cb);
    } else {
        return (void)fail(l, "HFMA2 form");
    }
    if (bits(x, 62, 1) && form != 4) return (void)fail(l, "HFMA2 absolute operand");
    emit(l, base(l, 0x5d00000000000000ull, rd, a) | ((uint64_t)b << 20) | ((uint64_t)b_swz << 28) | ((uint64_t)c_neg << 30) |
                ((uint64_t)(a_neg ^ b_neg) << 31) | ((uint64_t)sat << 32) | ((uint64_t)c_swz << 35) | ((uint64_t)c << 39) |
                ((uint64_t)a_swz << 47));
    if (relu) h2_relu(l, rd);
}

// Half pairs on Maxwell, which has no HMNMX2 and no .RELU: m = HSET2.GT.BF (a > b: 1.0 or 0.0 per half), then
// max(a, b) = m * (a - b) + b through HADD2 and HFMA2; relu(x) = x * (x > 0) through HMUL2.
#define HSET2_GT_BF(rd, ra, aswz, rb, bswz) (0x5d18000000000000ull | (rd) | ((uint64_t)(ra) << 8) | ((uint64_t)(rb) << 20) | \
                                             ((uint64_t)(bswz) << 28) | ((uint64_t)4 << 35) | ((uint64_t)7 << 39) | ((uint64_t)(aswz) << 47) | ((uint64_t)1 << 49))
#define HMUL2_R(rd, ra, rb) (0x5d08000000000000ull | (rd) | ((uint64_t)(ra) << 8) | ((uint64_t)(rb) << 20))
#define HADD2_SUB_R(rd, ra, aswz, rb, bswz) (0x5d10000000000000ull | (rd) | ((uint64_t)(ra) << 8) | ((uint64_t)(rb) << 20) | \
                                             ((uint64_t)(bswz) << 28) | ((uint64_t)1 << 31) | ((uint64_t)(aswz) << 47))
static void h2_relu(L *l, uint32_t rd) {
    uint32_t m = S(l, 1);
    emit(l, l->pred | HSET2_GT_BF(m, rd, 0, RZ, 0));
    emit(l, l->pred | HMUL2_R(rd, rd, m));
}

// HMNMX2 (0x040, Ampere): min (predicate 87-90 true) or max of two half pairs; register form, plain operands.
static void lower_hmnmx2(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8), a = bits(x, 24, 8), b = bits(x, 32, 8), a_swz = bits(x, 74, 2), b_swz = bits(x, 60, 2);
    int is_min = bits(x, 87, 3) == 7 ? !bits(x, 90, 1) : -1;
    // Negated operands (a: bit 73, b: bit 62; the upscaler's 0a9afab0 takes max(-x, -y), 2026-11-06) become a
    // subtraction from zero into scratch first, swizzle applied, so the sequence below sees plain pairs. Absolute
    // values (72, 63) are not lowered.
    if (op != 0x240 || is_min < 0 || bits(x, 63, 1) || bits(x, 72, 1) || bits(x, 76, 11)) return (void)fail(l, "HMNMX2 mode");
    if (bits(x, 73, 1)) emit(l, l->pred | HADD2_SUB_R(S(l, 3), RZ, 0, a, a_swz)), a = S(l, 3), a_swz = 0;
    if (bits(x, 62, 1)) emit(l, l->pred | HADD2_SUB_R(S(l, 4), RZ, 0, b, b_swz)), b = S(l, 4), b_swz = 0;
    uint32_t m = S(l, 1), d = S(l, 2);
    emit(l, l->pred | (is_min ? HSET2_GT_BF(m, b, b_swz, a, a_swz) : HSET2_GT_BF(m, a, a_swz, b, b_swz)));
    emit(l, l->pred | HADD2_SUB_R(d, a, a_swz, b, b_swz));
    emit(l, l->pred | (HFMA2_R(rd, m, 0, d, b) | ((uint64_t)b_swz << 35)));
}

// HMUL2 (0x232 register b; 0x432 / 0x832 an immediate half pair in 32-63; 0xa32 a constant, as HADD2's): packed
// half multiply, HADD2's operand fields (a 24-31, swizzle 74-75, abs 73, neg 72; b swizzle 60-61, abs 62, neg 63),
// saturate 77, FMZ 80 (2026-11-06: the level's upscaler and final pass). Maxwell HMUL2 R (Ryujinx InstHmul2R):
// a 8-15, b 20-27, b swizzle 28-29, abs b 30, negate (the product) 31, sat 32, FMZ 39-40, abs a 44, a swizzle 47-48.
static void lower_hmul2(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8), a = bits(x, 24, 8), a_swz = bits(x, 74, 2);
    int a_abs = bits(x, 73, 1), a_neg = bits(x, 72, 1);
    if (bits(x, 76, 1) || bits(x, 78, 2) || (bits(x, 81, 10) != 0 && op == 0x232)) return (void)fail(l, "HMUL2 mode");
    uint32_t b, b_swz = 0; int b_abs = 0, b_neg = 0;
    if (op == 0x232) {
        b = bits(x, 32, 8), b_swz = bits(x, 60, 2), b_abs = bits(x, 62, 1), b_neg = bits(x, 63, 1);
    } else if (op == 0x432 || op == 0x832) {
        b = S(l, 0);
        mov_imm(l, b, x->w[1]);
    } else {
        Src c = {0}; c.kind = 2, c.cb_off = bits(x, 38, 16), c.cb_bank = bits(x, 54, 5);
        if (!cbuf_ok(&c)) return (void)fail(l, "HMUL2 constant");
        b = S(l, 0), b_swz = bits(x, 60, 2), b_abs = bits(x, 62, 1), b_neg = bits(x, 63, 1);
        mov_cbuf(l, b, &c);
    }
    emit(l, base(l, 0x5d08000000000000ull, rd, a) | ((uint64_t)b << 20) | ((uint64_t)b_swz << 28) | ((uint64_t)b_abs << 30) |
                ((uint64_t)(a_neg ^ b_neg) << 31) | ((uint64_t)bits(x, 77, 1) << 32) | ((uint64_t)bits(x, 80, 1) << 39) |
                ((uint64_t)a_abs << 44) | ((uint64_t)a_swz << 47));
}

// HSETP2 (0x234 register b: 32-39, swizzle 60-61, abs 62, neg 63; 0x434 an immediate half pair in 32-63): a
// 24-31, swizzle 74-75, neg 72, abs 73; combine 69-70, H_AND 71, compare 76-79, FTZ 80, P 81-83, Q 84-86, input
// predicate 87-89 (negated by 90) (Lichii's Ampere front end, GPL, read for the field meanings; 2026-11-06: the
// level's upscaler 0a9afab0). Maxwell HSETP2 R (Ryujinx InstHsetp2R): P 3-5, Q 0-2, a 8-15, b 20-27, b swizzle
// 28-29, abs b 30, neg b 31, compare 35-38, input 39-41 (negated by 42), neg a 43, abs a 44, combine 45-46,
// a swizzle 47-48, H_AND 49, FTZ 6.
static void lower_hsetp2(L *l, const Insn *x, uint32_t op) {
    uint32_t b, bswz = 0, babs = 0, bneg = 0;
    if (bits(x, 91, 1)) return (void)fail(l, "HSETP2 mode");
    if (op == 0x434) {
        b = S(l, 0);
        mov_imm(l, b, x->w[1]);
    } else {
        if ((x->w[1] & 0x0fffff00u) || bits(x, 64, 5)) return (void)fail(l, "HSETP2 mode");
        b = bits(x, 32, 8), bswz = bits(x, 60, 2), babs = bits(x, 62, 1), bneg = bits(x, 63, 1);
    }
    emit(l, 0x5d20000000000000ull | l->pred | bits(x, 84, 3) | ((uint64_t)bits(x, 81, 3) << 3) | ((uint64_t)bits(x, 80, 1) << 6) |
                ((uint64_t)bits(x, 24, 8) << 8) | ((uint64_t)b << 20) | ((uint64_t)bswz << 28) | ((uint64_t)babs << 30) |
                ((uint64_t)bneg << 31) | ((uint64_t)bits(x, 76, 4) << 35) | ((uint64_t)bits(x, 87, 3) << 39) |
                ((uint64_t)bits(x, 90, 1) << 42) | ((uint64_t)bits(x, 72, 1) << 43) | ((uint64_t)bits(x, 73, 1) << 44) |
                ((uint64_t)bits(x, 69, 2) << 45) | ((uint64_t)bits(x, 74, 2) << 47) | ((uint64_t)bits(x, 71, 1) << 49));
}

// F2FP (0x23e, Ampere) .PACK_AB: Rd = (half of Ra) << 16 | half of Rb, rounding 79-80 (round to nearest only).
// Maxwell: two F2F.F16.F32 (the half in the low 16 bits) and a PRMT.
static void lower_f2fp(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8), a = bits(x, 24, 8);
    if (op != 0x23e || bits(x, 72, 7) || bits(x, 79, 13) || bits(x, 64, 8) != RZ) return (void)fail(l, "F2FP mode");
    uint32_t b = bits(x, 32, 8), hi = S(l, 0), lo = S(l, 1);
    emit(l, base(l, 0x5ca8000000000000ull, hi, 0) | ((uint64_t)1 << 8) | ((uint64_t)2 << 10) | ((uint64_t)a << 20));
    emit(l, base(l, 0x5ca8000000000000ull, lo, 0) | ((uint64_t)1 << 8) | ((uint64_t)2 << 10) | ((uint64_t)b << 20));
    emit(l, l->pred | PRMT_IMM(rd, lo, 0x5410, hi));
}

// BAR (0xb1d): BAR.SYNC with barrier 0 and the whole workgroup (every other field 0). Maxwell BAR.SYNC 0x0.
static void lower_bar(L *l, const Insn *x) {
    if (bits(x, 16, 8) || bits(x, 24, 8) || x->w[1] || (x->w[2] & 0x0fffffffu)) return (void)fail(l, "BAR mode");
    emit(l, 0xf0a81b8000000000ull | l->pred);
}

// PRMT (0x016): byte permute, selector in the b slot (immediate usually), mode 72-74.
static void lower_prmt(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), mode = bits(x, 72, 3);
    if (bits(x, 75, 17)) return (void)fail(l, "PRMT mode");
    Src a = src0(x, 0), b = srcn(l, x, 1, 0), c = srcn(l, x, 2, 0);
    uint32_t rc = to_reg(l, c, 2, 0, 0);
    int is32;
    uint64_t w = b_slot(l, b, 0x5bc0000000000000ull, 0x4bc0000000000000ull, 0x36c0000000000000ull, 0, 0, 1, &is32);
    emit(l, w | l->pred | rd | ((uint64_t)a.reg << 8) | ((uint64_t)rc << 39) | ((uint64_t)mode << 48));
}

// POPC (0x109) and FLO (0x100): source in the b slot; FLO: signed 73, .SH 74; 63 inverts the source.
static void lower_popc_flo(L *l, const Insn *x, int flo) {
    uint32_t rd = bits(x, 16, 8);
    if (bits(x, 24, 8) || (x->w[2] & ~0x600u & ~0xe0000u) != 0 || (bits(x, 81, 3) != 7 && bits(x, 81, 3) != 0))
        return (void)fail(l, "POPC/FLO mode");
    Src b = srcn(l, x, 1, 0);
    uint32_t rb = to_reg(l, b, 1, 0, 0);
    if (flo)
        emit(l, base(l, 0x5c30000000000000ull, rd, 0) | ((uint64_t)rb << 20) | ((uint64_t)bits(x, 63, 1) << 40) |
                    ((uint64_t)bits(x, 74, 1) << 41) | ((uint64_t)bits(x, 73, 1) << 48));
    else
        emit(l, base(l, 0x5c08000000000000ull, rd, 0) | ((uint64_t)rb << 20) | ((uint64_t)bits(x, 63, 1) << 40));
}

// BREV (0x101): Maxwell SHR with .BREV (bit 40) by 0, wrap.
static void lower_brev(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8);
    if (x->w[2] || bits(x, 24, 8)) return (void)fail(l, "BREV mode");
    Src b = srcn(l, x, 1, 0);
    uint32_t rb = to_reg(l, b, 1, 0, 0);
    emit(l, base(l, 0x3828000000000000ull, rd, rb) | ((uint64_t)1 << 39) | ((uint64_t)1 << 40));
}

// SGXT (0x01a): sign (73) or zero extend the low n bits (b slot); .W 75 wraps the width.
static void lower_sgxt(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8);
    Src b = srcn(l, x, 1, 0);
    if (bits(x, 74, 1) || bits(x, 76, 16) || b.kind != 1 || b.imm > 32) return (void)fail(l, "SGXT mode");
    // BFE Rd, Ra, (width << 8) | 0: extract width bits from 0, signed by bit 48.
    emit(l, base(l, 0x3800000000000000ull, rd, bits(x, 24, 8)) | ((uint64_t)(b.imm << 8) << 20) | ((uint64_t)bits(x, 73, 1) << 48));
}

// P2R (0x803): Rd = (Ra & ~mask) | (predicates & mask) at the byte 76-77; R2P (0x804): the reverse.
static void lower_p2r_r2p(L *l, const Insn *x, int r2p) {
    uint32_t rd = bits(x, 16, 8), ra = bits(x, 24, 8);
    if (x->w[1] > 0x7f || (x->w[2] & ~0x3000u) || (r2p && rd)) return (void)fail(l, "P2R/R2P mode");
    if (r2p) emit(l, 0x38f0000000000000ull | l->pred | ((uint64_t)ra << 8) | ((uint64_t)x->w[1] << 20) | ((uint64_t)bits(x, 76, 2) << 41));
    else emit(l, base(l, 0x38e8000000000000ull, rd, ra) | ((uint64_t)x->w[1] << 20) | ((uint64_t)bits(x, 76, 2) << 41));
}

// PLOP3 (0x01c): Pd = LUT(Pa, Pb, Pc) with table bits 64-66 and 72-76, inputs at 87-90, 77-80, 68-71 (bit 3
// negates), outputs 81-83 and 84-86; bit 67 makes the third a uniform predicate. Maxwell PSETP covers
// (a op b) op2 c for and/or/xor; other tables are not lowered.
static void lower_plop3(L *l, const Insn *x) {
    uint32_t pd = bits(x, 81, 3), pd2 = bits(x, 84, 3), lut = bits(x, 64, 3) | (bits(x, 72, 5) << 3);
    uint32_t a = bits(x, 87, 4), b = bits(x, 77, 4), c = bits(x, 68, 4);
    if (pd2 != 7 || bits(x, 16, 8) || x->w[1]) return (void)fail(l, "PLOP3 mode");
    if (bits(x, 67, 1)) c = (c & 8) | up(l, c & 7);
    if (pd == 7) return;
    // Tables of (a & b) | c style: PSETP Pd, PT, Pa, Pb, Pc with BoolOpAB at 24-25 and BoolOpC at 45-46
    // (0 and, 1 or, 2 xor). The 8-bit table indexes (a, b, c) as bits 2, 1, 0.
    static const struct { uint32_t lut, op_ab, op_c; } tables[] = {
        {0x80, 0, 0}, {0xfe, 1, 1}, {0xa8, 0, 1}, {0xea, 1, 0}, {0xf8, 0, 1}, {0xc0, 0, 0}, {0xfc, 1, 0}, {0x96, 2, 2},
        {0x78, 0, 2}, {0x3c, 1, 2}, {0xe8, 2, 1}, {0xb8, 0, 0},
    };
    // An input that is PT (7) or !PT (15) is a constant: only the table rows where it has that value count, so a
    // table of two inputs (a ^ b with c = PT, the usual 0x28) matches the three-input candidates that agree on them.
    uint32_t care = 0xff;
    for (uint32_t v = 0; v < 8; v++) {
        if ((a & 7) == 7 && ((v >> 2 & 1) != !(a >> 3))) care &= ~(1u << v);
        if ((b & 7) == 7 && ((v >> 1 & 1) != !(b >> 3))) care &= ~(1u << v);
        if ((c & 7) == 7 && ((v & 1) != !(c >> 3))) care &= ~(1u << v);
    }
    // The table of (a op_ab b) op_c c over inputs negated by n (bit 2 a, bit 1 b, bit 0 c) for each candidate.
    for (uint32_t n = 0; n < 8; n++) {
        for (unsigned i = 0; i < sizeof tables / sizeof tables[0]; i++) {
            uint32_t t = 0;
            for (uint32_t v = 0; v < 8; v++) {
                uint32_t u = v ^ n, va = u >> 2 & 1, vb = u >> 1 & 1, vc = u & 1, ab, r;
                ab = tables[i].op_ab == 0 ? (va & vb) : tables[i].op_ab == 1 ? (va | vb) : (va ^ vb);
                r = tables[i].op_c == 0 ? (ab & vc) : tables[i].op_c == 1 ? (ab | vc) : (ab ^ vc);
                t |= r << v;
            }
            if ((t ^ lut) & care) continue;
            uint32_t na = a ^ ((n >> 2 & 1) << 3), nb = b ^ ((n >> 1 & 1) << 3), nc = c ^ ((n & 1) << 3);
            emit(l, 0x5090000000000000ull | l->pred | 7 | ((uint64_t)pd << 3) | ((uint64_t)(nc & 7) << 12) | ((uint64_t)(nc >> 3) << 15) |
                        ((uint64_t)tables[i].op_ab << 24) | ((uint64_t)(nb & 7) << 29) | ((uint64_t)(nb >> 3) << 32) |
                        ((uint64_t)(na & 7) << 39) | ((uint64_t)(na >> 3) << 42) | ((uint64_t)tables[i].op_c << 45));
            return;
        }
    }
    // The rest of the three-input tables: any order of the inputs (a & !(b & c) is b, c first), and the second PSETP
    // destination, which holds (!(X op Y)) op Z where the first holds (X op Y) op Z. Tried after the fixed forms
    // above so the lowering of programs that already lowered stays as it was.
    static const uint8_t perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    const uint32_t in[3] = {a, b, c};
    for (int second = 0; second < 2; second++)
        for (int pi = 0; pi < 6; pi++)
            for (uint32_t n = 0; n < 8; n++)
                for (uint32_t oab = 0; oab < 3; oab++)
                    for (uint32_t oc = 0; oc < 3; oc++) {
                        uint32_t t = 0;
                        for (uint32_t v = 0; v < 8; v++) {
                            uint32_t val[3] = {v >> 2 & 1, v >> 1 & 1, v & 1}, x = val[perms[pi][0]] ^ (n >> 2 & 1),
                                     y = val[perms[pi][1]] ^ (n >> 1 & 1), z = val[perms[pi][2]] ^ (n & 1);
                            uint32_t xy = oab == 0 ? (x & y) : oab == 1 ? (x | y) : (x ^ y);
                            if (second) xy ^= 1;
                            t |= (oc == 0 ? (xy & z) : oc == 1 ? (xy | z) : (xy ^ z)) << v;
                        }
                        if ((t ^ lut) & care) continue;
                        uint32_t nx = in[perms[pi][0]] ^ ((n >> 2 & 1) << 3), ny = in[perms[pi][1]] ^ ((n >> 1 & 1) << 3),
                                 nz = in[perms[pi][2]] ^ ((n & 1) << 3);
                        emit(l, 0x5090000000000000ull | l->pred | (second ? pd : 7) | ((uint64_t)(second ? 7 : pd) << 3) |
                                    ((uint64_t)(nz & 7) << 12) | ((uint64_t)(nz >> 3) << 15) | ((uint64_t)oab << 24) |
                                    ((uint64_t)(ny & 7) << 29) | ((uint64_t)(ny >> 3) << 32) | ((uint64_t)(nx & 7) << 39) |
                                    ((uint64_t)(nx >> 3) << 42) | ((uint64_t)oc << 45));
                        return;
                    }
    (void)fail(l, "PLOP3 table");
}

// SHFL (0x189): modes at 58-59 (idx, up, down, bfly); lane in Rb (0x589) or an immediate at 53-57; the
// clamp/segment mask as an immediate at 40-52 or in Rc (0x989); predicate output 81-83.
static void lower_shfl(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8), ra = bits(x, 24, 8), mode = bits(x, 58, 2), pout = bits(x, 81, 3);
    if (op == 0xf89) {
        if ((x->w[2] & ~0xe00u) != 0x000e0000u || bits(x, 32, 8) || bits(x, 60, 4)) return (void)fail(l, "SHFL immediate mode");
        emit(l, base(l, 0xef10000000000000ull, rd, ra) | ((uint64_t)bits(x, 53, 5) << 20) | ((uint64_t)1 << 28) | ((uint64_t)1 << 29) |
                    ((uint64_t)mode << 30) | ((uint64_t)bits(x, 40, 13) << 34) | ((uint64_t)pout << 48));
    } else if (op == 0x589) {
        if ((x->w[2] & ~0x000e0000u) || bits(x, 53, 5) || bits(x, 60, 4)) return (void)fail(l, "SHFL register lane mode");
        emit(l, base(l, 0xef10000000000000ull, rd, ra) | ((uint64_t)bits(x, 32, 8) << 20) | ((uint64_t)1 << 29) |
                    ((uint64_t)mode << 30) | ((uint64_t)bits(x, 40, 13) << 34) | ((uint64_t)pout << 48));
    } else {
        if ((x->w[1] & 0xf01fffffu) || (x->w[2] & ~0x000e00ffu)) return (void)fail(l, "SHFL register mask mode");
        emit(l, base(l, 0xef10000000000000ull, rd, ra) | ((uint64_t)bits(x, 53, 5) << 20) | ((uint64_t)1 << 28) |
                    ((uint64_t)mode << 30) | ((uint64_t)bits(x, 64, 8) << 39) | ((uint64_t)pout << 48));
    }
}

// VOTE (0x806): ballot into Rd, predicate output 81-83 (all/any/eq by 72-73), input 87-90.
static void lower_vote(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8);
    if (x->w[1] || bits(x, 72, 2) > 2 || (x->w[2] & ~0x078e0300u)) return (void)fail(l, "VOTE mode");
    emit(l, 0x50d8000000000000ull | l->pred | rd | ((uint64_t)bits(x, 87, 4) << 39) | ((uint64_t)bits(x, 81, 3) << 45) |
                ((uint64_t)bits(x, 72, 2) << 48));
}

// FSWZADD (0x022): per-lane add/subtract pattern at 32-39; Maxwell's lane codes 1 and 2 are swapped.
static void lower_fswzadd(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), swz = x->w[1];
    if (swz > 0xff || x->w[2] > 255) return (void)fail(l, "FSWZADD mode");
    uint32_t mx = ((swz & 0x55) << 1) | ((swz & 0xaa) >> 1);
    emit(l, base(l, 0x50f8000000000000ull, rd, bits(x, 24, 8)) | ((uint64_t)bits(x, 64, 8) << 20) | ((uint64_t)mx << 28));
}

// R2UR (0x1c2): the uniform takes the register's value (the first active lane's on the console; every lane holds
// the same value where the compiler relies on it, so a move stands in).
// REDUX (0x3c4, 2026-11-06: 27 of the level's compute kernels): a warp-wide reduction of a general register
// (24-31) over the active lanes into a uniform register (16-23). words[2] gives the operation: 0xc000 SUM (modulo
// 2^32), 0x14000 MAX, 0x10000 MIN (unsigned, or signed with bit 73), 0x4000 OR, 0x8000 XOR (Lichii's Ampere front
// end, GPL, read for the field meanings). Lowered as a pass over the 32 lanes: the active mask from VOTE.ANY, each
// lane's value by SHFL.IDX, turned into the operation's identity where the lane is not in the mask by mask
// arithmetic (no predicate: the programs have none to spare), and folded in. Signed MIN/MAX work in the unsigned
// order with the sign bit flipped.
static void lower_redux(L *l, const Insn *x) {
    uint32_t target = ur_dst(l, bits(x, 16, 8)), src = bits(x, 24, 8), mode = x->w[2] & ~0x200u;
    int sgn = bits(x, 73, 1), max = mode == 0x14000, min = mode == 0x10000, sum = mode == 0xc000;
    uint32_t lop = mode == 0x4000 ? 1 : mode == 0x8000 ? 2 : 0;
    if (x->w[1] || (!max && !min && !sum && !lop) || (sgn && !max && !min) || l->pred != (uint64_t)PT << 16)
        return (void)fail(l, "REDUX mode");
    if (target == RZ) return;                                   // the result is never read
    uint32_t mask = S(l, 0), t = S(l, 1), k = S(l, 2), acc = S(l, 3);
    emit(l, l->pred | 0x50d8000000000000ull | mask | ((uint64_t)PT << 39) | ((uint64_t)PT << 45) |
                ((uint64_t)1 << 48));                                                     // VOTE.ANY mask, PT
    mov_imm(l, acc, min ? 0xffffffffu : 0);
    for (uint32_t lane = 0; lane < 32; lane++) {
        emit(l, l->pred | 0xef10000000000000ull | t | ((uint64_t)src << 8) | ((uint64_t)lane << 20) | ((uint64_t)1 << 28) |
                    ((uint64_t)1 << 29) | ((uint64_t)0x1f << 34) | ((uint64_t)PT << 48));   // SHFL.IDX
        if (sgn) emit(l, l->pred | LOP32I | t | ((uint64_t)t << 8) | ((uint64_t)0x80000000u << 20) | ((uint64_t)2 << 53));      // ^ sign
        emit(l, l->pred | SHL_IMM(k, mask, 31 - lane));                                              // lane's bit to the top
        emit(l, l->pred | 0x3829000000000000ull | k | ((uint64_t)k << 8) | ((uint64_t)31 << 20));    // SHR.S32 31: 0 or ~0
        if (min)   // c = t | ~k
            emit(l, l->pred | 0x5c40000000000000ull | t | ((uint64_t)t << 8) | ((uint64_t)k << 20) | ((uint64_t)1 << 40) |
                        ((uint64_t)1 << 41) | ((uint64_t)PT << 48));
        else       // c = t & k
            emit(l, l->pred | 0x5c40000000000000ull | t | ((uint64_t)t << 8) | ((uint64_t)k << 20) | ((uint64_t)PT << 48));
        if (sum) emit(l, l->pred | 0x5c10000000000000ull | acc | ((uint64_t)acc << 8) | ((uint64_t)t << 20));   // IADD
        else if (max || min)                                                                      // IMNMX.U32
            emit(l, l->pred | 0x5c20000000000000ull | acc | ((uint64_t)acc << 8) | ((uint64_t)t << 20) | ((uint64_t)(max ? 15 : 7) << 39));
        else emit(l, l->pred | 0x5c40000000000000ull | acc | ((uint64_t)acc << 8) | ((uint64_t)t << 20) | ((uint64_t)lop << 41) |
                         ((uint64_t)PT << 48));                                                   // LOP OR / XOR
    }
    if (sgn) emit(l, l->pred | LOP32I | acc | ((uint64_t)acc << 8) | ((uint64_t)0x80000000u << 20) | ((uint64_t)2 << 53));
    mov_reg(l, target, acc);
}

static void lower_r2ur(L *l, const Insn *x) {
    uint32_t target = ur_dst(l, bits(x, 16, 8));
    if (x->w[1] || x->w[2] != 0x000e0000u) return (void)fail(l, "R2UR mode");
    if (target != RZ) mov_reg(l, target, bits(x, 24, 8));
}

// The uniform datapath's ALU ops (0x080-0x0ff) take uniform registers in every register slot and immediates in
// the b slot (forms 4 / 2); with the registers standing in they are the plain op. A UPn guard (predicate field
// other than PT) and the ops with uniform predicate operands are not lowered yet.
static void lower_uniform_alu(L *l, const Insn *x, uint32_t op) {
    uint32_t base_op = op & 0x1ff, form = op >> 9;
    uint32_t g = bits(x, 12, 4);
    if (g != 7) l->pred = ((uint64_t)up(l, g & 7) | ((uint64_t)(g >> 3) << 3)) << 16;   // @UPn guard
    uint32_t plain;
    int has_c = 1;
    switch (base_op) {
    case 0x086: {                                             // VOTEU: ballot into a uniform register, UPn output
        uint32_t target = ur_dst(l, bits(x, 16, 8)), pout = bits(x, 81, 3);
        if (x->w[1] || bits(x, 72, 2) > 2 || (x->w[2] & ~0x078e0300u) || bits(x, 24, 8)) return (void)fail(l, "VOTEU mode");
        emit(l, 0x50d8000000000000ull | l->pred | target | ((uint64_t)bits(x, 87, 4) << 39) | ((uint64_t)up(l, pout) << 45) |
                    ((uint64_t)bits(x, 72, 2) << 48));
        return;
    }
    case 0x08c: {                                             // UISETP: predicates are uniform ones
        Insn y = *x;
        y.w[0] = (x->w[0] & 0xffffu) | (ur_in(l, bits(x, 24, 8), 5) << 24);
        if (form == 1) y.w[1] = (x->w[1] & ~0xffu) | ur_in(l, bits(x, 32, 8), 6);
        y.w[2] = (x->w[2] & ~0x27fu & ~(0x7fu << 17)) | (up(l, bits(x, 68, 3)) << 4) | (up(l, bits(x, 81, 3)) << 17) |
                 (up(l, bits(x, 84, 3)) << 20) | (up(l, bits(x, 87, 3)) << 23);
        return lower_isetp(l, &y);
    }
    case 0x087: {                                             // USEL: the condition is a uniform predicate
        Insn y = *x;
        y.w[0] = (x->w[0] & 0xffffu) | (ur_dst(l, bits(x, 16, 8)) << 16) | (ur_in(l, bits(x, 24, 8), 5) << 24);
        if (form == 1) y.w[1] = (x->w[1] & ~0xffu) | ur_in(l, bits(x, 32, 8), 6);
        y.w[2] = (x->w[2] & ~(0x7u << 23) & ~(1u << 27)) | (up(l, bits(x, 87, 3)) << 23);
        return lower_sel(l, &y, 0);
    }
    case 0x09c: {                                             // UPLOP3: PLOP3 over uniform predicates
        // Same fields as PLOP3 (table 64-66 and 72-76, inputs 87-90, 77-80, 68-71, outputs 81-83 and 84-86), every
        // predicate a uniform one (UPT = 7 reads true). Each goes through up(): a free predicate, or the swapped
        // slot, whose new value lower_one merges back into the register standing for the uniform predicates.
        Insn y = *x;
        uint32_t f[5] = {bits(x, 87, 4), bits(x, 77, 4), bits(x, 68, 4), bits(x, 81, 3), bits(x, 84, 3)};
        for (int i = 0; i < 3; i++) f[i] = (f[i] & 8) | up(l, f[i] & 7);
        for (int i = 3; i < 5; i++) f[i] = up(l, f[i]);
        y.w[2] = (x->w[2] & ~(0xfu << 23 | 0xfu << 13 | 0xfu << 4 | 1u << 3 | 7u << 17 | 7u << 20 | 1u << 27)) | f[0] << 23 | f[1] << 13 |
                 f[2] << 4 | f[3] << 17 | f[4] << 20;
        return lower_plop3(l, &y);
    }
    case 0x082:                                               // UMOV: register (0xc82, UR at 32-37) or immediate (0x882)
        if (bits(x, 16, 8) == 63 || !ur_has_reg(l, bits(x, 16, 8))) return;   // URZ, or every use forwards
        if (op == 0x882) return mov_imm(l, ur(l, bits(x, 16, 8)), x->w[1]);
        if (op == 0xc82) return mov_reg(l, ur(l, bits(x, 16, 8)), ur_in(l, bits(x, 32, 6), 5));
        return (void)fail(l, "UMOV form");
    case 0x085: plain = 0x017; has_c = 0; break;
    case 0x090: plain = 0x010; break;
    case 0x091: plain = 0x011; break;
    case 0x092: plain = 0x012; break;
    case 0x096: plain = 0x016; break;
    case 0x099: plain = 0x019; break;
    case 0x09a: plain = 0x01a; has_c = 0; break;
    case 0x0a4: plain = 0x024; break;
    case 0x0a5: plain = 0x025; break;
    case 0x0bd: plain = 0x100; has_c = 0; break;
    case 0x0be: plain = 0x101; has_c = 0; break;
    case 0x0bf: plain = 0x109; has_c = 0; break;
    default: return (void)fail(l, "uniform datapath opcode");
    }
    Insn y = *x;
    int one_source = plain == 0x100 || plain == 0x101 || plain == 0x109;
    uint32_t dst = bits(x, 16, 8);
    if (plain == 0x025) {                                     // a pair: both registers, or neither (dropped)
        if (!ur_has_reg(l, dst) && !ur_has_reg(l, dst + 1) && bits(x, 81, 6) == 0x3f) return;
        if (!ur_has_reg(l, dst) || !ur_has_reg(l, dst + 1)) return (void)fail(l, "uniform pair half without a register");
        if (ur(l, dst + 1) != ur(l, dst) + 1) {
#ifdef SM86_TOOL
            if (getenv("SM86_POOL"))
                fprintf(stderr, "pair UR%u: map %u %u pair flags %u %u needs %u %u ur_next %u scratch_base %u\n", dst, l->ur_map[dst],
                        l->ur_map[dst + 1], l->ur_pair[dst], l->ur_pair[dst + 1], l->needs_reg[dst], l->needs_reg[dst + 1], l->ur_next, l->scratch_base);
#endif
            return (void)fail(l, "uniform pair not adjacent");
        }
    } else if (!ur_has_reg(l, dst) && bits(x, 81, 6) == 0x3f && dst != 63) {
        return;                                               // never read, no carry out: nothing to compute
    }
    y.w[0] = (x->w[0] & 0xffffu & ~0x1ffu) | plain | (ur_dst(l, dst) << 16) |
             ((one_source ? 0 : ur_in(l, bits(x, 24, 8), 5)) << 24);
    if (form == 1) y.w[1] = (x->w[1] & ~0xffu) | ur_in(l, bits(x, 32, 8), 6);
    if (has_c) y.w[2] = (x->w[2] & ~0xffu) | ur_in(l, bits(x, 64, 8), 7);
    y.w[2] &= ~(1u << 27);                                    // bit 91: the uniform marker
    if (plain == 0x010 || plain == 0x011 || plain == 0x012 || plain == 0x024 || plain == 0x025) {
        // The predicate fields the op has (NAK's encoders): IADD3 77, 81, 84, 87; LEA, LOP3, IMAD, IMAD.WIDE 81, 87.
        // The others are 0 in NAK's encoding, not 7, and must not be read as UP0.
        for (unsigned f = 77; f <= 87; f += (f == 77 ? 4 : 3)) {
            if (plain != 0x010 && (f == 77 || f == 84)) continue;
            uint32_t v = bits(x, f, 3);
            if (v != 7) y.w[2] = (y.w[2] & ~(7u << (f - 64))) | (up(l, v) << (f - 64));
        }
    }
    uint32_t nop = y.w[0] & 0xfff;
    switch (plain) {
    case 0x017: return lower_imnmx(l, &y);
    case 0x010: return lower_iadd3(l, &y, nop);
    case 0x011: return lower_lea(l, &y, nop);
    case 0x012: return lower_lop3(l, &y);
    case 0x016: return lower_prmt(l, &y);
    case 0x019: return lower_shf(l, &y, nop);
    case 0x01a: return lower_sgxt(l, &y);
    case 0x024: return lower_imad(l, &y, nop);
    case 0x025: return lower_imad_wide(l, &y);
    case 0x100: return lower_popc_flo(l, &y, 1);
    case 0x101: return lower_brev(l, &y);
    default: return lower_popc_flo(l, &y, 0);
    }
}


// LDL (0x983) / STL (0x387): local memory, size 73-75 (Maxwell's codes), 24-bit offset at 40-63, address register
// at 24-31; a uniform address (bit 91) at 32-37 (load) or 64-69 (store) is added first.
static void lower_ldl_stl(L *l, const Insn *x, int store) {
    uint32_t size = bits(x, 73, 3), address = bits(x, 24, 8), offset = bits(x, 40, 24);
    uint32_t value = store ? bits(x, 32, 8) : bits(x, 16, 8);
    uint32_t words = size == 6 ? 4 : size == 5 ? 2 : 1;
    if (size > 6 || bits(x, 84, 1) != 1 || (store && bits(x, 85, 1)) || bits(x, 86, 1) || bits(x, 76, 8) || (store && bits(x, 16, 8)) ||
        (!store && bits(x, 87, 4)))
        return (void)fail(l, "local memory access mode");
    if (bits(x, 91, 1)) {
        uint32_t u = to_reg(l, ur_src(l, store ? bits(x, 64, 6) : bits(x, 32, 6)), 2, 0, 0);
        if (address == RZ) address = u;
        else if (u != RZ) {
            emit(l, base(l, 0x5c10000000000000ull, S(l, 2), address) | ((uint64_t)u << 20));
            address = S(l, 2);
        }
    } else if (!store && bits(x, 32, 8)) {
        return (void)fail(l, "local load register field");
    }
    if (!store && value == RZ) return;
    if (value != RZ && (value & (words - 1))) return (void)fail(l, "local memory register alignment");
    if (!store && words > 1 && address >= value && address < value + words) {
        mov_reg(l, S(l, 2), address);
        address = S(l, 2);
    }
    for (uint32_t w = 0; w < words; w++)
        emit(l, (store ? 0xef50000000000000ull : 0xef40000000000000ull) | l->pred | (value == RZ ? RZ : value + w) |
                    ((uint64_t)address << 8) | ((uint64_t)(offset + 4 * w) << 20) | ((uint64_t)(size < 4 ? size : 4) << 48));
}

// LDS (0x984) / STS (0x388): shared memory; the register offset scaled by 78-79; a uniform base (bit 91) at 32-37
// (load) or 64-69 (store, 0x988).
static void lower_lds_sts(L *l, const Insn *x, uint32_t op) {
    int store = op != 0x984;
    uint32_t size = bits(x, 73, 3), stride = bits(x, 78, 2), address = bits(x, 24, 8), offset = bits(x, 40, 24);
    int uniform = (op == 0x984 || op == 0x988) && bits(x, 91, 1);
    if (size > 6 || bits(x, 76, 2) || bits(x, 80, 11) || (store && bits(x, 16, 8)) || (!uniform && !store && bits(x, 32, 8)) ||
        (op == 0x988 && !uniform))
        return (void)fail(l, "shared memory access mode");
    if (stride && address != RZ) {
        emit(l, base(l, 0x3848000000000000ull, S(l, 2), address) | ((uint64_t)(stride + 1) << 20));
        address = S(l, 2);
    }
    if (uniform) {
        uint32_t u = to_reg(l, ur_src(l, op == 0x988 ? bits(x, 64, 6) : bits(x, 32, 6)), 3, 0, 0);
        if (u != RZ) {
            emit(l, base(l, 0x5c10000000000000ull, S(l, 2), address) | ((uint64_t)u << 20));
            address = S(l, 2);
        }
    }
    uint32_t data = store ? bits(x, 32, 8) : bits(x, 16, 8);
    emit(l, (store ? 0xef58000000000000ull : 0xef48000000000000ull) | l->pred | data | ((uint64_t)address << 8) |
                ((uint64_t)offset << 20) | ((uint64_t)size << 48));
}

// RED (0x98e): global reduction, op 87-89 (Maxwell's numbering), type 73-75 (0 U32, 1 S32, 3 F32), data 32-39,
// register address pair at 24-31 (RZ: the uniform pair at 64-69), signed 24-bit offset.
static void lower_red(L *l, const Insn *x) {
    uint32_t type = bits(x, 73, 3), operation = bits(x, 87, 3), data = bits(x, 32, 8);
    int uniform = bits(x, 24, 8) == RZ;
    uint32_t variable = 0x03800e00u | (uniform ? 0x3fu : 0u), mode = uniform ? 0x0c10e100u : 0x0010e100u;
    int32_t offset = (int32_t)(bits(x, 40, 24) << 8) >> 8;
    if ((x->w[2] & ~variable) != mode || (type != 0 && type != 1 && type != 3) || (type == 3 && operation != 0))
        return (void)fail(l, "RED mode");
    uint32_t address = uniform ? ur_pair_in(l, bits(x, 64, 6), 2, NOREG, 1) : bits(x, 24, 8);
    if (address == NOREG || (address & 1)) return (void)fail(l, "RED address");
    emit(l, 0xebf8000000000000ull | l->pred | data | ((uint64_t)address << 8) | ((uint64_t)type << 20) |
                ((uint64_t)operation << 23) | ((uint64_t)((uint32_t)offset & 0xfffff) << 28) | ((uint64_t)1 << 48));
}

// ATOMG (0x3a8 register address pair; 0x9a8 uniform pair at 64-69 with RZ at 24-31): op 87-90, U32, data 32-39.
static void lower_atomg(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8), operation = bits(x, 87, 4), data = bits(x, 32, 8), address;
    int32_t offset = (int32_t)(bits(x, 40, 24) << 8) >> 8;
    if (op == 0x3a8) {
        address = bits(x, 24, 8);
        if ((x->w[2] & ~0x07800000u) != 0x001ee100u || operation > 8 || (address & 1)) return (void)fail(l, "ATOMG mode");
    } else {
        if (bits(x, 24, 8) != RZ || (x->w[2] & ~0x3fu & ~0x07800000u) != 0x081ee140u) return (void)fail(l, "ATOMG uniform mode");
        address = ur_pair(l, bits(x, 64, 6), 2);
        if (address == NOREG) return;
    }
    emit(l, 0xed00000000000000ull | l->pred | rd | ((uint64_t)address << 8) | ((uint64_t)data << 20) |
                ((uint64_t)((uint32_t)offset & 0xfffff) << 28) | ((uint64_t)1 << 48) | ((uint64_t)operation << 52));
}

// TXQ (0xb6f, bound; 0x370 TXQ.B): texture dimensions of the level in Ra; mask 72-75, second destination 64-71.
static void lower_txq(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8), rd2 = bits(x, 64, 8), mask = bits(x, 72, 4), tid = bits(x, 40, 14), bank = bits(x, 54, 5);
    int bindless = op == 0x370;
    uint32_t components = (uint32_t)__builtin_popcount(mask);
    if (bits(x, 59, 5) != (bindless ? 1u : 0u) || mask == 0 || bits(x, 76, 14) || bits(x, 91, 1) || (!bindless && bank >= 18))
        return (void)fail(l, "TXQ mode");
    int direct = rd2 == RZ && components <= 2 && rd != RZ;
    uint32_t target = direct ? rd : S(l, 0), ra = bits(x, 24, 8);
    int bound = !bindless && bank == 2;
    if (!bound) {
        // The handle first in Ra, the level after it: Ra, Ra+1 in scratch 4, 5.
        if (bindless) mov_reg(l, S(l, 4), ra);
        else {
            Src h = {0}; h.kind = 2, h.cb_off = tid * 4, h.cb_bank = bank;
            if (!cbuf_ok(&h)) return (void)fail(l, "TXQ handle address");
            mov_cbuf(l, S(l, 4), &h);
        }
        mov_reg(l, S(l, 5), bindless ? (ra == RZ ? RZ : ra + 1) : ra);
        ra = S(l, 4);
    }
    emit(l, (bound ? 0xdf48000000000000ull : 0xdf50000000000000ull) | l->pred | target | ((uint64_t)ra << 8) | ((uint64_t)1 << 22) |
                ((uint64_t)mask << 31) | ((uint64_t)(bound ? tid : 0) << 36) | ((uint64_t)bits(x, 90, 1) << 49));
    if (direct) return;
    for (uint32_t i = 0; i < components; i++) {
        uint32_t t = i < 2 ? (rd == RZ ? RZ : rd + i) : (rd2 == RZ ? RZ : rd2 + (i - 2));
        if (t != RZ) mov_reg(l, t, S(l, (int)i));
    }
}

// TMML (0xb69 bound, 0xf69 uniform handle, 0x36a TMML.B): the LOD of the coordinates (all in Ra when Rb is RZ,
// else split as TEX packs them); 1D/2D/3D/CUBE at 61-63 (Maxwell's dimension codes doubled).
static void lower_tmml(L *l, const Insn *x, uint32_t op) {
    static const uint32_t coord_count[4] = {1, 2, 3, 3};
    uint32_t rd = bits(x, 16, 8), rd2 = bits(x, 64, 8), ra = bits(x, 24, 8), rb = bits(x, 32, 8);
    uint32_t mask = bits(x, 72, 4), dim = bits(x, 61, 3), tid = bits(x, 40, 14), bank = bits(x, 54, 5);
    int bindless = op != 0xb69;
    uint32_t coords = dim < 4 ? coord_count[dim] : 0;
    uint32_t in_a = rb == RZ || op == 0x36a ? coords : (coords + 1) / 2;
    if (coords == 0 || bits(x, 76, 4) || bits(x, 80, 10) || mask == 0 || bits(x, 59, 1) != (op == 0x36a ? 1u : 0u) ||
        (op == 0xf69 && bits(x, 46, 13)) || (!bindless && bank >= 18))
        return (void)fail(l, "TMML mode");
    uint32_t handle = op == 0xf69 ? to_reg(l, ur_src(l, bits(x, 40, 6)), 4, 0, 0) : op == 0x36a ? rb : RZ;
    for (uint32_t i = 0; i < coords; i++) mov_reg(l, S(l, (int)i), i < in_a ? ra + i : rb + (i - in_a));
    int bound = !bindless && bank == 2;
    if (!bound) {
        if (bindless) mov_reg(l, S(l, 4), handle);
        else {
            Src h = {0}; h.kind = 2, h.cb_off = tid * 4, h.cb_bank = bank;
            if (!cbuf_ok(&h)) return (void)fail(l, "TMML handle address");
            mov_cbuf(l, S(l, 4), &h);
        }
    }
    emit(l, (bound ? 0xdf58000000000000ull : 0xdf60000000000000ull) | l->pred | S(l, 0) | ((uint64_t)S(l, 0) << 8) |
                ((uint64_t)(bound ? RZ : S(l, 4)) << 20) | ((uint64_t)(dim * 2) << 28) | ((uint64_t)mask << 31) |
                ((uint64_t)(bound ? tid : 0) << 36) | ((uint64_t)bits(x, 90, 1) << 49));
    uint32_t components = (uint32_t)__builtin_popcount(mask);
    for (uint32_t i = 0; i < components; i++) {
        uint32_t t = (rd2 == RZ || i < 2) ? (rd == RZ ? RZ : rd + i) : rd2 + (i - 2);
        if (t != RZ) mov_reg(l, t, S(l, (int)i));
    }
}

// TXD (0xb6c): gradients; coordinates in Ra (x, y[, layer]), the four derivatives in Rb; 2D (1) or array 2D (5).
static void lower_txd(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), high = bits(x, 64, 8), ra = bits(x, 24, 8), rb = bits(x, 32, 8);
    uint32_t mask = bits(x, 72, 4), dim = bits(x, 61, 3), tid = bits(x, 40, 14), bank = bits(x, 54, 5);
    uint32_t components = (uint32_t)__builtin_popcount(mask);
    if ((x->w[2] & ~(0xfffu | (1u << 26))) != 0x001e0000u || mask == 0 || bits(x, 59, 2) || (dim != 1 && dim != 5) || bank >= 18 ||
        (components > 1 && (rd & 1)))
        return (void)fail(l, "TXD mode");
    int bound = bank == 2;
    uint32_t rbm = rb;
    if (!bound) {
        Src h = {0}; h.kind = 2, h.cb_off = tid * 4, h.cb_bank = bank;
        if (!cbuf_ok(&h)) return (void)fail(l, "TXD handle address");
        mov_cbuf(l, S(l, 4), &h);
        for (uint32_t i = 0; i < 4; i++) mov_reg(l, S(l, 5 + (int)i > 7 ? 7 : 5 + (int)i), rb + i);
        return (void)fail(l, "TXD from another descriptor bank");
    }
    int consecutive = components <= 2 || high == rd + 2;
    uint32_t target = consecutive ? rd : S(l, 0);
    emit(l, 0xde00000000000000ull | l->pred | target | ((uint64_t)ra << 8) | ((uint64_t)rbm << 20) |
                ((uint64_t)(dim == 5 ? 3 : 2) << 28) | ((uint64_t)mask << 31) | ((uint64_t)tid << 36) | ((uint64_t)bits(x, 90, 1) << 49) |
                ((uint64_t)7 << 51));
    if (consecutive) return;
    for (uint32_t i = 0; i < components; i++) {
        if (i >= 2 && high == RZ) break;
        mov_reg(l, i < 2 ? rd + i : high + (i - 2), S(l, (int)i));
    }
}

// TLD (0xb66 bound, 0xf66 uniform handle): texel fetch at level zero (lod 87-89 = 1) or a level (3); 1D/2D/3D
// and array 2D; x in Ra and y in Rb for 2D, (x, y) in Ra, Ra+1 and z in Rb for 3D, (layer, x) and y for arrays;
// the level follows in the Rb list. Maxwell TLD takes the coordinates in Ra.. and the level in Rb (bit 55).
static void lower_tld(L *l, const Insn *x, uint32_t op) {
    uint32_t rd = bits(x, 16, 8), rd2 = bits(x, 64, 8), ra = bits(x, 24, 8), rb = bits(x, 32, 8);
    uint32_t mask = bits(x, 72, 4), dim = bits(x, 61, 3), tid = bits(x, 40, 14), bank = bits(x, 54, 5), lod = bits(x, 87, 3);
    int array = dim == 5, volume = dim == 2, level = lod == 3, bindless = op == 0xf66;
    // Texel offsets (bit 76, 2D only): x and y in Ra and Ra+1, then Rb holds the LOD (level fetches) and the packed
    // offsets after it, as Maxwell's TLD.AOFFI reads Ra and Rb. Taken from the 2D fetches of the Great Deku Tree
    // scene's compute kernels (Ra constant, Rb the offsets register, 40 fetches); other dimensions are refused.
    int offsets = bits(x, 76, 1);
    if (offsets && dim != 1) return (void)fail(l, "TLD mode");
    if (!bits(x, 60, 1) || (dim != 0 && dim != 1 && !array && !volume) || bits(x, 77, 4) || bits(x, 81, 3) != 7 ||
        (lod != 1 && !level) || bits(x, 59, 1) || mask == 0 || (bindless && (bits(x, 46, 13) || !bits(x, 91, 1))) ||
        (!bindless && bank >= 18))
        return (void)fail(l, "TLD mode");
    uint32_t coords[3], n = 0, lev = RZ;
    if (array || volume) coords[n++] = ra, coords[n++] = ra + 1, coords[n++] = rb;
    else if (dim == 1 && offsets) coords[n++] = ra, coords[n++] = ra == RZ ? RZ : ra + 1;
    else if (dim == 1) coords[n++] = ra, coords[n++] = rb;
    else coords[n++] = ra;
    if (level) lev = (array || volume || (dim == 1 && !offsets)) ? rb + 1 : rb;
    if (level && dim == 0) lev = rb;
    uint32_t offs = offsets ? (level ? rb + 1 : rb) : RZ;
    int contiguous = coords[0] != RZ;
    for (uint32_t i = 1; i < n; i++) contiguous &= coords[i] == coords[0] + i;
    uint32_t c0 = coords[0];
    if (!contiguous) { for (uint32_t i = 0; i < n; i++) mov_reg(l, S(l, (int)i), coords[i]); c0 = S(l, 0); }
    int bound = !bindless && bank == 2;
    uint32_t m0 = level ? lev : offs;                    // the LOD and the offsets lie side by side in Rb, Rb + 1
    if (!bound) {
        if (bindless) to_reg_into(l, ur_src(l, bits(x, 40, 6)), S(l, 4));
        else {
            Src h = {0}; h.kind = 2, h.cb_off = tid * 4, h.cb_bank = bank;
            if (!cbuf_ok(&h)) return (void)fail(l, "TLD handle address");
            mov_cbuf(l, S(l, 4), &h);
        }
        int next = 5;
        if (level) mov_reg(l, S(l, next++), lev);
        if (offsets) mov_reg(l, S(l, next++), offs);
        m0 = S(l, 4);
    }
    uint32_t components = (uint32_t)__builtin_popcount(mask);
    int split = rd2 != RZ;
    int direct = (components <= 2 || (split ? rd2 == rd + 2 : 1)) && rd != RZ;
    uint32_t target = direct ? rd : S(l, 0);
    uint32_t mdim = array ? 3 : volume ? 4 : dim == 1 ? 2 : 0;
    emit(l, (bound ? 0xdc00000000000000ull : 0xdd00000000000000ull) | l->pred | target | ((uint64_t)c0 << 8) | ((uint64_t)m0 << 20) |
                ((uint64_t)mdim << 28) | ((uint64_t)mask << 31) | ((uint64_t)(bound ? tid : 0) << 36) | ((uint64_t)bits(x, 90, 1) << 49) |
                ((uint64_t)7 << 51) | ((uint64_t)level << 55) | ((uint64_t)offsets << 35));
    if (direct) return;
    for (uint32_t i = 0; i < components; i++) {
        uint32_t t = (!split || i < 2) ? (rd == RZ ? RZ : rd + i) : rd2 + (i - 2);
        if (t != RZ) mov_reg(l, t, S(l, (int)i));
    }
}

// IMAD.HI (0x027 forms): Rd = high word of a * b + c; signed by 73.
static void lower_imad_hi(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8);
    int is_signed = bits(x, 73, 1);
    Src a = src0(x, 0), b = srcn(l, x, 1, 0), c = srcn(l, x, 2, 0);
    if (bits(x, 72, 1) || bits(x, 74, 7) || bits(x, 81, 3) != 7 || bits(x, 84, 3) || bits(x, 87, 4) != 15)
        return (void)fail(l, "IMAD.HI mode");
    uint32_t rb = to_reg(l, b, 1, 0, 0), rc = to_reg(l, c, 2, 0, 0);
    uint64_t sign = is_signed ? ((uint64_t)1 << 48) | ((uint64_t)1 << 53) : 0;
    emit(l, base(l, 0x5a00000000000000ull, rd, a.reg) | ((uint64_t)rb << 20) | ((uint64_t)rc << 39) | sign | ((uint64_t)1 << 54));
}

// IABS (0x013): max(x, -x), signed.
// F2IP.U8.F32 (0x243, register form; tools/sm86_dis.py names it FCHK): two floats become bytes, Rb's in bits 0-7 and
// Ra's (RZ: none) in bits 8-15 of Rd, each clamped to 0..255 with NaN as 0 and converted with the rounding mode at
// 78-79 (the game's output packing after a multiplication by 255). Operand and mode fields as Lichii's Ampere front
// end reads them (GPL, read for the field meanings only): w1 holds only Rb, w2 is 0x4ff plus the rounding in 78-79.
// The rounding numbering is taken to be Maxwell F2I's (39-40), which no game data has settled. Maxwell: FMNMX against
// 0.0 (max) and 255.0 (min) as the clamp, F2I.U32 with that rounding, ISCADD to merge the bytes.
static void lower_f2ip(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8), ra = bits(x, 24, 8), rb = bits(x, 32, 8), round = bits(x, 78, 2);
    if ((x->w[1] & ~0xffu) || (x->w[2] & ~0xc000u) != 0x4ffu || rd == RZ) return (void)fail(l, "F2IP mode");
    uint32_t lo = S(l, 0), hi = S(l, 1), top = S(l, 2);
    mov_imm(l, top, 0x437f0000u);
    for (int i = 0; i < 2; i++) {
        uint32_t src = i ? ra : rb, t = i ? hi : lo;
        if (i && ra == RZ) break;
        emit(l, base(l, 0x5c60000000000000ull, t, src) | ((uint64_t)RZ << 20) | ((uint64_t)15 << 39));            // max(src, 0)
        emit(l, base(l, 0x5c60000000000000ull, t, t) | ((uint64_t)top << 20) | ((uint64_t)7 << 39));             // min(t, 255)
        emit(l, base(l, 0x5cb0000000000000ull, t, 0) | ((uint64_t)2 << 8) | ((uint64_t)2 << 10) | ((uint64_t)t << 20) |
                    ((uint64_t)round << 39));                                                                       // F2I.U32
    }
    if (ra == RZ) mov_reg(l, rd, lo);
    else emit(l, base(l, 0x5c18000000000000ull, rd, hi) | ((uint64_t)lo << 20) | ((uint64_t)8 << 39));              // (hi << 8) + lo
}

static void lower_iabs(L *l, const Insn *x) {
    uint32_t rd = bits(x, 16, 8);
    if (x->w[2] || bits(x, 24, 8)) return (void)fail(l, "IABS mode");
    Src b = srcn(l, x, 1, 0);
    uint32_t rb = to_reg(l, b, 1, 0, 0), t = S(l, 2);
    emit(l, base(l, 0x5c10000000000000ull, t, RZ) | ((uint64_t)rb << 20) | ((uint64_t)1 << 48));
    emit(l, base(l, 0x5c20000000000000ull, rd, rb) | ((uint64_t)t << 20) | ((uint64_t)7 << 39) | ((uint64_t)1 << 42) | ((uint64_t)1 << 48));
}

// ---- control flow

static int64_t branch_target(const L *l, const Insn *x, int bra) {
    uint64_t raw = (bra ? (x->w[1] & ~3u) : x->w[1]) | ((uint64_t)(x->w[2] & 0x3ffff) << 32);
    return (int64_t)l->index * 16 + 16 + sext(raw, 50);
}

static void lower_bra(L *l, const Insn *x) {
    uint32_t mode = bits(x, 32, 2), pc = bits(x, 87, 4);
    if ((mode <= 1 && bits(x, 91, 1)) || (mode == 1 && (bits(x, 82, 5) || bits(x, 92, 13)))) return (void)fail(l, "BRA mode");
    if (mode == 2) return;                                    // .DIV: taken only on a diverged warp; never here
    int64_t t = branch_target(l, x, 1);
    if (t < 0 || t % 16 || (uint64_t)t / 16 > l->n) return (void)fail(l, "BRA target");
    uint64_t p = l->pred;
    if (pc != 7) {
        // @Pg BRA Pc: both must hold. Skip the branch unless Pg, then branch on Pc.
        if (bits(x, 12, 4) != 7) { reloc(l, l->index + 1); emit(l, 0xe24000000000000full | ((uint64_t)(bits(x, 12, 4) ^ 8) << 16)); }
        p = (uint64_t)pc << 16;
    }
    if ((uint64_t)t / 16 == l->index && p == ((uint64_t)7 << 16)) {
        // The compiler's sentinel after the last EXIT, an unconditional branch to itself: an end here (the
        // translator's decoder rejects the self-loop when it follows a split block).
        emit(l, 0xe30000000007000full);
        return;
    }
    reloc(l, (uint32_t)(t / 16));
    emit(l, 0xe24000000000000full | p);
}

static void lower_exit(L *l, const Insn *x) {
    uint32_t pc = bits(x, 87, 4);
    if (bits(x, 84, 2)) return (void)fail(l, "EXIT mode");
    if (pc != 7) {
        if (bits(x, 12, 4) != 7) { reloc(l, l->index + 1); emit(l, 0xe24000000000000full | ((uint64_t)(bits(x, 12, 4) ^ 8) << 16)); }
        emit(l, 0xe30000000000000full | ((uint64_t)pc << 16));
        return;
    }
    emit(l, 0xe30000000000000full | l->pred);
}

// CALL.REL.NOINC pushes no return address: the compiler uses it predicated to skip the next instruction (a
// loop's back edge) and, with a link register set by MOV before it, as a real call whose callee returns with
// RET.REL.NODEC. The first is a branch; the second becomes Maxwell CAL / RET, which keep their own stack, and the
// link register is left as the dead value it is.
static void lower_call(L *l, const Insn *x) {
    if (bits(x, 16, 16) || (x->w[2] & ~0x3ffffu) != 0x03c00000u) return (void)fail(l, "CALL mode");
    int64_t t = branch_target(l, x, 0);
    if (t < 0 || t % 16 || (uint64_t)t / 16 > l->n) return (void)fail(l, "CALL target");
    if (t == (int64_t)(l->index + 1) * 16 + 16 || bits(x, 12, 4) != 7) {
        reloc(l, (uint32_t)(t / 16));
        emit(l, 0xe24000000000000full | l->pred);
        return;
    }
    reloc(l, (uint32_t)(t / 16));
    emit(l, 0xe26000000007000full);
}

static void lower_ret(L *l, const Insn *x) {
    uint64_t raw = x->w[1] | ((uint64_t)(x->w[2] & 0x3ffff) << 32);
    if ((x->w[0] & 0x00ffffffu) != 0x00007950u || (x->w[2] & ~0x3ffffu) != 0x03c00000u ||
        (int64_t)l->index * 16 + 16 + sext(raw, 50) != 0)
        return (void)fail(l, "RET mode");
    emit(l, 0xe32000000007000full);
}

// BRX: the table holds absolute native offsets and the immediate is minus the next instruction's, so the target
// is the table value. Maxwell BRX adds the register to its own address + 8 + imm24: the immediate is set to
// minus that, so the (remapped) table value is the target again.
static void lower_brx(L *l, const Insn *x, uint32_t op) {
    uint64_t raw = x->w[1] | ((uint64_t)(x->w[2] & 0x3ffff) << 32);
    if (bits(x, 12, 4) != 7 || (x->w[2] & ~0x3ffffu) != (op == 0x958 ? 0x0b800000u : 0x03800000u))
        return (void)fail(l, "BRX mode");
    if ((int64_t)l->index * 16 + 16 + sext(raw, 50) != 0) return (void)fail(l, "BRX relative base");
    uint32_t src = op == 0x958 ? ur(l, bits(x, 24, 8)) : bits(x, 24, 8);
    int64_t offset = -(int64_t)addr_of(l, l->nout) - 8;
    if (offset < -0x800000) return (void)fail(l, "BRX range");
    emit(l, 0xe25000000000000full | l->pred | ((uint64_t)src << 8) | (((uint64_t)offset & 0xffffff) << 20));
}

static void lower_one_inner(L *l, const Insn *x);

static void lower_one(L *l, const Insn *x) {
    uint32_t p = bits(x, 12, 4);
    if (p == 15) return;                                      // @!PT
    l->pred = (uint64_t)p << 16;
    uint32_t swap = l->up_swap ? up_fields(x) : 0;
    if (swap) {
        // Save P(mask), load the uniform predicates into them, run the instruction, merge them back, restore.
        // PLOP3 is the one instruction here with ordinary predicate outputs (and inputs): an output in a swapped
        // slot is the instruction's result, kept out of the merge and the restore (2026-11-06: the upscaler's
        // first layer ends its tile loop on PLOP3 P0 = UP0; restoring P0 made the loop run until its counters
        // wrapped, 2.2 s per dispatch and a GPU timeout in play); an input in one would read the uniform value.
        uint32_t wr = 0, rd = 0, op = x->w[0] & 0xfff, temps = 0;
        uint8_t copy_from[8] = {0};
        Insn y = *x;                                          // the instruction with its conflicting inputs redirected
        if (op == 0x81c) {
            uint32_t pd = bits(x, 81, 3), pd2 = bits(x, 84, 3), a = bits(x, 87, 3), b = bits(x, 77, 3);
            if (pd != 7) wr |= 1u << pd;
            if (pd2 != 7) wr |= 1u << pd2;
            if (a != 7) rd |= 1u << a;
            if (b != 7) rd |= 1u << b;
            if (!bits(x, 67, 1) && bits(x, 68, 3) != 7) rd |= 1u << bits(x, 68, 3);
            if (rd & swap) {
                // A real input names the predicate a uniform one is swapped into (P0 = P0 & UP0 is the usual form,
                // 2026-11-06: 10 compute kernels of the Great Deku Tree scene). Each such input is copied into a
                // predicate the instruction does not name before the swap, and read from the copy; the copy's
                // predicate is saved with the swapped ones and restored after.
                uint32_t named = swap | wr | rd | (bits(x, 12, 3) != 7 ? 1u << bits(x, 12, 3) : 0), conflict = rd & swap;
                for (uint32_t n = 0; n < 7; n++) {
                    if (!(conflict >> n & 1)) continue;
                    uint32_t k = 0;
                    while (k < 7 && (named >> k & 1)) k++;
                    if (k == 7) return (void)fail(l, "PLOP3 reads the predicate a uniform one is swapped into");
                    named |= 1u << k, temps |= 1u << k;
                    if (bits(x, 87, 3) == n) y.w[2] = (y.w[2] & ~(7u << 23)) | k << 23;
                    if (bits(x, 77, 3) == n) y.w[2] = (y.w[2] & ~(7u << 13)) | k << 13;
                    if (!bits(x, 67, 1) && bits(x, 68, 3) == n) y.w[2] = (y.w[2] & ~(7u << 4)) | k << 4;
                    copy_from[k] = (uint8_t)n;
                }
                x = &y;
            }
        }
        uint32_t keep = swap & ~wr;
        if (!l->up_reg_set) {
            l->up_reg = l->liveness && l->ur_next >= l->scratch_base ? alloc_dead(l, 0, l->n - 1) : alloc_gpr(l);
            if (l->up_reg == NOREG) return (void)fail(l, "no general register free for the uniform predicates");
            l->up_reg_set = 1;
        }
        uint32_t save = S(l, 7);
        emit(l, 0x38e8000000070000ull | save | ((uint64_t)RZ << 8) | ((uint64_t)(swap | temps) << 20));
        for (uint32_t k = 0; k < 7; k++)                      // PSETP Pk = Pn & PT & PT: the real input before the swap
            if (temps >> k & 1)
                emit(l, 0x5090000000000000ull | 7 | ((uint64_t)k << 3) | ((uint64_t)PT << 12) | ((uint64_t)PT << 29) |
                            ((uint64_t)copy_from[k] << 39) | ((uint64_t)7 << 16));
        emit(l, 0x38f0000000070000ull | ((uint64_t)l->up_reg << 8) | ((uint64_t)swap << 20));
        lower_one_inner(l, x);
        if (keep) emit(l, 0x38e8000000070000ull | l->up_reg | ((uint64_t)l->up_reg << 8) | ((uint64_t)keep << 20));
        if (keep | temps) emit(l, 0x38f0000000070000ull | ((uint64_t)save << 8) | ((uint64_t)(keep | temps) << 20));
        return;
    }
    lower_one_inner(l, x);
}

static void lower_one_inner(L *l, const Insn *x) {
    uint32_t op = x->w[0] & 0xfff, base_op = op & 0x1ff;
    Insn stripped;
    if ((op >> 9 == 6 || op >> 9 == 7) && base_op < 0x120 && bits(x, 91, 1)) {
        stripped = *x, stripped.w[2] &= ~(1u << 27), x = &stripped;   // bit 91 marks the uniform operand
    }
    switch (op) {
    case 0x947: return lower_bra(l, x);
    case 0x94d: return lower_exit(l, x);
    case 0x918: case 0x91a: case 0x945: case 0x941: case 0x942: case 0x948: return;   // NOP, DEPBAR, BSSY, BSYNC, BREAK, WARPSYNC
    case 0x95b:
        if (x->w[1] || x->w[2] != 0x03800000u) return (void)fail(l, "KILL mode");
        return emit(l, 0xe33000000000000full | l->pred);
    case 0x944: return lower_call(l, x);
    case 0x950: return lower_ret(l, x);
    case 0x949: case 0x958: return lower_brx(l, x, op);
    case 0x202: case 0x802: case 0xa02: case 0xc02: return lower_mov(l, x, op);
    case 0x220: case 0x820: case 0xa20: case 0xc20: return lower_fmul(l, x);
    case 0x221: case 0x421: case 0x621: case 0xe21: return lower_fadd(l, x);
    case 0x223: case 0x423: case 0x623: case 0x823: case 0xa23: case 0xc23: case 0xe23: return lower_ffma(l, x);
    case 0x209: case 0x809: case 0xa09: case 0xc09: return lower_fmnmx(l, x);
    case 0x20b: case 0x80b: case 0xa0b: case 0xc0b: return lower_fsetp(l, x);
    case 0x20a: case 0x80a: case 0xa0a: case 0xc0a: return lower_fset(l, x);
    case 0x20c: case 0x80c: case 0xa0c: case 0xc0c: return lower_isetp(l, x);
    case 0x207: case 0x807: case 0xa07: case 0xc07: return lower_sel(l, x, 0);
    case 0x208: case 0x808: case 0xa08: case 0xc08: return lower_sel(l, x, 1);
    case 0x210: case 0x810: case 0xa10: case 0xc10: return lower_iadd3(l, x, op);
    case 0x211: case 0x811: case 0xa11: case 0xc11: return lower_lea(l, x, op);
    case 0x212: case 0x812: case 0xa12: case 0xc12: return lower_lop3(l, x);
    case 0x217: case 0x817: case 0xa17: case 0xc17: return lower_imnmx(l, x);
    case 0x219: case 0x419: case 0x819: case 0xa19: case 0xc19: return lower_shf(l, x, op);
    case 0x81a: return lower_sgxt(l, x);
    case 0x816: case 0x216: case 0xa16: return lower_prmt(l, x);
    case 0x224: case 0x424: case 0x624: case 0x824: case 0xa24: case 0xc24: case 0xe24: return lower_imad(l, x, op);
    case 0x225: case 0x825: return lower_imad_wide(l, x);
    case 0x245: case 0xa45: case 0xc45: return lower_i2fp(l, x);
    case 0x305: case 0xb05: case 0xd05: return lower_f2i(l, x);
    case 0x306: case 0xd06: return lower_i2f(l, x);
    case 0x307: case 0xb07: return lower_frnd(l, x);
    case 0x308: case 0xb08: case 0xd08: return lower_mufu(l, x);
    case 0x300: case 0xd00: return lower_popc_flo(l, x, 1);
    case 0x309: case 0xd09: return lower_popc_flo(l, x, 0);
    case 0x301: return lower_brev(l, x);
    case 0x230: case 0x430: case 0x630: return lower_hadd2(l, x, op);
    case 0x326: case 0x526: return lower_ipa(l, x);
    case 0x321: return lower_ald_ast(l, x, 0);
    case 0x322: return lower_ald_ast(l, x, 1);
    case 0xb82: return lower_ldc(l, x);
    case 0xab9: case 0xabb: return lower_uldc(l, x, op);
    case 0x919: case 0x9c3: return lower_s2r(l, x, op);
    case 0x805: return lower_cs2r(l, x);
    case 0x381: case 0x981: return lower_ldg(l, x, op);
    case 0x386: case 0x986: return lower_stg(l, x, op);
    case 0xb60: case 0xf60: case 0x361: return lower_tex(l, x, op);
    case 0xb9b: return lower_sust(l, x);
    case 0xb97: return lower_suld(l, x);
    case 0x59f: return lower_sured(l, x);
    case 0x231: case 0x431: case 0x631: case 0x831: case 0xa31: return lower_hfma2(l, x, op);
    case 0x23e: return lower_f2fp(l, x, op);
    case 0x234: case 0x434: return lower_hsetp2(l, x, op);
    case 0x232: case 0x432: case 0x832: case 0xa32: return lower_hmul2(l, x, op);
    case 0x240: return lower_hmnmx2(l, x, op);
    case 0xb1d: return lower_bar(l, x);
    case 0x83b: return lower_ldsm(l, x);
    case 0x23c: return lower_hmma(l, x);
    case 0x23a: return lower_movm(l, x);
    case 0xb63: case 0xf63: return lower_tld4(l, x, op);
    case 0xb66: case 0xf66: return lower_tld(l, x, op);
    case 0xb6f: case 0x370: return lower_txq(l, x, op);
    case 0xb69: case 0xf69: case 0x36a: return lower_tmml(l, x, op);
    case 0xb6c: return lower_txd(l, x);
    case 0x983: return lower_ldl_stl(l, x, 0);
    case 0x387: case 0x987: return lower_ldl_stl(l, x, 1);
    case 0x984: case 0x388: case 0x988: return lower_lds_sts(l, x, op);
    case 0x98e: return lower_red(l, x);
    case 0x3a8: case 0x9a8: return lower_atomg(l, x, op);
    case 0x227: case 0x827: case 0xa27: return lower_imad_hi(l, x);
    case 0x213: case 0x813: case 0xa13: return lower_iabs(l, x);
    case 0x243: return lower_f2ip(l, x);
    case 0x803: return lower_p2r_r2p(l, x, 0);
    case 0x804: return lower_p2r_r2p(l, x, 1);
    case 0x81c: return lower_plop3(l, x);
    case 0xf89: case 0x589: case 0x989: return lower_shfl(l, x, op);
    case 0x806: return lower_vote(l, x);
    case 0x822: return lower_fswzadd(l, x);
    case 0x3c2: return lower_r2ur(l, x);
    case 0x3c4: return lower_redux(l, x);
    default:
        if (base_op >= 0x080 && base_op < 0x100) return lower_uniform_alu(l, x, op);
        return (void)fail(l, "opcode");
    }
}

// ---- the program

void sm86_lower_free(Sm86Lowered *out) {
    free(out->code), free(out->c1);
    out->code = NULL, out->c1 = NULL;
}

int sm86_lower(const uint8_t *code, uint32_t code_size, const uint8_t *control, uint32_t control_size, Sm86Lowered *out) {
    return sm86_lower_ex(code, code_size, control, control_size, out, sm86_lower_tensor_cores);
}

static int lower_attempt(const uint8_t *code, uint32_t code_size, const uint8_t *control, uint32_t control_size,
                         Sm86Lowered *out, int tensor, int force_swap) {
    memset(out, 0, sizeof *out);
    Sm86Control c;
    if (!sm86_control_parse(control, control_size, &c) || c.code_size > code_size) {
        snprintf(out->error, sizeof out->error, "control section");
        return 0;
    }
    L l = {0};
    l.force_swap = force_swap;
    l.stage = c.stage;
    l.graphics = c.stage != 5;
    l.tensor = tensor && !l.graphics && c.local[0] * c.local[1] * c.local[2] % 32 == 0;
    l.control = control, l.ctl_size = c.size, l.c1 = code + c.c1_off, l.c1_len = c.c1_len;
    uint32_t header = l.graphics ? SM86_HEADER : 0;
    l.in = (const Insn *)(code + header);
    l.n = (c.code_end - header) / 16;
    if (l.n == 0 || l.n > 65536) { snprintf(out->error, sizeof out->error, "program size"); return 0; }
    l.first_out = calloc(l.n + 1, sizeof *l.first_out);
    if (!l.first_out) { snprintf(out->error, sizeof out->error, "out of memory"); return 0; }
    scan_registers(&l);
    int blocks = build_blocks(&l);
    if (!blocks) free(l.leader), l.leader = NULL;              // a jump table without its record: no local forwarding
    scan_uniform(&l);
    if (l.scratch_base + NSCRATCH > 255 && blocks && !l.failed) {
        // No room above the program's registers: scratch comes from the registers dead at each instruction.
        l.liveness = compute_liveness(&l);
        if (!l.liveness) free(l.live_out), l.live_out = NULL;
    }
    for (l.index = 0; l.index < l.n && !l.failed; l.index++) {
        l.first_out[l.index] = l.nout;
        if (l.leader && l.leader[l.index]) memset(l.cur_def, 0, sizeof l.cur_def);
        if (l.liveness) scratch_at(&l, l.index);
        lower_one(&l, &l.in[l.index]);
        int pair;
        uint8_t kind, bank;
        uint32_t value;
        uint32_t w = ur_write(&l.in[l.index], &pair, &kind, &bank, &value);
        if (w < 63) {
            l.cur_def[w].kind = kind, l.cur_def[w].bank = bank, l.cur_def[w].value = value;
            if (pair && w + 1 < 63) l.cur_def[w + 1].kind = kind == 1 ? 1 : 3, l.cur_def[w + 1].bank = bank, l.cur_def[w + 1].value = value + 4;
        }
    }
    l.first_out[l.n] = l.nout;
    if (!l.failed) emit(&l, 0xe30000000007000full);          // a branch past the end lands on an EXIT
    // Relocations.
    for (uint32_t i = 0; i < l.nreloc && !l.failed; i++) {
        uint32_t o = l.reloc[i].out, t = l.first_out[l.reloc[i].target];
        int64_t rel = (int64_t)addr_of(&l, t) - (int64_t)addr_of(&l, o) - 8;
        if (rel < -0x800000 || rel > 0x7fffff) { l.index = l.reloc[i].target; fail(&l, "branch distance"); break; }
        l.out[o] |= ((uint64_t)rel & 0xffffff) << 20;
    }
    if (!l.failed) {
        // The jump tables: every record names a BRX (byte offset in the code, header included), its entry count
        // and the table's byte offset in c1; each entry is a native instruction offset to move.
        out->c1_len = c.c1_len;
        out->c1 = malloc(c.c1_len ? c.c1_len : 1);
        if (!out->c1) fail(&l, "out of memory");
        else {
            memcpy(out->c1, code + c.c1_off, c.c1_len);
            uint32_t count = 0;
            if (c.size >= SM86_CONTROL_MIN + 4) memcpy(&count, control + SM86_CONTROL_MIN, 4);
            if (count > 64 || SM86_CONTROL_MIN + 4 + count * 16 > c.size) count = 0;
            for (uint32_t r = 0; r < count && !l.failed; r++) {
                uint32_t rec[4];
                memcpy(rec, control + SM86_CONTROL_MIN + 4 + r * 16, 16);
                uint32_t entries = rec[1], at = rec[3];
                if (at + entries * 4 > c.c1_len) { fail(&l, "jump-table record"); break; }
                for (uint32_t e = 0; e < entries; e++) {
                    uint32_t v;
                    memcpy(&v, out->c1 + at + e * 4, 4);
                    if (v % 16 || v / 16 > l.n) { fail(&l, "jump-table entry"); break; }
                    v = addr_of(&l, l.first_out[v / 16]);
                    memcpy(out->c1 + at + e * 4, &v, 4);
                }
            }
            if (count == 0 && c.c1_len) {
                // Without records (not seen in the archives), move every word that could be an entry.
                for (uint32_t e = 0; e * 4 + 4 <= c.c1_len; e++) {
                    uint32_t v;
                    memcpy(&v, out->c1 + e * 4, 4);
                    if (v % 16 == 0 && v / 16 < l.n) v = addr_of(&l, l.first_out[v / 16]), memcpy(out->c1 + e * 4, &v, 4);
                }
            }
        }
    }
    if (!l.failed) {
        // Serialise: header, then the words with a zero scheduling word before every group of three.
        uint32_t size = addr_of(&l, l.nout);
        out->code = calloc(size + 32, 1);
        if (!out->code) fail(&l, "out of memory");
        else {
            if (l.graphics) memcpy(out->code, code, 0x50);
            for (uint32_t k = 0; k < l.nout; k++) memcpy(out->code + addr_of(&l, k), &l.out[k], 8);
            out->code_size = size;
        }
    }
    if (l.failed) {
        snprintf(out->error, sizeof out->error, "%s", l.error);
        out->error_offset = header + l.error_index * 16;
        sm86_lower_free(out);
    }
    free(l.out), free(l.first_out), free(l.reloc), free(l.leader), free(l.live_out);
    return !l.failed;
}

int sm86_lower_ex(const uint8_t *code, uint32_t code_size, const uint8_t *control, uint32_t control_size, Sm86Lowered *out,
                  int tensor) {
    if (lower_attempt(code, code_size, control, control_size, out, tensor, 0)) return 1;
    if (strcmp(out->error, "no predicate free for a uniform predicate")) return 0;
    return lower_attempt(code, code_size, control, control_size, out, tensor, 1);
}
