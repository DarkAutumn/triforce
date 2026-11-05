/*
 * tforacle: a minimal libretro frontend around FCEUmm used as the emulation
 * oracle for triforce. Development tooling only: FCEUmm is GPL, so this
 * library is loaded at runtime by tools (libloading / ctypes) and is never
 * linked into shipping crates.
 *
 * One oracle per process: FCEUmm keeps its state in globals.
 */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "libretro.h"
#include "fceu-types.h"
#include "x6502struct.h"

/* FCEUmm globals (exported because the core is built without a version script). */
extern X6502 X;
extern uint32_t timestamp;
extern uint64_t timestampbase;
extern int scanline;
extern uint8_t RAM[0x800];
extern uint8_t NTARAM[0x800], PALRAM[0x20], SPRAM[0x100];
extern uint8_t UPALRAM[0x03];
extern uint8_t *XBuf;
extern uint8_t *Page[32], *VPage[8];
extern void (*tfo_instr_hook)(uint16_t pc, uint8_t opcode);

typedef struct {
    uint16_t pc;
    uint8_t opcode, a, x, y, p, sp;
    uint64_t cycle;
    int16_t scanline;
} tfo_trace_row;

typedef void (*tfo_trace_cb)(const tfo_trace_row *, void *);

static uint8_t g_buttons;
static int g_open;
static char g_sysdir[256];
static tfo_trace_cb g_trace_cb;
static void *g_trace_user;

static void log_noop(enum retro_log_level level, const char *fmt, ...) {
    (void)level;
    (void)fmt;
}

static bool env_cb(unsigned cmd, void *data) {
    switch (cmd) {
    case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
        return true;
    case RETRO_ENVIRONMENT_GET_VARIABLE: {
        struct retro_variable *var = (struct retro_variable *)data;
        if (!var || !var->key)
            return false;
        if (!strcmp(var->key, "fceumm_region")) {
            var->value = "NTSC";
            return true;
        }
        if (!strcmp(var->key, "fceumm_overscan_h_left") || !strcmp(var->key, "fceumm_overscan_h_right") ||
            !strcmp(var->key, "fceumm_overscan_v_top") || !strcmp(var->key, "fceumm_overscan_v_bottom")) {
            var->value = "0";
            return true;
        }
        /* Movies store the exact controller byte; let opposing directions through. */
        if (!strcmp(var->key, "fceumm_up_down_allowed")) {
            var->value = "enabled";
            return true;
        }
        return false;
    }
    case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
    case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
        *(const char **)data = g_sysdir;
        return true;
    case RETRO_ENVIRONMENT_GET_LOG_INTERFACE:
        ((struct retro_log_callback *)data)->log = log_noop;
        return true;
    default:
        return false;
    }
}

static void video_cb(const void *d, unsigned w, unsigned h, size_t pitch) {
    (void)d; (void)w; (void)h; (void)pitch;
}
static void audio_cb(int16_t l, int16_t r) { (void)l; (void)r; }
static size_t audio_batch_cb(const int16_t *d, size_t frames) { (void)d; return frames; }
static void poll_cb(void) {}

static int16_t input_cb(unsigned port, unsigned device, unsigned index, unsigned id) {
    (void)index;
    if (port != 0 || device != RETRO_DEVICE_JOYPAD)
        return 0;
    switch (id) {
    case RETRO_DEVICE_ID_JOYPAD_A: return (g_buttons & 0x01) != 0;
    case RETRO_DEVICE_ID_JOYPAD_B: return (g_buttons & 0x02) != 0;
    case RETRO_DEVICE_ID_JOYPAD_SELECT: return (g_buttons & 0x04) != 0;
    case RETRO_DEVICE_ID_JOYPAD_START: return (g_buttons & 0x08) != 0;
    case RETRO_DEVICE_ID_JOYPAD_UP: return (g_buttons & 0x10) != 0;
    case RETRO_DEVICE_ID_JOYPAD_DOWN: return (g_buttons & 0x20) != 0;
    case RETRO_DEVICE_ID_JOYPAD_LEFT: return (g_buttons & 0x40) != 0;
    case RETRO_DEVICE_ID_JOYPAD_RIGHT: return (g_buttons & 0x80) != 0;
    default: return 0;
    }
}

static void instr_hook(uint16_t pc, uint8_t opcode) {
    tfo_trace_row row;
    row.pc = pc;
    row.opcode = opcode;
    row.a = X.A;
    row.x = X.X;
    row.y = X.Y;
    row.p = X.P;
    row.sp = X.S;
    row.cycle = timestampbase + timestamp; /* CPU cycles since power-on at instruction start */
    row.scanline = (int16_t)scanline;
    g_trace_cb(&row, g_trace_user);
}

static uint8_t *g_rom;
static size_t g_rom_len;

int tfo_open(const char *rom_path) {
    FILE *f;
    long len;
    struct retro_game_info info;
    uint8_t *sram;

    if (g_open)
        return -1;
    f = fopen(rom_path, "rb");
    if (!f)
        return -2;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    g_rom = (uint8_t *)malloc((size_t)len);
    g_rom_len = (size_t)len;
    if (fread(g_rom, 1, g_rom_len, f) != g_rom_len) {
        fclose(f);
        return -3;
    }
    fclose(f);

    snprintf(g_sysdir, sizeof g_sysdir, "/tmp/tforacle-%d", (int)getpid());

    retro_set_environment(env_cb);
    retro_set_video_refresh(video_cb);
    retro_set_audio_sample(audio_cb);
    retro_set_audio_sample_batch(audio_batch_cb);
    retro_set_input_poll(poll_cb);
    retro_set_input_state(input_cb);
    retro_init();

    memset(&info, 0, sizeof info);
    info.path = rom_path;
    info.data = g_rom;
    info.size = g_rom_len;
    if (!retro_load_game(&info))
        return -4;
    retro_set_controller_port_device(0, RETRO_DEVICE_JOYPAD);

    sram = (uint8_t *)retro_get_memory_data(RETRO_MEMORY_SAVE_RAM);
    if (!sram || retro_get_memory_size(RETRO_MEMORY_SAVE_RAM) != 0x2000)
        return -5;
    memset(sram, 0, 0x2000);
    g_buttons = 0;
    g_open = 1;
    return 0;
}

void tfo_close(void) {
    if (!g_open)
        return;
    retro_unload_game();
    retro_deinit();
    free(g_rom);
    g_rom = NULL;
    g_open = 0;
}

void tfo_run_frame(uint8_t nes_buttons) {
    g_buttons = nes_buttons;
    retro_run();
}

uint8_t *tfo_ram(void) { return RAM; }

uint8_t *tfo_prg_ram(void) { return (uint8_t *)retro_get_memory_data(RETRO_MEMORY_SAVE_RAM); }

void tfo_read_chr(uint8_t out[0x2000]) {
    unsigned a;
    for (a = 0; a < 0x2000; a++)
        out[a] = VPage[a >> 10][a];
}

void tfo_read_ciram(uint8_t out[0x800]) { memcpy(out, NTARAM, 0x800); }

void tfo_read_palette(uint8_t out[32]) {
    unsigned i;
    for (i = 0; i < 32; i++) {
        if (!(i & 3))
            out[i] = (i & 0xC) ? UPALRAM[((i & 0xC) >> 2) - 1] : PALRAM[0];
        else
            out[i] = PALRAM[i];
    }
}

void tfo_read_oam(uint8_t out[256]) { memcpy(out, SPRAM, 256); }

void tfo_read_pixels(uint8_t out[256 * 240]) {
    unsigned i;
    for (i = 0; i < 256 * 240; i++)
        out[i] = XBuf[i] & 0x3F;
}

size_t tfo_state_size(void) { return retro_serialize_size(); }

int tfo_save_state(uint8_t *buf, size_t n) { return retro_serialize(buf, n) ? 0 : -1; }

int tfo_load_state(const uint8_t *buf, size_t n) { return retro_unserialize(buf, n) ? 0 : -1; }

void tfo_set_trace(tfo_trace_cb cb, void *user) {
    g_trace_cb = cb;
    g_trace_user = user;
    tfo_instr_hook = cb ? instr_hook : NULL;
}

/* CPU run budget left after the last frame (FCEUmm `X.count`, 1/16 dot units; <= 0 after a frame). */
int32_t tfo_count(void) { return X.count; }
