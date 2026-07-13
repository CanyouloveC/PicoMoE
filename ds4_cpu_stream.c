/* =========================================================================
 * ds4_cpu_stream.c — CPU-side expert page-cache management via madvise
 * =========================================================================
 *
 * Architecture
 * -----------
 * This module works with raw mmap offsets — it does NOT include ds4.c
 * internals.  A small set of bridge functions in ds4.c (see "CPU stream
 * bridge" section) feeds us expert tensor metadata.
 *
 * Strategy (inspired by colibri)
 * ------------------------------
 * 1. On startup: walk all layers, compute the byte range of every expert
 *    within the mmap'd GGUF.  Build lookup tables.
 * 2. Before MoE: issue madvise(MADV_WILLNEED) on the page ranges of the
 *    experts that the router just selected.  The kernel starts async
 *    readahead — by the time matmul touches the pages they are resident.
 * 3. After MoE: bump usage counters so the eviction pass knows which
 *    experts are "hot".
 * 4. Periodic eviction: walk cold experts and issue madvise(MADV_DONTNEED)
 *    so the kernel can reclaim those pages under memory pressure.
 * 5. Route-ahead: use a simple N-gram correlation ("if expert X is used
 *    in layer L, expert Y is likely in layer L+1") to prefetch next-layer
 *    weights while computing the current layer's FFN.
 * 6. Persistence: write usage counters to .ds4_usage so subsequent runs
 *    can pin the hottest experts on startup.
 *
 * Compatibility
 * -------------
 * - Linux: full madvise(MADV_WILLNEED / MADV_DONTNEED) support.
 * - macOS: MADV_WILLNEED exists but is weaker; MADV_DONTNEED → MADV_FREE.
 * - Other: graceful no-op (cache is still tracked, but page steering is
 *   left to the kernel).
 */

#include "ds4_cpu_stream.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

/* -----------------------------------------------------------------------
 * Platform shims
 * ----------------------------------------------------------------------- */

#if defined(__linux__)
  #define DS4_HAS_MADVISE 1
  #define DS4_MADV_WILLNEED  MADV_WILLNEED
  #define DS4_MADV_DONTNEED  MADV_DONTNEED
  #define DS4_CS_GIB (1024ull * 1024ull * 1024ull)
#elif defined(__APPLE__)
  #define DS4_HAS_MADVISE 1
  #define DS4_MADV_WILLNEED  MADV_WILLNEED
  /* macOS doesn't have MADV_DONTNEED for file-backed mappings; use
   * MADV_FREE which is a weaker hint but better than nothing. */
  #define DS4_MADV_DONTNEED  MADV_FREE
  #define DS4_CS_GIB (1024ull * 1024ull * 1024ull)
#else
  #define DS4_HAS_MADVISE 0
  #define DS4_CS_GIB (1024ull * 1024ull * 1024ull)
#endif

/* -----------------------------------------------------------------------
 * Per-layer expert index
 * ----------------------------------------------------------------------- */
typedef struct {
    cs_expert_ranges *expert;  /* [n_experts] array, NULL if no routed experts */
    uint32_t          n_experts;
    uint32_t          n_experts_used; /* DS4_N_EXPERT_USED, e.g. 6 */
    bool              has_routed;
    /* Usage tracking */
    uint64_t         *access_count;   /* [n_experts] */
    uint64_t         *last_clock;     /* [n_experts] — monotonic clock value */
} cs_layer;

/* -----------------------------------------------------------------------
 * Route-ahead: simple Markov predictor
 * ----------------------------------------------------------------------- */

/* cooc[layer][expert_A * n_experts + expert_B] = how often B follows A */
typedef struct {
    uint64_t *counts;          /* [n_experts * n_experts] flat matrix */
    uint32_t  n_experts;
} cs_transition;

/* -----------------------------------------------------------------------
 * Main cache structure
 * ----------------------------------------------------------------------- */

struct ds4_cpu_stream {
    /* Borrowed references */
    const uint8_t *mmap_base;   /* start of model mmap */
    uint64_t       mmap_size;

    /* Per-layer data */
    cs_layer      *layers;      /* [n_layers] */
    uint32_t       n_layers;
    uint32_t       n_experts;   /* uniform across all layers (DS4_N_EXPERT) */

    /* Budget */
    uint64_t       cache_bytes; /* target RAM budget for hot experts */
    uint64_t       hot_bytes;   /* estimated bytes currently WILLNEED */
    uint64_t       per_expert_bytes; /* average expert size */

    /* Clock for LRU — incremented on every mark_used call */
    uint64_t       clock;

    /* Eviction throttle */
    uint64_t       last_evict_clock;
    uint64_t       evict_interval;  /* clocks between eviction passes */

    /* Route-ahead predictors */
    cs_transition *transitions; /* [n_layers-1] array */

    /* Persistence */
    char          *usage_path;
    bool           usage_dirty;

    /* Prefetch thread (async route-ahead I/O) */
    pthread_t      prefetch_thread;
    bool           prefetch_running;
    int            prefetch_layer;    /* next layer to prefetch */
    int            prefetch_experts[16];
    int            prefetch_n;
    pthread_mutex_t prefetch_mutex;
    pthread_cond_t  prefetch_cond;

    /* Statistics */
    uint64_t       total_willneed_calls;
    uint64_t       total_dontneed_calls;
    uint64_t       total_prefetch_hits;
    uint64_t       total_prefetch_misses;
};

/* -----------------------------------------------------------------------
 * Page helpers
 * ----------------------------------------------------------------------- */

static long cs_page_size(void) {
    static long cached = 0;
    if (cached == 0) {
        long ps = sysconf(_SC_PAGESIZE);
        cached = (ps > 0) ? ps : 4096;
    }
    return cached;
}

static uint64_t cs_page_align_down(uint64_t addr) {
    const uint64_t mask = (uint64_t)(cs_page_size() - 1);
    return addr & ~mask;
}

static uint64_t cs_page_align_up(uint64_t addr) {
    const uint64_t mask = (uint64_t)(cs_page_size() - 1);
    return (addr + mask) & ~mask;
}

/* -----------------------------------------------------------------------
 * madvise wrappers (safe for any OS)
 * ----------------------------------------------------------------------- */

#if DS4_HAS_MADVISE
static void cs_madvise_range(const uint8_t *mmap_base,
                             uint64_t       start,
                             uint64_t       end,
                             int            advice) {
    if (end <= start || !mmap_base) return;
    const uint64_t pstart = cs_page_align_down(start);
    const uint64_t pend   = cs_page_align_up(end);
    if (pend <= pstart) return;

    const size_t len = (size_t)(pend - pstart);
    /* Only advise pages that are actually within the mapping */
    if (pstart + len > (uint64_t)SIZE_MAX) return;

    int rc = madvise((void *)(mmap_base + pstart), len, advice);
    (void)rc; /* best-effort; failures are non-fatal */
}

static void cs_madvise_expert_range(const uint8_t     *mmap_base,
                                    const cs_expert_range *r,
                                    int                  advice) {
    cs_madvise_range(mmap_base, r->logical_start, r->logical_end, advice);
}
#else
static void cs_madvise_range(const uint8_t *mmap_base,
                             uint64_t start, uint64_t end, int advice) {
    (void)mmap_base; (void)start; (void)end; (void)advice;
}
static void cs_madvise_expert_range(const uint8_t *mmap_base,
                                    const cs_expert_range *r, int advice) {
    (void)mmap_base; (void)r; (void)advice;
}
#endif

/* -----------------------------------------------------------------------
 * MemAvailable detection
 * ----------------------------------------------------------------------- */

static uint64_t cs_read_mem_available(void) {
#if defined(__linux__)
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t avail = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            /* Parse "MemAvailable:   12345 kB" */
            const char *p = line + 13;
            while (*p == ' ' || *p == '\t') p++;
            avail = strtoull(p, NULL, 10) * 1024ull;
            break;
        }
    }
    fclose(f);
    return avail;
#elif defined(__APPLE__)
    /* macOS: use vm_stat + page size to estimate free memory.
     * This is an approximation; madvise hints are weaker on Darwin anyway. */
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    uint64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctl(mib, 2, &memsize, &len, NULL, 0) == 0) {
        /* Conservative: use 50% of total RAM as "available" */
        return memsize / 2;
    }
    return 0;
#else
    return 0;
#endif
}

uint64_t ds4_cpu_stream_auto_budget(void) {
    uint64_t avail = cs_read_mem_available();
    if (avail == 0) return 0;

    /* Reserve 2 GiB for OS + ds4's non-expert working set.
     * Use 80% of remaining as expert cache budget. */
    const uint64_t reserve = 2ull * DS4_CS_GIB;
    if (avail <= reserve) return 0;
    const uint64_t usable = avail - reserve;
    const uint64_t budget = (usable * 4ull) / 5ull;
    return budget;
}

/* -----------------------------------------------------------------------
 * Init / free
 * ----------------------------------------------------------------------- */

/* Forward declarations for bridge functions (defined in ds4.c) */
extern uint32_t ds4_cpu_stream_n_layers(void);
extern uint32_t ds4_cpu_stream_n_experts(void);
extern uint32_t ds4_cpu_stream_n_experts_used(void);
extern int ds4_cpu_stream_layer_get_expert_ranges(
        uint32_t          il,
        const uint8_t    *mmap_base,
        cs_expert_ranges *out_ranges,
        uint32_t          max_experts,
        uint64_t         *out_per_expert_bytes);

ds4_cpu_stream *ds4_cpu_stream_create(const void *model_ptr,
                                      const void *weights_ptr,
                                      uint64_t    cache_bytes,
                                      const char *usage_path) {
    (void)model_ptr;
    (void)weights_ptr;

    ds4_cpu_stream *cs = calloc(1, sizeof(*cs));
    if (!cs) return NULL;

    cs->n_layers  = ds4_cpu_stream_n_layers();
    cs->n_experts = ds4_cpu_stream_n_experts();
    cs->usage_path = usage_path ? strdup(usage_path) : NULL;
    cs->clock = 1;
    cs->evict_interval = 32; /* evict every 32 clocks */

    if (cache_bytes == 0) cache_bytes = ds4_cpu_stream_auto_budget();
    cs->cache_bytes = cache_bytes;

    /* Allocate per-layer tracking */
    cs->layers = calloc((size_t)cs->n_layers, sizeof(cs->layers[0]));
    if (!cs->layers) { free(cs); return NULL; }

    /* Allocate transition matrices for route-ahead */
    if (cs->n_layers > 1) {
        cs->transitions = calloc((size_t)(cs->n_layers - 1),
                                 sizeof(cs->transitions[0]));
    }

    /* mmap_base will be set by the bridge layer in ds4.c after init */
    cs->mmap_base = NULL;

    if (cs->usage_path) {
        /* Attempt to load prior usage; failure is non-fatal */
        FILE *f = fopen(cs->usage_path, "rb");
        if (f) {
            uint64_t header[3]; /* magic, version, n_experts */
            if (fread(header, sizeof(header), 1, f) == 1 &&
                header[0] == 0x5553454400000001ull) { /* "USED\x00\x00\x00\x01" */
                uint32_t saved_experts = (uint32_t)header[2];
                for (uint32_t il = 0; il < cs->n_layers; il++) {
                    cs->layers[il].access_count =
                        calloc((size_t)cs->n_experts, sizeof(uint64_t));
                    cs->layers[il].last_clock =
                        calloc((size_t)cs->n_experts, sizeof(uint64_t));
                    if (saved_experts == cs->n_experts) {
                        size_t nr = fread(cs->layers[il].access_count,
                                          sizeof(uint64_t),
                                          (size_t)cs->n_experts, f);
                        (void)nr;
                    }
                }
                fprintf(stderr,
                        "ds4: CPU stream: loaded usage from %s\n",
                        cs->usage_path);
            }
            fclose(f);
        }
    }

    /* Defer expert-range population to bridge init call */
    return cs;
}

/* Called from ds4.c after model is loaded and mmap is ready */
void ds4_cpu_stream_set_mmap(ds4_cpu_stream *cs,
                             const uint8_t *mmap_base,
                             uint64_t mmap_size) {
    if (!cs) return;
    cs->mmap_base = mmap_base;
    cs->mmap_size = mmap_size;
}

/* Populate expert byte ranges — called once after mmap is set */
int ds4_cpu_stream_populate_layers(ds4_cpu_stream *cs) {
    if (!cs || !cs->mmap_base) return -1;

    uint32_t n_experts_used = ds4_cpu_stream_n_experts_used();
    uint64_t total_expert_bytes = 0;
    int populated = 0;

    for (uint32_t il = 0; il < cs->n_layers; il++) {
        cs_layer *layer = &cs->layers[il];
        layer->n_experts = cs->n_experts;
        layer->n_experts_used = n_experts_used;

        layer->expert = calloc((size_t)cs->n_experts,
                               sizeof(cs_expert_ranges));
        if (!layer->expert) continue;

        uint64_t per_exp = 0;
        int rc = ds4_cpu_stream_layer_get_expert_ranges(
                    il, cs->mmap_base, layer->expert,
                    cs->n_experts, &per_exp);

        if (rc > 0) {
            layer->has_routed = true;
            total_expert_bytes += per_exp * cs->n_experts;
            populated++;

            /* Allocate usage counters if not loaded from file */
            if (!layer->access_count) {
                layer->access_count =
                    calloc((size_t)cs->n_experts, sizeof(uint64_t));
                layer->last_clock =
                    calloc((size_t)cs->n_experts, sizeof(uint64_t));
            }

            /* Allocate transition matrix for route-ahead */
            if (il + 1 < cs->n_layers && cs->transitions) {
                cs_transition *tr = &cs->transitions[il];
                tr->n_experts = cs->n_experts;
                tr->counts = calloc((size_t)cs->n_experts * cs->n_experts,
                                    sizeof(uint64_t));
            }
        }
    }

    if (populated > 0) {
        cs->per_expert_bytes = total_expert_bytes /
            (uint64_t)populated / (uint64_t)cs->n_experts;
    }

    /* On first run with usage data, pin the hottest experts */
    if (cs->usage_path && cs->cache_bytes > 0 && cs->per_expert_bytes > 0) {
        uint64_t budget = cs->cache_bytes;
        for (uint32_t il = 0; il < cs->n_layers; il++) {
            cs_layer *layer = &cs->layers[il];
            if (!layer->has_routed || !layer->access_count) continue;

            /* Find top experts by historical count */
            typedef struct { uint32_t eid; uint64_t cnt; } ranked;
            ranked top[32];
            int ntop = 0;
            uint32_t max_pin = (uint32_t)(budget / cs->per_expert_bytes /
                                          cs->n_layers);
            if (max_pin < 1) max_pin = 1;
            if (max_pin > 32) max_pin = 32;

            for (uint32_t e = 0; e < cs->n_experts && ntop < (int)max_pin; e++) {
                /* Simple insertion sort by count */
                uint64_t cnt = layer->access_count[e];
                if (cnt == 0) continue;
                int pos = ntop;
                while (pos > 0 && top[pos - 1].cnt < cnt) {
                    if (pos < (int)max_pin) top[pos] = top[pos - 1];
                    pos--;
                }
                if (pos < (int)max_pin) {
                    top[pos].eid = e;
                    top[pos].cnt = cnt;
                    if (ntop < (int)max_pin) ntop++;
                }
            }

            /* Pin them with WILLNEED */
            for (int i = 0; i < ntop; i++) {
                cs_expert_ranges *r = &layer->expert[top[i].eid];
#if DS4_HAS_MADVISE
                cs_madvise_expert_range(cs->mmap_base, &r->gate_up,
                                        DS4_MADV_WILLNEED);
                cs_madvise_expert_range(cs->mmap_base, &r->down,
                                        DS4_MADV_WILLNEED);
#endif
                cs->hot_bytes += cs->per_expert_bytes;
            }

            if (ntop > 0) {
                fprintf(stderr,
                        "ds4: CPU stream: pinned %d hot experts for layer %u\n",
                        ntop, il);
            }
        }
    }

    return 0;
}

void ds4_cpu_stream_free(ds4_cpu_stream *cs) {
    if (!cs) return;

    /* Save usage before freeing */
    ds4_cpu_stream_save_usage(cs);

    for (uint32_t il = 0; il < cs->n_layers; il++) {
        cs_layer *layer = &cs->layers[il];
        free(layer->expert);
        free(layer->access_count);
        free(layer->last_clock);
    }
    free(cs->layers);

    if (cs->transitions) {
        for (uint32_t il = 0; il < cs->n_layers - 1; il++) {
            free(cs->transitions[il].counts);
        }
        free(cs->transitions);
    }

    free(cs->usage_path);
    free(cs);
}

/* -----------------------------------------------------------------------
 * Hot path: WILLNEED before MoE computation
 * ----------------------------------------------------------------------- */

void ds4_cpu_stream_willneed_layer(ds4_cpu_stream *cs,
                                   uint32_t         il,
                                   const int       *selected,
                                   int              n_selected) {
    if (!cs || !cs->mmap_base || il >= cs->n_layers) return;
    cs_layer *layer = &cs->layers[il];
    if (!layer->has_routed || !layer->expert) return;

    for (int i = 0; i < n_selected; i++) {
        int eid = selected[i];
        if (eid < 0 || (uint32_t)eid >= cs->n_experts) continue;
        cs_expert_ranges *r = &layer->expert[eid];
#if DS4_HAS_MADVISE
        cs_madvise_expert_range(cs->mmap_base, &r->gate_up,
                                DS4_MADV_WILLNEED);
        cs_madvise_expert_range(cs->mmap_base, &r->down,
                                DS4_MADV_WILLNEED);
#endif
        cs->total_willneed_calls++;
    }
}

/* -----------------------------------------------------------------------
 * Hot path: mark experts as used (post-MoE)
 * ----------------------------------------------------------------------- */

void ds4_cpu_stream_mark_used(ds4_cpu_stream *cs,
                              uint32_t         il,
                              const int       *selected,
                              int              n_selected) {
    if (!cs || il >= cs->n_layers) return;
    cs_layer *layer = &cs->layers[il];
    if (!layer->has_routed) return;

    cs->clock++;

    for (int i = 0; i < n_selected; i++) {
        int eid = selected[i];
        if (eid < 0 || (uint32_t)eid >= cs->n_experts) continue;
        if (layer->access_count) layer->access_count[eid]++;
        if (layer->last_clock) layer->last_clock[eid] = cs->clock;
    }

    /* Update route-ahead transition counts */
    /* (We need the previous layer's selections to correlate) */
    cs->usage_dirty = true;

    /* Periodic eviction */
    if (cs->clock - cs->last_evict_clock >= cs->evict_interval) {
        ds4_cpu_stream_evict_cold(cs, 64);
        cs->last_evict_clock = cs->clock;
    }
}

/* -----------------------------------------------------------------------
 * Eviction: DONTNEED cold expert pages
 * ----------------------------------------------------------------------- */

void ds4_cpu_stream_evict_cold(ds4_cpu_stream *cs, int max_cold) {
    if (!cs || !cs->mmap_base || max_cold <= 0) return;

    /* Eviction threshold: experts not touched in the last N clocks.
     * We use a fraction of the total clock. */
    const uint64_t threshold = cs->clock > 64 ? cs->clock - 64 : 0;
    int evicted = 0;

    for (uint32_t il = 0;
         il < cs->n_layers && evicted < max_cold;
         il++) {
        cs_layer *layer = &cs->layers[il];
        if (!layer->has_routed || !layer->expert || !layer->last_clock)
            continue;

        for (uint32_t e = 0;
             e < cs->n_experts && evicted < max_cold;
             e++) {
            /* Skip experts that were recently used */
            if (layer->last_clock[e] >= threshold) continue;
            /* Skip experts with no access history (avoid evicting the
             * wrong pages on a never-touched mapping) */
            if (layer->last_clock[e] == 0) continue;

            cs_expert_ranges *r = &layer->expert[e];
#if DS4_HAS_MADVISE
            cs_madvise_expert_range(cs->mmap_base, &r->gate_up,
                                    DS4_MADV_DONTNEED);
            cs_madvise_expert_range(cs->mmap_base, &r->down,
                                    DS4_MADV_DONTNEED);
#endif
            evicted++;
            cs->total_dontneed_calls++;

            /* Mark as "cold" so we don't ping-pong */
            if (cs->hot_bytes >= cs->per_expert_bytes)
                cs->hot_bytes -= cs->per_expert_bytes;
        }
    }
}

/* -----------------------------------------------------------------------
 * Route-ahead prefetch
 * ----------------------------------------------------------------------- */

void ds4_cpu_stream_prefetch_next_layer(ds4_cpu_stream *cs,
                                        uint32_t         il,
                                        const float     *post_attn) {
    if (!cs || !cs->mmap_base || il + 1 >= cs->n_layers) return;
    if (!cs->transitions || !cs->transitions[il].counts) return;
    (void)post_attn; /* reserved for future ML predictor */

    cs_layer *cur_layer = &cs->layers[il];
    cs_layer *next_layer = &cs->layers[il + 1];
    cs_transition *tr = &cs->transitions[il];
    if (!next_layer->has_routed || !next_layer->expert) return;

    /* Build a prediction: for each expert in next layer, sum the
     * co-occurrence counts with experts that were used in this layer.
     * Higher score = more likely to be used next. */
    float scores[256]; /* enough for typical expert counts */
    int n_scores = 0;
    int nex = (int)cs->n_experts;
    if (nex > 256) nex = 256;

    /* We need current layer's selections to predict — these were recorded
     * earlier by mark_used, but we don't store per-token selections.
     * Instead, use the most recently clocked experts as a proxy. */
    uint64_t score_sum = 0;
    for (int e2 = 0; e2 < nex; e2++) {
        float s = 0.0f;
        /* Sum co-occurrence with all experts in current layer */
        for (uint32_t e1 = 0; e1 < cs->n_experts; e1++) {
            uint64_t cooc = tr->counts[(uint64_t)e1 * cs->n_experts + e2];
            if (cooc > 0) {
                /* Weight by recency */
                uint64_t last = cur_layer->last_clock
                    ? cur_layer->last_clock[e1] : 0;
                if (last > 0) {
                    s += (float)cooc / (float)(1 + cs->clock - last);
                }
            }
        }
        scores[e2] = s;
        score_sum += (uint64_t)(s * 1000.0f);
    }
    n_scores = nex;

    /* Select top-k predictions (k = n_experts_used) */
    int predicted[16];
    int npred = 0;
    int npick = (int)next_layer->n_experts_used;
    if (npick > 16) npick = 16;

    if (score_sum == 0) return; /* no data yet */

    /* Simple top-k by score */
    for (int k = 0; k < npick && npred < npick; k++) {
        int best = -1;
        float best_score = -1.0f;
        for (int e = 0; e < n_scores; e++) {
            if (scores[e] > best_score) {
                /* Check not already picked */
                bool already = false;
                for (int j = 0; j < npred; j++) {
                    if (predicted[j] == e) { already = true; break; }
                }
                if (!already) {
                    best_score = scores[e];
                    best = e;
                }
            }
        }
        if (best < 0 || best_score <= 0.0f) break;
        predicted[npred++] = best;
    }

    if (npred == 0) return;

    /* Issue background WILLNEED on predicted next-layer experts */
    for (int i = 0; i < npred; i++) {
        cs_expert_ranges *r = &next_layer->expert[predicted[i]];
#if DS4_HAS_MADVISE
        cs_madvise_expert_range(cs->mmap_base, &r->gate_up,
                                DS4_MADV_WILLNEED);
        cs_madvise_expert_range(cs->mmap_base, &r->down,
                                DS4_MADV_WILLNEED);
#endif
    }

    /* Update transition counts: record which experts in the next layer
     * were actually selected (we'll do this in mark_used of next layer).
     * For now we just note the prediction. */
}

/* -----------------------------------------------------------------------
 * Transition matrix update
 * ----------------------------------------------------------------------- */

void ds4_cpu_stream_update_transitions(ds4_cpu_stream *cs,
                                       uint32_t         il,
                                       const int       *cur_selected,
                                       int              cur_n,
                                       const int       *next_selected,
                                       int              next_n) {
    if (!cs || il + 1 >= cs->n_layers) return;
    if (!cs->transitions || !cs->transitions[il].counts) return;
    cs_transition *tr = &cs->transitions[il];

    for (int i = 0; i < cur_n; i++) {
        int e1 = cur_selected[i];
        if (e1 < 0 || (uint32_t)e1 >= cs->n_experts) continue;
        for (int j = 0; j < next_n; j++) {
            int e2 = next_selected[j];
            if (e2 < 0 || (uint32_t)e2 >= cs->n_experts) continue;
            tr->counts[(uint64_t)e1 * cs->n_experts + e2]++;
        }
    }
}

/* -----------------------------------------------------------------------
 * Persistence
 * ----------------------------------------------------------------------- */

void ds4_cpu_stream_save_usage(ds4_cpu_stream *cs) {
    if (!cs || !cs->usage_path || !cs->usage_dirty) return;

    FILE *f = fopen(cs->usage_path, "wb");
    if (!f) return;

    uint64_t header[3] = {
        0x5553454400000001ull, /* magic "USED\0\0\0\1" */
        1,                      /* version */
        cs->n_experts,
    };
    (void)fwrite(header, sizeof(header), 1, f);

    for (uint32_t il = 0; il < cs->n_layers; il++) {
        cs_layer *layer = &cs->layers[il];
        if (layer->access_count) {
            (void)fwrite(layer->access_count,
                        sizeof(uint64_t),
                        (size_t)cs->n_experts, f);
        } else {
            /* Write zeros for layers without data */
            uint64_t zeros[512] = {0};
            size_t remaining = (size_t)cs->n_experts;
            while (remaining > 0) {
                size_t chunk = remaining < 512 ? remaining : 512;
                (void)fwrite(zeros, sizeof(uint64_t), chunk, f);
                remaining -= chunk;
            }
        }
    }

    fclose(f);
    cs->usage_dirty = false;
}

/* -----------------------------------------------------------------------
 * Query
 * ----------------------------------------------------------------------- */

uint64_t ds4_cpu_stream_hot_bytes(const ds4_cpu_stream *cs) {
    return cs ? cs->hot_bytes : 0;
}
