#ifndef DS4_CPU_STREAM_H
#define DS4_CPU_STREAM_H

/* =========================================================================
 * ds4_cpu_stream.h — CPU-side expert streaming via madvise page-cache hints
 * =========================================================================
 *
 * Designed for machines where RAM is much smaller than the model (e.g. 32 GB
 * machine running an 81 GB GGUF).  Instead of relying on the kernel's
 * general-purpose page-cache LRU, we track expert hotness per layer and use
 * madvise(MADV_WILLNEED / MADV_DONTNEED) to steer which expert weights stay
 * resident.  This is inspired by colibri's explicit expert paging but hooks
 * into ds4's existing mmap-based CPU matmul path with zero change to the
 * matmul kernels themselves.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/* Expert byte-range descriptor within the mmap'd GGUF.
 * Populated by the bridge layer in ds4.c at init time. */
typedef struct {
    uint64_t logical_start;    /* unaligned byte offset of expert data */
    uint64_t logical_end;      /* unaligned byte offset (exclusive) */
} cs_expert_range;

typedef struct {
    cs_expert_range gate_up;   /* gate_proj + up_proj combined range */
    cs_expert_range down;      /* down_proj */
} cs_expert_ranges;

/* Opaque handle — everything lives behind this pointer so ds4.c never
 * needs to see the internal tracking tables. */
typedef struct ds4_cpu_stream ds4_cpu_stream;

/* -----------------------------------------------------------------------
 * Lifecycle
 * ----------------------------------------------------------------------- */

/* Create a CPU streaming manager.  `model` and `weights` are borrowed
 * (must outlive the manager).  `cache_bytes` is the RAM budget for expert
 * caching — pass 0 to auto-detect from MemAvailable.  `usage_path` is
 * where per-expert hotness is saved / loaded (NULL disables persistence). */
ds4_cpu_stream *ds4_cpu_stream_create(const void *model,
                                      const void *weights,
                                      uint64_t    cache_bytes,
                                      const char *usage_path);

void ds4_cpu_stream_free(ds4_cpu_stream *cs);

/* Called after model_open — provides the mmap base needed for madvise. */
void ds4_cpu_stream_set_mmap(ds4_cpu_stream *cs,
                             const uint8_t   *mmap_base,
                             uint64_t         mmap_size);

/* Populate per-layer expert byte ranges.  Must be called after
 * weights_bind and after set_mmap.  Returns 0 on success. */
int ds4_cpu_stream_populate_layers(ds4_cpu_stream *cs);

/* -----------------------------------------------------------------------
 * Per-token hot path (called from the decode loop)
 * ----------------------------------------------------------------------- */

/* Before computing MoE for layer `il`: issue MADV_WILLNEED on the page
 * ranges that hold the selected experts' gate/up/down tensors.  The kernel
 * will start async readahead so the data is likely resident by the time
 * the matmul kernels touch it. */
void ds4_cpu_stream_willneed_layer(ds4_cpu_stream *cs,
                                   uint32_t         il,
                                   const int       *selected,
                                   int              n_selected);

/* After computing MoE for layer `il`: record that these experts were used
 * so the next eviction pass can protect them. */
void ds4_cpu_stream_mark_used(ds4_cpu_stream *cs,
                              uint32_t         il,
                              const int       *selected,
                              int              n_selected);

/* -----------------------------------------------------------------------
 * Periodic maintenance (called once per N tokens, NOT every token)
 * ----------------------------------------------------------------------- */

/* Walk all layers and issue MADV_DONTNEED on expert page ranges that have
 * not been touched recently.  `max_cold` caps the number of experts
 * evicted in one pass to avoid jitter. */
void ds4_cpu_stream_evict_cold(ds4_cpu_stream *cs, int max_cold);

/* -----------------------------------------------------------------------
 * Route-ahead prefetch (optional, improves hit rate)
 * ----------------------------------------------------------------------- */

/* Use layer `il`'s post-attention hidden state to predict which experts
 * layer `il+1` will route to, then issue background MADV_WILLNEED so the
 * next layer's weights are ready.  `post_attn` points to the HC state
 * after attention (DS4_N_HC * DS4_N_EMBD floats).
 *
 * If no predictor is available, this falls back to a correlation heuristic:
 * "experts that co-occur in layer il+1 when layer il selected expert X".
 */
void ds4_cpu_stream_prefetch_next_layer(ds4_cpu_stream *cs,
                                        uint32_t         il,
                                        const float     *post_attn);

/* Transition matrix update: record that when experts `cur_selected` were
 * used in layer `il`, experts `next_selected` were used in layer `il+1`.
 * This builds the co-occurrence table used by route-ahead prefetch. */
void ds4_cpu_stream_update_transitions(ds4_cpu_stream *cs,
                                       uint32_t         il,
                                       const int       *cur_selected,
                                       int              cur_n,
                                       const int       *next_selected,
                                       int              next_n);

/* -----------------------------------------------------------------------
 * Persistence
 * ----------------------------------------------------------------------- */

/* Flush usage counters to disk so the next run can pin hot experts early. */
void ds4_cpu_stream_save_usage(ds4_cpu_stream *cs);

/* -----------------------------------------------------------------------
 * Utilities
 * ----------------------------------------------------------------------- */

/* Return a reasonable expert-cache budget (bytes) given the system's free
 * RAM.  Returns 0 if the system is too constrained. */
uint64_t ds4_cpu_stream_auto_budget(void);

/* Return the number of bytes currently tracked as "hot" (WILLNEED'd). */
uint64_t ds4_cpu_stream_hot_bytes(const ds4_cpu_stream *cs);

#endif /* DS4_CPU_STREAM_H */
