/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _BCACHEFS_BTREE_KEY_CACHE_H
#define _BCACHEFS_BTREE_KEY_CACHE_H

#include "btree/bkey.h"


struct alloc_request;

#ifndef __KERNEL__
bool bch2_test_key_cache_going_ro_abort_enabled(void);
bool bch2_test_key_cache_going_ro_abort_active(void);
int bch2_test_key_cache_going_ro_abort_dirty(struct bch_fs *, u64);
void bch2_test_key_cache_going_ro_abort_ready(struct bch_fs *);
void bch2_test_key_cache_going_ro_abort_split_attempted(void);
void bch2_test_key_cache_going_ro_abort_reserve_get_result(struct bch_fs *,
							    const struct alloc_request *, int);
#else
static inline bool bch2_test_key_cache_going_ro_abort_enabled(void)
{
	return false;
}

static inline bool bch2_test_key_cache_going_ro_abort_active(void)
{
	return false;
}

static inline void bch2_test_key_cache_going_ro_abort_ready(struct bch_fs *c)
{
}



static inline int
bch2_test_key_cache_going_ro_abort_dirty(struct bch_fs *c, u64 nr)
{
	return 0;
}
static inline void bch2_test_key_cache_going_ro_abort_split_attempted(void)
{
}

static inline void
bch2_test_key_cache_going_ro_abort_reserve_get_result(struct bch_fs *c,
						       const struct alloc_request *req,
						       int ret)
{
}
#endif
static inline size_t bch2_nr_btree_keys_need_flush(struct bch_fs *c)
{
	size_t nr_dirty = atomic_long_read(&c->btree.key_cache.nr_dirty);
	size_t nr_keys = atomic_long_read(&c->btree.key_cache.nr_keys);
	size_t max_dirty = 1024 + nr_keys  / 2;

	return max_t(ssize_t, 0, nr_dirty - max_dirty);
}

static inline ssize_t __bch2_btree_key_cache_must_wait(struct bch_fs *c)
{
	size_t nr_dirty = atomic_long_read(&c->btree.key_cache.nr_dirty);
	size_t nr_keys = atomic_long_read(&c->btree.key_cache.nr_keys);
	size_t max_dirty = 4096 + (nr_keys * 3) / 4;

	return nr_dirty - max_dirty;
}

static inline bool bch2_btree_key_cache_must_wait(struct bch_fs *c)
{
	return __bch2_btree_key_cache_must_wait(c) > 0;
}

static inline bool bch2_btree_key_cache_wait_done(struct bch_fs *c)
{
	size_t nr_dirty = atomic_long_read(&c->btree.key_cache.nr_dirty);
	size_t nr_keys = atomic_long_read(&c->btree.key_cache.nr_keys);
	size_t max_dirty = 2048 + (nr_keys * 5) / 8;

	return nr_dirty <= max_dirty;
}

int bch2_btree_key_cache_journal_flush(struct journal *,
				struct journal_entry_pin *, u64);
int bch2_btree_key_cache_flush_going_ro(struct bch_fs *);

static inline int bch2_btree_key_cache_cmp_fn(struct rhashtable_compare_arg *arg,
					      const void *obj)
{
	const struct bkey_cached *ck = obj;
	const struct bkey_cached_key *key = arg->key;

	return ck->key.btree_id != key->btree_id ||
		!bpos_eq(ck->key.pos, key->pos);
}

static const struct rhashtable_params bch2_btree_key_cache_params = {
	.head_offset		= offsetof(struct bkey_cached, hash),
	.key_offset		= offsetof(struct bkey_cached, key),
	.key_len		= sizeof(struct bkey_cached_key),
	.obj_cmpfn		= bch2_btree_key_cache_cmp_fn,
	.automatic_shrinking	= true,
};

__always_inline __flatten
static struct bkey_cached *
bch2_btree_key_cache_find(struct bch_fs *c, enum btree_id btree_id, struct bpos pos)
{
	struct bkey_cached_key key = {
		.btree_id	= btree_id,
		.pos		= pos,
	};

	return rhashtable_lookup_fast(&c->btree.key_cache.table, &key,
				      bch2_btree_key_cache_params);
}

int bch2_btree_path_traverse_cached(struct btree_trans *, btree_path_idx_t, unsigned);

void bch2_btree_insert_key_cached(struct btree_trans *, struct btree_insert_entry *);
void bch2_btree_key_cache_drop(struct btree_trans *,
			       struct btree_path *);

void bch2_fs_btree_key_cache_exit(struct bch_fs_btree_key_cache *);
int bch2_fs_btree_key_cache_init(struct bch_fs_btree_key_cache *);

void bch2_btree_key_cache_to_text(struct printbuf *, struct bch_fs_btree_key_cache *);

void bch2_btree_key_cache_exit(void);
int __init bch2_btree_key_cache_init(void);

#endif /* _BCACHEFS_BTREE_KEY_CACHE_H */
