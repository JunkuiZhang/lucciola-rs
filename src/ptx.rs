pub(crate) static ACTIVATION_PTX: &'static str =
    include_str!(concat!(env!("OUT_DIR"), "/activation.cu.ptx"));

pub(crate) static ATTENTION_PTX: &'static str =
    include_str!(concat!(env!("OUT_DIR"), "/attention.cu.ptx"));

pub(crate) static EMBEDDING_PTX: &'static str =
    include_str!(concat!(env!("OUT_DIR"), "/embedding.cu.ptx"));

pub(crate) static KV_CACHE_PTX: &'static str =
    include_str!(concat!(env!("OUT_DIR"), "/kv_cache.cu.ptx"));

pub(crate) static RMSNORM_PTX: &'static str =
    include_str!(concat!(env!("OUT_DIR"), "/rmsnorm.cu.ptx"));

pub(crate) static ROPE_PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/rope.cu.ptx"));

pub(crate) static SAMPLING_PTX: &'static str =
    include_str!(concat!(env!("OUT_DIR"), "/sampling.cu.ptx"));

pub(crate) static SORT_PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/sort.cu.ptx"));

pub(crate) static SCAN_SAMPLE_PTX: &'static str =
    include_str!(concat!(env!("OUT_DIR"), "/scan_sample.cu.ptx"));
