pub(crate) static ACTIVATION_PTX: &'static str =
    include_str!(concat!(env!("OUT_DIR"), "/activation.cu.ptx"));
pub(crate) static KV_CACHE_PTX: &'static str =
    include_str!(concat!(env!("OUT_DIR"), "/kv_cache.cu.ptx"));
pub(crate) static RMSNORM_PTX: &'static str =
    include_str!(concat!(env!("OUT_DIR"), "/rmsnorm.cu.ptx"));
pub(crate) static ROPE_PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/rope.cu.ptx"));
pub(crate) static SOFTMAX_PTX: &'static str =
    include_str!(concat!(env!("OUT_DIR"), "/softmax.cu.ptx"));
