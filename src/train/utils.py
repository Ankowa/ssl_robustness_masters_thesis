def poly_lr_scheduler(  # from dino-v2 config files
    iter_num,
    total_iters,
    warmup_iters,
    warmup_ratio=1e-06,
    power=1.0,
):
    if iter_num < warmup_iters:
        factor = 1 - (iter_num / warmup_iters) * warmup_ratio
    else:
        factor = (1 - 1 / (total_iters - iter_num + 1)) ** power

    return factor
