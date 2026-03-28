from config import num_perf_trials, num_warmup
import torch

def time_execution_event_template(context, device, synchronize, event_class, eval_target):
    get_inputs = context['get_inputs']
    get_init_inputs = context['get_init_inputs']
    generated_elapsed_times = []
    ModelNew = context[eval_target]
    inputs = get_inputs()
    inputs = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]
    init_inputs = get_init_inputs()
    init_inputs = [
        x.to(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]
    with torch.no_grad():
        custom_model = ModelNew(*init_inputs).to(device)
        def internel_eval(kernel_fn, elapsed_times):
            for _ in range(num_warmup):
                kernel_fn(*inputs)
                synchronize(device=device)
                # 对于使用workspace的操作符，需要额外的同步
                synchronize(device=device)
            for trail in range(num_perf_trials):
                start_event = event_class(enable_timing=True)
                end_event = event_class(enable_timing=True)
                start_event.record()
                kernel_fn(*inputs)
                end_event.record()
                # Synchronize to ensure the events have completed
                synchronize(device=device)
                # 对于使用workspace的操作符，需要额外的同步确保资源释放
                synchronize(device=device)
                # Calculate the elapsed time in milliseconds
                elapsed_time_ms = start_event.elapsed_time(end_event)
                elapsed_times.append(elapsed_time_ms)
        internel_eval(custom_model, generated_elapsed_times)
    return generated_elapsed_times


def time_execution_event_template_with_baseline(
    context,
    device,
    synchronize,
    event_class,
    eval_target="ModelNew",
    baseline_target="Model",
):
    """
    Time both generated op (eval_target, default: ModelNew) and baseline op
    (baseline_target, default: Model) using the same inputs / timing method.

    Returns:
        (generated_elapsed_times, baseline_elapsed_times_or_none)
    """
    get_inputs = context["get_inputs"]
    get_init_inputs = context["get_init_inputs"]

    generated_elapsed_times = []
    baseline_elapsed_times = []

    ModelNew = context.get(eval_target, None)
    ModelBaseline = context.get(baseline_target, None)

    inputs = get_inputs()
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

    init_inputs = get_init_inputs()
    init_inputs = [x.to(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

    def internel_eval(kernel_fn, elapsed_times):
        for _ in range(num_warmup):
            kernel_fn(*inputs)
            synchronize(device=device)
            # 对于使用workspace的操作符，需要额外的同步
            synchronize(device=device)
        for _ in range(num_perf_trials):
            start_event = event_class(enable_timing=True)
            end_event = event_class(enable_timing=True)
            start_event.record()
            kernel_fn(*inputs)
            end_event.record()
            # Synchronize to ensure the events have completed
            synchronize(device=device)
            # 对于使用workspace的操作符，需要额外的同步确保资源释放
            synchronize(device=device)
            elapsed_time_ms = start_event.elapsed_time(end_event)
            elapsed_times.append(elapsed_time_ms)

    with torch.no_grad():
        if ModelNew is None:
            raise KeyError(f"Missing {eval_target} in context")
        custom_model = ModelNew(*init_inputs).to(device)
        internel_eval(custom_model, generated_elapsed_times)

        # baseline is best-effort; on failure return None
        if ModelBaseline is None:
            return generated_elapsed_times, None
        try:
            baseline_model = ModelBaseline(*init_inputs).to(device)
            internel_eval(baseline_model, baseline_elapsed_times)
        except Exception:
            return generated_elapsed_times, None

    return generated_elapsed_times, baseline_elapsed_times
